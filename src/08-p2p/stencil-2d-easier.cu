#include "../stencil-2d-util.h"

#include "../cuda-util.h"


__global__ void stencil2D(const double *__restrict__ u, double *__restrict__ uNew,
                          size_t localBeginInnerX, size_t localEndInnerX,
                          size_t localBeginInnerY, size_t localEndInnerY,
                          size_t localNumCellsX) {

    const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tidy = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t i0 = localBeginInnerX + tidx;
    const size_t i1 = localBeginInnerY + tidy;

    if (i0 < localEndInnerX && i1 < localEndInnerY) {
        uNew[i0 + i1 * localNumCellsX] = u[i0 + i1 * localNumCellsX]
            + alpha * (
                      u[(i0 - 1) +  i1      * localNumCellsX]
                +     u[(i0 + 1) +  i1      * localNumCellsX]
                +     u[ i0      + (i1 + 1) * localNumCellsX]
                +     u[ i0      + (i1 - 1) * localNumCellsX]
                - 4 * u[ i0      +  i1      * localNumCellsX]);
    }
}


struct Patch {
    // patch boundaries in global coordinates excluding boundaries
    size_t globalInnerBeginX;
    size_t globalInnerEndX;
    size_t globalInnerBeginY;
    size_t globalInnerEndY;

    // patch local extents including halos/ boundaries
    size_t localNumCellsX;
    size_t localNumCellsY;
    size_t localSize;          // in bytes

    // pointers to the GPU allocation
    double* d_localU;
    double* d_localUNew;

    // execution configuration
    dim3 blockSize;
    dim3 gridSize;

    // additional stream for async copies
    cudaStream_t stream;
};


int main(int argc, char *argv[]) {
    // determine application parameters
    size_t globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval;
    parseCLA_2d(argc, argv, globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval);

    int numDevices = 0;
    checkCudaError(cudaGetDeviceCount(&numDevices));
    std::cout << "Using " << numDevices << " GPUs" << std::endl;

    // enable P2P access
    for (int srcDeviceIdx = 0; srcDeviceIdx < numDevices; ++srcDeviceIdx) {
        checkCudaError(cudaSetDevice(srcDeviceIdx));
        for (int dstDeviceIdx = 0; dstDeviceIdx < numDevices; ++dstDeviceIdx) {
            if (TODO != TODO) {
                int canAccessPeer = 0;
                checkCudaError(cudaDeviceCanAccessPeer(&canAccessPeer, srcDeviceIdx, dstDeviceIdx));
                if (canAccessPeer) {
                    checkCudaError(cudaDeviceEnablePeerAccess(TODO, 0));
                } else {
                    std::cerr << "Warning: Device " << srcDeviceIdx << " cannot access peer device " << dstDeviceIdx << std::endl;
                }
            }
        }
    }

    // initialize patches
    int numPatches = numDevices; // one patch per device

    Patch *patches = new Patch[numPatches];

    size_t patchHeight = ceilingDivide(globalNumCellsY - 2, numPatches);
    for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx) {
        checkCudaError(cudaSetDevice(patchIdx));

        auto &patch = patches[patchIdx];

        // no partitioning in the x-dimension
        patch.globalInnerBeginX = 1;
        patch.globalInnerEndX = globalNumCellsX - 1;

        patch.globalInnerBeginY = 1 + patchIdx * patchHeight;
        patch.globalInnerEndY   = std::min(     // the end is either
            1 + (patchIdx + 1) * patchHeight,   // the beginning of the next patch, or
            globalNumCellsY - 1);               // the end of the global domain

        // local extents including halos
        patch.localNumCellsX = patch.globalInnerEndX - patch.globalInnerBeginX + 2;   // two halo layers of size one each
        patch.localNumCellsY = patch.globalInnerEndY - patch.globalInnerBeginY + 2;
        patch.localSize = patch.localNumCellsX * patch.localNumCellsY * sizeof(double);

        // execution configuration
        auto numInnerCellsX = patch.globalInnerEndX - patch.globalInnerBeginX;
        auto numInnerCellsY = patch.globalInnerEndY - patch.globalInnerBeginY;

        patch.blockSize = dim3(16, 16);
        patch.gridSize  = dim3(
            ceilingDivide(numInnerCellsX, patch.blockSize.x),
            ceilingDivide(numInnerCellsY, patch.blockSize.y));

        // stream
        checkCudaError(cudaStreamCreate(&patch.stream));
    }

    // allocate CPU
    double *u;
    checkCudaError(cudaMallocHost(&u, globalNumCellsX * globalNumCellsY * sizeof(double)));

    // allocate GPU per patch
    for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx) {
        checkCudaError(cudaSetDevice(patchIdx));

        auto &patch = patches[patchIdx];

        checkCudaError(cudaMalloc((void **)&patch.d_localU, patch.localSize));
        checkCudaError(cudaMalloc((void **)&patch.d_localUNew, patch.localSize));
    }

    // init temperature fields including their boundaries
    initTemperature(u, globalNumCellsX, globalNumCellsY);

    // copy data to GPU
    for (int deviceIdx = 0; deviceIdx < numDevices; ++deviceIdx) {
        checkCudaError(cudaSetDevice(deviceIdx));

        const auto &patch = patches[deviceIdx];

        auto sourceIdx = (patch.globalInnerBeginY - 1) * globalNumCellsX;   // include halo row
        checkCudaError(cudaMemcpy(
            patch.d_localU,                 // destination - local to the patch
            &u[sourceIdx],                  // source - one large host array
            patch.localSize,                // size including halos
            cudaMemcpyHostToDevice));       // direction

        // initialize uNew by copying from u
        checkCudaError(cudaMemcpy(patch.d_localUNew, patch.d_localU, patch.localSize, cudaMemcpyDeviceToDevice));
    }

    // define print and work
    auto gatherTemperature = [&]() {
        // gather data from all patches
        // Note: this could be optimized - see the course 'Fundamentals of Accelerated Computing with Modern CUDA C++'
        for (int deviceIdx = 0; deviceIdx < numDevices; ++deviceIdx) {
            checkCudaError(cudaSetDevice(deviceIdx));

            const auto &patch = patches[deviceIdx];

            checkCudaError(cudaMemcpy(
                &u[patch.globalInnerBeginY * globalNumCellsX],                      // destination - a slice in the global host array; skip halo row
                &patch.d_localU[1 * patch.localNumCellsX],                          // source - local to the patch; skip top halo
                (patch.localNumCellsY - 2) * patch.localNumCellsX * sizeof(double), // size excluding top and bottom halo rows (but including left/right halos)
                cudaMemcpyDeviceToHost));                                           // direction
        }
    };

    auto print = [&](size_t it) {
        std::cout << "  Completed iteration " << it << std::endl;

        std::string idx = std::to_string(it);
        if (idx.size() < 6) idx = std::string(6 - idx.size(), '0') + idx;

        gatherTemperature();

        writeTemperatureNpy("../output/temperature_" + idx + ".npy", u, globalNumCellsX, globalNumCellsY);
    };

    auto haloExchange = [&](int deviceIdx) {
        // exchange halos between patches
        const auto &patch = patches[deviceIdx];

        // push data to lower neighbor
        if (deviceIdx > 0) {
            const auto &lowerPatch = patches[deviceIdx - 1];

            // top halo of lower neighbor
            auto destPtr = &lowerPatch.d_localUNew[(lowerPatch.localNumCellsY - 1) * lowerPatch.localNumCellsX];
            // first inner row of current patch
            auto srcPtr  = &patch.d_localUNew[1 * patch.localNumCellsX];

            auto size = patch.localNumCellsX * sizeof(double);  // size of one row

            checkCudaError(cudaMemcpyAsync(destPtr, srcPtr, size, cudaMemcpyDeviceToDevice, patch.stream));
        }

        // push data to upper neighbor
        if (deviceIdx < numDevices - 1) {
            const auto &upperPatch = patches[deviceIdx + 1];

            // bottom halo of upper neighbor
            auto destPtr = &upperPatch.d_localUNew[0 * upperPatch.localNumCellsX];
            // last inner row of current patch
            auto srcPtr  = &patch.d_localUNew[(patch.localNumCellsY - 2) * patch.localNumCellsX];

            auto size = patch.localNumCellsX * sizeof(double);  // size of one row

            checkCudaError(cudaMemcpyAsync(destPtr, srcPtr, size, cudaMemcpyDeviceToDevice, patch.stream));
        }
    };

    auto work = [&](size_t it) {
        for (int deviceIdx = 0; deviceIdx < numDevices; ++deviceIdx) {
            checkCudaError(cudaSetDevice(deviceIdx));

            const auto &patch = patches[deviceIdx];

            stencil2D<<<patch.gridSize, patch.blockSize, 0, patch.stream>>>(
                patch.d_localU, patch.d_localUNew,                  // local pointers
                1, patch.localNumCellsX - 1,                        // local x interval
                1, patch.localNumCellsY - 1,                        // local y interval
                patch.localNumCellsX                                // stride for index linearization
            );

            // no synchronization necessary here due to stream mechanics

            haloExchange(deviceIdx);
        }

        for (int deviceIdx = 0; deviceIdx < numDevices; ++deviceIdx) {
            checkCudaError(cudaSetDevice(deviceIdx));

            const auto &patch = patches[deviceIdx];

            checkCudaError(cudaStreamSynchronize(patch.stream));
        }

        for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx) {
            auto &patch = patches[patchIdx];
            std::swap(patch.d_localU, patch.d_localUNew);
        }

        if (printInterval > 0 && 0 == (it % printInterval))
            print(it);
    };

    // warm-up
    for (size_t i = 0; i < numItWarmUp; ++i)
        work(i);

    // measurement
    for (int deviceIdx = 0; deviceIdx < numDevices; ++deviceIdx) {
        checkCudaError(cudaSetDevice(deviceIdx));
        checkCudaError(cudaDeviceSynchronize());
    }
    nvtxRangePushA("work");
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < numItTimed; ++i)
            work(i + numItWarmUp);    // account for warm-up iterations in the print interval computation

    for (int deviceIdx = 0; deviceIdx < numDevices; ++deviceIdx) {
        checkCudaError(cudaSetDevice(deviceIdx));
        checkCudaError(cudaDeviceSynchronize());
    }
    auto end = std::chrono::steady_clock::now();
    nvtxRangePop();

    // print stats and diagnostic result
    printStats(end - start, numItTimed, globalNumCellsX * globalNumCellsY, sizeof(double) + sizeof(double), 7);

    gatherTemperature();
    auto totalTemperature = accumulateTemperature(u, globalNumCellsX, globalNumCellsY);
    std::cout << "  Total temperature is " << totalTemperature << std::endl;

    // clean up
    for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx) {
        checkCudaError(cudaSetDevice(patchIdx));

        const auto &patch = patches[patchIdx];

        checkCudaError(cudaStreamDestroy(patch.stream));

        checkCudaError(cudaFree(patch.d_localU));
        checkCudaError(cudaFree(patch.d_localUNew));
    }

    checkCudaError(cudaFreeHost(u));

    delete[] patches;

    return 0;
}
