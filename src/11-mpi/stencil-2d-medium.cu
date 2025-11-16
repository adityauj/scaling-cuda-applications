#include "../stencil-2d-util.h"

#include "../cuda-util.h"

#include <mpi.h>


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

    // pointers to CPU allocation
    double* localU;

    // pointers to the GPU allocation
    double* d_localU;
    double* d_localUNew;

    // execution configuration
    dim3 blockSize;
    dim3 gridSize;
};


int main(int argc, char *argv[]) {
    // initialize MPI
    TODO: initialize MPI

    int numRanks = 0;
    TODO
    int rank = 0;
    TODO

    // determine application parameters
    size_t globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval;
    parseCLA_2d(argc, argv, globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval);

    // choose GPU
    int numDevicesPerNode = 0;
    checkCudaError(cudaGetDeviceCount(&numDevicesPerNode));

    int deviceId = TODO;
    checkCudaError(cudaSetDevice(deviceId));

    std::cout << "Rank " << rank << " using device " << deviceId << std::endl;

    // initialize patches
    int numPatches = TODO;  // one patch per rank
    checkCudaError(cudaSetDevice(deviceId));

    Patch patch;                // each rank has only one patch
    int patchIdx = rank;

    size_t patchHeight = ceilingDivide(globalNumCellsY - 2, numPatches);

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

    // allocate CPU
    checkCudaError(cudaMallocHost(&patch.localU, patch.localSize));

    // allocate GPU
    checkCudaError(cudaMalloc((void **)&patch.d_localU, patch.localSize));
    checkCudaError(cudaMalloc((void **)&patch.d_localUNew, patch.localSize));

    // init temperature fields including their boundaries
    initTemperaturePatch(patch.localU,
        TODO,       // global index space
        TODO,       // local index space
        TODO        // global offset for this patch
    );

    // copy data to GPU
    checkCudaError(cudaMemcpy(patch.d_localU, patch.localU, patch.localSize, cudaMemcpyHostToDevice));

    // initialize uNew by copying from u
    checkCudaError(cudaMemcpy(patch.d_localUNew, patch.d_localU, patch.localSize, cudaMemcpyDeviceToDevice));

    // define print and work
    auto print = [&](size_t it) {
        std::cout << "  Completed iteration " << it << std::endl;

        std::string idx = std::to_string(it);
        if (idx.size() < 6) idx = std::string(6 - idx.size(), '0') + idx;

        // all ranks copy data back to CPU in parallel
        checkCudaError(cudaMemcpy(patch.localU, patch.d_localU, patch.localSize, cudaMemcpyDeviceToHost));

        // Note: brute-force full-synchronize approach for simplicity - optimize with point-to-point messages triggering next write
        for (int printRank = 0; printRank < numRanks; ++printRank) {
            MPI_Barrier(MPI_COMM_WORLD);    // make sure all ranks are in the same iteration

            if (rank == printRank) {        // only one rank per loop iteration is allowed to write
                writeTemperaturePatchNpy("../output/temperature_" + idx + ".npy",
                    patch.localU,
                    globalNumCellsY, globalNumCellsX, patch.localNumCellsX, patch.localNumCellsY,
                    numPatches, rank);
            }
        }
    };

    // halo exchange
    auto haloExchange = [&]() {
        MPI_Request requests[4] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL };

        // exchange data with lower neighbor
        if (rank > 0) {
            auto recvPtr = TODO;    // bottom halo of current patch
            auto sendPtr = TODO;    // first inner row of current patch
            MPI_Irecv(
                recvPtr,                // destination pointer
                patch.localNumCellsX,   // number of elements: one row
                MPI_DOUBLE,             // datatype
                rank - 1,               // source rank
                0,                      // tag
                MPI_COMM_WORLD,         // communicator
                &requests[0]);          // request handle - req for wait later
            MPI_Isend(
                sendPtr,                // source pointer
                patch.localNumCellsX,   // number of elements: one row
                MPI_DOUBLE,             // datatype
                rank - 1,               // destination rank
                0,                      // tag
                MPI_COMM_WORLD,         // communicator
                &requests[1]);          // request handle - req for wait later
        }

        // exchange data with upper neighbor
        if (rank < numRanks - 1) {
            auto recvPtr = TODO;   // top halo of current patch
            auto sendPtr = TODO;   // last inner row of current patch
            MPI_Irecv(
                recvPtr,                // destination pointer
                patch.localNumCellsX,   // number of elements: one row
                MPI_DOUBLE,             // datatype
                rank + 1,               // source rank
                0,                      // tag
                MPI_COMM_WORLD,         // communicator
                &requests[2]);          // request handle - req for wait later
            MPI_Isend(
                sendPtr,                // source pointer
                patch.localNumCellsX,   // number of elements: one row
                MPI_DOUBLE,             // datatype
                rank + 1,               // destination rank
                0,                      // tag
                MPI_COMM_WORLD,         // communicator
                &requests[3]);          // request handle - req for wait later
        }

        // wait for all communications to complete
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    };
    
    auto work = [&](size_t it) {
        stencil2D<<<patch.gridSize, patch.blockSize>>>(TODO);

        checkCudaError(cudaDeviceSynchronize());

        haloExchange();

        std::swap(patch.d_localU, patch.d_localUNew);

        if (printInterval > 0 && 0 == (it % printInterval))
            print(it);
    };

    // warm-up
    for (size_t i = 0; i < numItWarmUp; ++i)
        work(i);

    // measurement
    checkCudaError(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePushA("work");
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < numItTimed; ++i)
            work(i + numItWarmUp);    // account for warm-up iterations in the print interval computation

    checkCudaError(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    nvtxRangePop();

    // print stats and diagnostic result
    if (0 == rank)
        printStats(end - start, numItTimed, globalNumCellsX * globalNumCellsY, sizeof(double) + sizeof(double), 7);

    checkCudaError(cudaMemcpy(patch.localU, patch.d_localU, patch.localSize, cudaMemcpyDeviceToHost));

    auto rankTotalTemperature = accumulateTemperature(patch.localU, patch.localNumCellsX, patch.localNumCellsY);
    std::cout << "  Total temperature on rank " << rank << " is " << rankTotalTemperature << std::endl;
    double totalTemperature = 0.0;
    MPI_Reduce(
        &rankTotalTemperature,  // send buffer
        &totalTemperature,      // receive buffer
        1,                      // count
        MPI_DOUBLE,             // datatype
        MPI_SUM,                // operation
        0,                      // root
        MPI_COMM_WORLD);        // communicator
    if (0 == rank)
        std::cout << "  Total temperature is " << totalTemperature << std::endl;

    // clean up
    checkCudaError(cudaFree(patch.d_localU));
    checkCudaError(cudaFree(patch.d_localUNew));

    checkCudaError(cudaFreeHost(patch.localU));

    TODO: finalize MPI

    return 0;
}
