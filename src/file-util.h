#pragma once

#include "util.h"

void writeNpyHeader(const std::string &filename, size_t globalNumCellsX, size_t globalNumCellsY) {
    // prep header content
    std::ostringstream header;

    // f8 > double, fortran_order == False > row-major, shape = (globalNumCellsY, globalNumCellsX)
    header << "{'descr': '<f8', 'fortran_order': False, 'shape': (" << globalNumCellsY << ", " << globalNumCellsX << "), }";

    // pad header to be multiple of 16 bytes
    size_t hlen = header.str().size();
    size_t pad = 16 - ((10 + hlen) % 16);
    for (size_t i = 0; i < pad; ++i)
        header << ' ';

    // truncate file (clear existing content) and write header in binary mode
    std::ofstream outfile(filename, std::ios::binary | std::ios::out | std::ios::trunc);

    // write magic string
    const char magic[] = "\x93NUMPY";
    outfile.write(magic, 6);

    // write version number
    unsigned char ver[2] = {1, 0};
    outfile.write((char*)ver, 2);

    // write header length
    uint16_t hlen_le = static_cast<uint16_t>(header.str().size());
    outfile.write((char*)&hlen_le, 2);

    // write header content
    outfile.write(header.str().data(), header.str().size());
}

void appendNpyData(const std::string &filename, const double *data, size_t globalNumCellsX, size_t globalNumCellsY) {
    // open file in binary append mode
    std::ofstream outfile(filename,  std::ios::binary | std::ios::out | std::ios::app);

    // write data
    outfile.write((char*)data, globalNumCellsX * globalNumCellsY * sizeof(double));
}

void writeTemperatureNpy(const std::string &filename, const double *data, size_t globalNumCellsX, size_t globalNumCellsY) {
    writeNpyHeader(filename, globalNumCellsX, globalNumCellsY);
    appendNpyData(filename, data, globalNumCellsX, globalNumCellsY);
}

void writeTemperaturePatchNpy(const std::string &filename, const double *data,
                              size_t globalNx, size_t globalNy, size_t localNumCellsX, size_t localNumCellsY,
                              int numPatches, int patchIdx) {

    if (0 == patchIdx)
        writeNpyHeader(filename, globalNx, globalNy);

    // adapt offsets to avoid writing halos multiple times
    size_t startRowSkip = (patchIdx > 0) ? 1 : 0;
    size_t rowsToWrite = (patchIdx < numPatches - 1) ? localNumCellsY - 1 : localNumCellsY;
    rowsToWrite -= startRowSkip;

    appendNpyData(filename, data + startRowSkip * globalNx, localNumCellsX, rowsToWrite);
}
