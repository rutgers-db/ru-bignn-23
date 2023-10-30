// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "filter_utils.h"

namespace po = boost::program_options;
using std::cout;
using std::endl;

int main(int argc, char *argv[])
{

    int64_t *row_index = nullptr;
    int32_t *col_index = nullptr;
    float *filter_value = nullptr;
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    // /home/ubuntu/big-ann-benchmarks/data/random-filter100000/data_metadata_100000_50
    cout << "transfer label file :" << argv[1] << " to " << argv[2] << endl;
    read_sparse_matrix(argv[1], rows, cols, nnz, row_index, col_index, filter_value);
    std::cout << "Matrix size: (" << rows << ", " << cols << "), non-zeros elements: " << nnz << std::endl;

    // std::string save_to_label_file_path = "/home/ubuntu/label_file_base_yfcc10m.txt";
    write_labels(argv[2], row_index, col_index, rows);

    delete[] row_index;
    delete[] col_index;
    delete[] filter_value;

    return 0;
}