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

int main()
{

    int64_t *row_index = nullptr;
    int32_t *col_index = nullptr;
    float *filter_value = nullptr;
    int64_t rows;
    int64_t cols;
    int64_t nnz;

    read_sparse_matrix("/home/ubuntu/big-ann-benchmarks/data/yfcc100M/query.metadata.public.100K.spmat", rows, cols,
                       nnz, row_index, col_index, filter_value);
    std::cout << "Matrix size: (" << rows << ", " << cols << "), non-zeros elements: " << nnz << std::endl;

    std::vector<std::vector<std::string>> filters;
    load_sparse_matrix("/home/ubuntu/big-ann-benchmarks/data/yfcc100M/query.metadata.public.100K.spmat", filters);

    cout << filters.size() << endl;
    for (int i = 0; i < 2; i++)
    {
        for (auto ele : filters.at(i))
        {
            cout << ele << ",";
        }
        cout << endl;
    }

    delete[] row_index;
    delete[] col_index;
    delete[] filter_value;

    return 0;
}