// index bindings of bipartite_index for python

#include <omp.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <filesystem>
#include <unistd.h>

#include "index.h"
#include "index_factory.h"

namespace py = pybind11;

class FilterDiskANN
{
  public:
    FilterDiskANN(const diskann::Metric &m, const std::string &index_prefix, const size_t &num_points,
                  const size_t &dimensions, const uint32_t &num_threads, const uint32_t &L)
    {
        auto index_search_params = diskann::IndexSearchParams(L, omp_get_num_procs());

        _index = new diskann::Index<uint8_t>(
            m, dimensions, num_points,
            nullptr,                                                           // index write params
            std::make_shared<diskann::IndexSearchParams>(index_search_params), // index search params
            0,                                                                 // num frozen points
            false,                                                             // not a dynamic_index
            false,                                                             // no enable_tags/ids
            false,                                                             // no concurrent_consolidate,
            false,                                                             // pq_dist_build
            0,                                                                 // num_pq_chunks
            false);                                                            // use_opq = false
        _index->load(index_prefix.c_str(), omp_get_num_procs(), L);
        std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    }

    ~FilterDiskANN()
    {
        delete _index;
    }

    void Search(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &queries_input,
                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> &query_filters_input, const size_t num_queries,
                const size_t knn, const uint32_t L, const uint32_t num_threads,
                py::array_t<uint32_t, py::array::c_style> &res_id)
    {

        auto queries = queries_input.unchecked();
        auto query_filters_data = query_filters_input.unchecked();
        auto res = res_id.mutable_unchecked();

        std::vector<uint32_t> ids;
        std::vector<uint8_t> one_query;

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)num_queries; i++)
        {
            std::vector<std::string> filters;
            for(size_t j = 0; j<query_filters_data.shape(i);j++){
                filters.emplace_back(std::to_string(query_filters_data(i,j)));
            }
            _index->search_with_multi_filters(queries.data(i), filters, knn, L, &res(i, 0), nullptr);
            // continue;
            // _index->search_with_multi_filters(one_query.data(), filters, knn, L, ids.data(), nullptr);
            //  index_bindings.cpp:73:46: error: no matching function for call to ‘diskann::Index<unsigned char,
            //  unsigned int, unsigned int>::search_with_multi_filters(const unsigned char*,
            //  std::vector<std::__cxx11::basic_string<char> >&, const uint64_t&, const uint64_t&, std::vector<unsigned
            //  int>&, std::nullptr_t)’
            // index_bindings.cpp:74:46: error: no matching function for call to ‘diskann::Index<unsigned char, unsigned
            // int, unsigned int>::search_with_multi_filters(std::vector<unsigned char>&,
            // std::vector<std::__cxx11::basic_string<char> >&, const uint64_t&, const uint64_t&, std::vector<unsigned
            // int>&, std::nullptr_t)’
        }
        // search_with_multi_filters(const DataType &query, const std::vector<std::string> &query_filters,
        //                                                     const size_t K, const uint32_t L, IndexType *indices,
        //                                                     float *distances);
        // std::cout << "finish in cpp"  << std::endl;
    }

  private:
    diskann::Index<uint8_t, uint32_t, uint32_t> *_index;
};

// void Build(const std::string &data_path, const std::string &index_path_prefix, const std::string &label_file,
//            const uint32_t &num_threads, const uint32_t &R, const uint32_t &L, const float &alpha)
// {

//     std::string data_type = "uint8";
//     std::string label_type = "uint";
//     diskann::Metric metric = diskann::Metric::L2;

//     size_t data_num, data_dim;
//     diskann::get_bin_metadata(data_path, data_num, data_dim);

//     auto config = diskann::IndexConfigBuilder()
//                       .with_metric(metric)
//                       .with_dimension(data_dim)
//                       .with_max_points(data_num)
//                       .with_data_load_store_strategy(diskann::MEMORY)
//                       .with_data_type(data_type)
//                       .with_label_type(label_type)
//                       .is_dynamic_index(false)
//                       .is_enable_tags(false)
//                       .is_use_opq(false)
//                       .is_pq_dist_build(false)
//                       .with_num_pq_chunks(0)
//                       .build();

//     auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
//                                   .with_filter_list_size(0) // 过滤的L
//                                   .with_alpha(alpha)
//                                   .with_saturate_graph(false) // 是否为饱和图，即每个node的邻居要==R
//                                   .with_num_threads(num_threads)
//                                   .build();

//     auto build_params = diskann::IndexBuildParamsBuilder(index_build_params)
//                             .with_universal_label("")
//                             .with_label_file(label_file)
//                             .with_save_path_prefix(index_path_prefix)
//                             .build();
//     auto index_factory = diskann::IndexFactory(config);
//     auto index = index_factory.create_instance();
//     index->build(data_path, data_num, build_params, label_file); // 调用的是Index类的build函数
//     index->save(index_path_prefix.c_str(), label_file); // 存了graph和data，search的时候要加载这两个
// }

PYBIND11_MODULE(filterdiskann, m)
{
    m.doc() = "pybind11 filterdiskann plugin"; // optional module docstring
    // enumerate...
    py::enum_<diskann::Metric>(m, "Metric")
        .value("L2", diskann::Metric::L2)
        .value("IP", diskann::Metric::INNER_PRODUCT)
        .value("COSINE", diskann::Metric::COSINE)
        .export_values();
    py::class_<FilterDiskANN>(m, "FilterDiskANN")
        .def(py::init<const diskann::Metric &, const std::string &, const size_t &, const size_t &, const uint32_t &,
                      const uint32_t &>())
        .def("search", &FilterDiskANN::Search);
    // m.def("build", &Build);
}