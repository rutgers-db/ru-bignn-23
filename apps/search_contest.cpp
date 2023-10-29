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

#include "index.h"
#include "memory_mapper.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"

namespace po = boost::program_options;

template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &query_file,
                        const uint32_t num_threads, const uint32_t recall_at, const std::vector<uint32_t> &Lvec,
                        const std::string &query_filter_file)
{
    using TagT = uint32_t;
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    std::vector<std::vector<std::string>> query_filters;
    std::string line, token;
    std::ifstream filter_reader(query_filter_file);
    while (std::getline(filter_reader, line))
    {
        std::istringstream line_reader(line);
        std::vector<std::string> filters;
        while (std::getline(line_reader, token, ','))
        {
            filters.push_back(token);
        }
        query_filters.push_back(filters);
    }
    assert(query_filters.size() == query_num);

    bool calc_recall_flag = false;

    bool filtered_search = true;

    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);

    std::cout << "num_frozen_pts:" << num_frozen_pts << std::endl;

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(query_dim)
                      .with_max_points(0)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(diskann_type_to_name<T>())
                      .with_label_type(diskann_type_to_name<LabelT>())
                      .with_tag_type(diskann_type_to_name<TagT>())
                      .is_dynamic_index(false)
                      .is_enable_tags(false)
                      .is_concurrent_consolidate(false)
                      .is_pq_dist_build(false)
                      .is_use_opq(false)
                      .with_num_pq_chunks(0)
                      .with_num_frozen_pts(num_frozen_pts)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    std::cout << "Index loaded" << std::endl;

    if (metric == diskann::FAST_L2)
        index->optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats;

    cmp_stats = std::vector<uint32_t>(query_num, 0);

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            auto qs = std::chrono::high_resolution_clock::now();
            if (filtered_search)
            {
                std::vector<std::string> raw_filter = query_filters[i];
                auto retval = index->search_with_multi_filters(query + i * query_aligned_dim, raw_filter, recall_at, L,
                                                               query_result_ids[test_id].data() + i * recall_at,
                                                               query_result_dists[test_id].data() + i * recall_at);
            }
        }
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    // uint64_t test_id = 0;
    // for (auto L : Lvec)
    // {
    //     if (L < recall_at)
    //     {
    //         diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at <<
    //         std::endl; continue;
    //     }
    //     std::string cur_result_path_prefix = result_path_prefix + "_" + std::to_string(L);

    //     std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
    //     diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

    //     cur_result_path = cur_result_path_prefix + "_dists_float.bin";
    //     diskann::save_bin<float>(cur_result_path, query_result_dists[test_id].data(), query_num, recall_at);

    //     test_id++;
    // }

    diskann::aligned_free(query);
    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, query_file, gt_file, label_type, query_filters_file;
    uint32_t num_threads, K;
    std::vector<uint32_t> Lvec;

    // Default paramters
    dist_fn = "l2"; // fixed dist_fn
    data_type = "uint8";
    K = 10;

    po::options_description desc{
        program_options_utils::make_program_description("search_memory_index", "Searches in-memory DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);
        required_configs.add_options()("query_filters_file", po::value<std::string>(&query_filters_file)->required(),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;

    if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                     "supported in general, and mips/fast_l2 only for floating "
                     "point data."
                  << std::endl;
        return -1;
    }

    try
    {
        if (data_type == std::string("uint8"))
        {
            return search_memory_index<uint8_t, uint32_t>(metric, index_path_prefix, query_file, num_threads, K, Lvec,
                                                          query_filters_file);
        }
    }
    catch (std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
// int main(){
//     return 0;
// }