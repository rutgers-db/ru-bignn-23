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
#include "filter_utils.h"

namespace po = boost::program_options;
using std::string;

void save_metadata(const std::string &meta_result_path, const string output_bin_result_path, const uint32_t L,
                   const uint32_t run_count, const string dataset, const std::vector<double> &search_times)
{
    string search_type_ = "knn_filtered";
    string distance_ = "euclidean";
    string build_time_ = "-1";
    string algo_ = "rubignn";
    string count_ = "10";
    string index_size_ = "-1";

    std::ofstream file;
    double best_search_time_ = *std::min_element(search_times.begin(), search_times.end());

    file.open(meta_result_path, std::ios_base::app);
    if (file)
    {
        file << output_bin_result_path << "," << build_time_ << "," << index_size_ << "," << algo_ << "," << dataset
             << "," << std::to_string(best_search_time_) << "," << algo_ << ","
             << "L" + std::to_string(L) << "," << run_count << "," << distance_ << "," << search_type_ << "," << count_
             << ",";
    }
    for (size_t i = 0; i < search_times.size() - 1; i++)
    {
        file << std::to_string(search_times[i]) << " ";
    }
    file << std::to_string(search_times.back());
    file << "\n";

    file.close();

    // descriptor["build_time"] = build_time
    // descriptor["index_size"] = index_size
    // descriptor["algo"] = definition.algorithm
    // descriptor["dataset"] = dataset
    // attrs = {
    //     "best_search_time": best_search_time,
    //     "name": str(algo),
    //     "run_count": run_count,
    //     "distance": distance,
    //     "type": search_type,
    //     "count": int(count),
    //     "search_times": search_times
    // }
}

template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &query_file,
                        const uint32_t num_threads, const uint32_t recall_at, const std::vector<uint32_t> &Lvec,
                        const std::string &query_filter_file, const std::string &result_path_prefix,
                        const string &dataset, const uint32_t runs)
{
    using TagT = uint32_t;
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    std::vector<std::vector<std::string>> query_filters;
    load_sparse_matrix(query_filter_file, query_filters);
    assert(query_filters.size() == query_num);

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

        std::vector<double> search_times;

        for (uint32_t run_count = 0; run_count < runs; run_count++)
        {
            std::cout << "Run " << run_count << "/" << runs << std::endl;

            query_result_ids[test_id].resize(recall_at * query_num);
            query_result_dists[test_id].resize(recall_at * query_num);
            std::vector<T *> res = std::vector<T *>();
            auto start = std::chrono::high_resolution_clock::now();
            omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
            for (int64_t i = 0; i < (int64_t)query_num; i++)
            {
                auto qs = std::chrono::high_resolution_clock::now();

                std::vector<std::string> raw_filter = query_filters[i];
                auto retval = index->search_with_multi_filters(query + i * query_aligned_dim, raw_filter, recall_at, L,
                                                               query_result_ids[test_id].data() + i * recall_at,
                                                               query_result_dists[test_id].data() + i * recall_at);
            }
            std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
            auto search_time = diff.count();
            std::cout << "Search with L=" << L << ", time=" << search_time << std::endl;
            search_times.emplace_back(search_time);
            if (result_path_prefix != "")
            {
                std::string cur_result_path_prefix = result_path_prefix + "_L" + std::to_string(L);
                std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
                diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);
            }
        }
        if (result_path_prefix != "")
        {
            std::string output_bin_result_path = result_path_prefix + "_L" + std::to_string(L);
            output_bin_result_path = output_bin_result_path + "_idx_uint32.bin";
            std::string cur_result_meta_path = result_path_prefix + "_search_metadata.txt";
            save_metadata(cur_result_meta_path, output_bin_result_path, L, runs, dataset, search_times);
        }
    }

    // TODO: save search metadata to file
    // store search_metadata for contest store_results() to hdf5 file

    diskann::aligned_free(query);
    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, query_file, gt_file, label_type, query_filters_file,
        result_path_prefix, dataset;
    uint32_t num_threads, K, runs;
    std::vector<uint32_t> Lvec;

    // Default paramters
    dist_fn = "l2"; // fixed dist_fn
    data_type = "uint8";
    K = 10;
    dataset = "yfcc-10M";

    po::options_description desc{
        program_options_utils::make_program_description("search_contest", "filter search for NeurIPS'23 contest")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);
        required_configs.add_options()("query_filters_file", po::value<std::string>(&query_filters_file)->required(),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        required_configs.add_options()("result_path_prefix", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("data_type", po::value<std::string>(&data_type)->default_value("uint8"),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        optional_configs.add_options()("recall_at,K", program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        optional_configs.add_options()("runs", po::value<uint32_t>(&runs)->default_value(1),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        optional_configs.add_options()("dataset", po::value<std::string>(&dataset)->default_value("yfcc-10M"),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        desc.add(required_configs).add(optional_configs);

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
                                                          query_filters_file, result_path_prefix, dataset, runs);
        }
        else if (data_type == std::string("float"))
        {
            return search_memory_index<float, uint32_t>(metric, index_path_prefix, query_file, num_threads, K, Lvec,
                                                        query_filters_file, result_path_prefix, dataset, runs);
        }
    }
    catch (std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}