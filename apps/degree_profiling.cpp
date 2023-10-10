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


int main(int argc, char **argv){
    uint32_t query_dim=192, nd=10000000, degree=96;
    std::string index_path="/data/local/big-ann-benchmarks-23/index/yfcc-10m/yfcc_R64_L120_SR96_stitched_index";
    std::string label_format_file = "/data/local/big-ann-benchmarks-23/index/yfcc-10m/yfcc_R64_L120_SR96_stitched_index_label_formatted.txt";
    std::string label_map_file = "/data/local/big-ann-benchmarks-23/index/yfcc-10m/yfcc_R64_L120_SR96_stitched_index_labels_map.txt";
    std::cout<<"index path: "<<index_path<<std::endl;
    std::string target_filter="18186";
    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);
    uint32_t start_point = 2098356;

    auto config = diskann::IndexConfigBuilder()
                    .with_metric(diskann::Metric::L2)
                    .with_dimension(query_dim)
                    .with_max_points(0)
                    .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                    .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                    .with_data_type(diskann_type_to_name<uint8_t>())
                    .with_label_type(diskann_type_to_name<uint32_t>())
                    .with_tag_type(diskann_type_to_name<uint32_t>())
                    .is_dynamic_index(false)
                    .is_enable_tags(false)
                    .is_concurrent_consolidate(false)
                    .is_pq_dist_build(false)
                    .is_use_opq(false)
                    .with_num_pq_chunks(0)
                    .with_num_frozen_pts(num_frozen_pts)
                    .build();

    std::unique_ptr<diskann::InMemGraphStore> _graph_store(new diskann::InMemGraphStore(nd,degree));
    std::unique_ptr<diskann::Distance<uint8_t>> distance_func(new diskann::DistanceL2UInt8());
    std::unique_ptr<diskann::InMemDataStore<uint8_t>> _data_store(new diskann::InMemDataStore(nd,query_dim,std::move(distance_func)));
    diskann::Index<uint8_t>* index = new diskann::Index<uint8_t>(config, std::move(_data_store),std::move(_graph_store));
    index->load(index_path.c_str(),1,100);
    

    std::vector<label_set> point_ids_to_labels;
    tsl::robin_map<std::string, uint32_t> labels_to_number_of_points;
    tsl::robin_map<std::string, std::string> label_map;
    label_set all_labels;
    std::string universal_label = "0";

    std::tie(point_ids_to_labels, labels_to_number_of_points, all_labels) =
        diskann::parse_label_file(label_format_file, universal_label);

    std::ifstream map_reader(label_map_file);
    std::string line,token1,token2;
    while (std::getline(map_reader,line)){
        std::istringstream new_iss(line);
        std::getline(new_iss,token1,'\t');
        std::getline(new_iss,token2,'\t');
        label_map[token1] = token2;
    }

    std::string target_label = label_map[target_filter];
    std::cout << "Index loaded" << std::endl;
    
    // find the maximum degree
    std::priority_queue<std::pair<int,uint32_t>> degree_pq;
    uint32_t num_of_start_point = 10;
    for (uint32_t i=0;i<nd;i++){
        if (point_ids_to_labels[i].find(target_label)!=point_ids_to_labels[i].end()){
            int degree = 0;
            for (auto neighbor: index->get_neighbor(i)){
                if (point_ids_to_labels[neighbor].find(target_label)!=point_ids_to_labels[neighbor].end()){
                    degree++;
                }
            }
            degree_pq.push(std::pair<int,uint32_t>(-degree,i));
            if (degree_pq.size()>num_of_start_point){
                degree_pq.pop();
            }
        }
    }

    std::cout<<"Test "<<degree_pq.size()<<" number of start points"<<std::endl;
    std::vector<uint32_t> global_visit_set;
    while (!degree_pq.empty()){
        auto degree_point = degree_pq.top();
        degree_pq.pop();
        uint32_t i = degree_point.second;
        start_point = i;
        std::queue<std::pair<uint32_t, uint32_t>> queue;
        std::vector<uint32_t> visit_set;
        queue.emplace(start_point,1);
        visit_set.emplace_back(start_point);
        while (!queue.empty()){
            auto p = queue.front();
            queue.pop();
            uint32_t id = p.first;
            std::vector<uint32_t> tag_neighbors;
            for (auto neighbor: index->get_neighbor(id)){
                if (point_ids_to_labels[neighbor].find(target_label)!=point_ids_to_labels[neighbor].end() && 
                    std::find(visit_set.begin(),visit_set.end(),neighbor)==visit_set.end()){
                        visit_set.emplace_back(neighbor);
                        queue.emplace(neighbor,p.second+1);
                        tag_neighbors.emplace_back(neighbor);
                }
            }
        }
        std::cout<<"Starting at "<<start_point<<"with degree "<<-degree_point.first<<" can reach "<<visit_set.size()<<" points."<<std::endl;
        for (uint32_t visit:visit_set){
            if (std::find(global_visit_set.begin(),global_visit_set.end(),visit)==global_visit_set.end()){
                global_visit_set.push_back(visit);
            }
        }
    }
    std::cout<<"Can visit "<<global_visit_set.size()<<" points in total"<<std::endl;
    // uint32_t maximum=0;
    // for (uint32_t i=0;i<nd;i++){
    //     if (point_ids_to_labels[i].find(target_label)!=point_ids_to_labels[i].end()){
    //         start_point = i;
    //         std::queue<std::pair<uint32_t, uint32_t>> queue;
    //         std::vector<uint32_t> visit_set;
    //         queue.emplace(start_point,1);
    //         visit_set.emplace_back(start_point);
    //         while (!queue.empty()){
    //             auto p = queue.front();
    //             queue.pop();
    //             uint32_t id = p.first;
    //             std::vector<uint32_t> tag_neighbors;
    //             for (auto neighbor: index->get_neighbor(id)){
    //                 if (point_ids_to_labels[neighbor].find(target_label)!=point_ids_to_labels[neighbor].end() && 
    //                     std::find(visit_set.begin(),visit_set.end(),neighbor)==visit_set.end()){
    //                         visit_set.emplace_back(neighbor);
    //                         queue.emplace(neighbor,p.second+1);
    //                         tag_neighbors.emplace_back(neighbor);
    //                 }
    //             }
    //             // std::cout<<"Layer "<<p.second<<", id="<<id<<", neighbors: {";
    //             // if (tag_neighbors.size()>0){
    //             //     for (uint32_t i=0;i<tag_neighbors.size()-1;i++){
    //             //         std::cout<<tag_neighbors[i]<<", ";
    //             //     }
    //             //     std::cout<<tag_neighbors[tag_neighbors.size()-1]<<"}"<<std::endl;
    //             // }
    //         }
    //         std::cout<<"Starting at "<<start_point<<" can reach "<<visit_set.size()<<" points."<<std::endl;
    //         if (visit_set.size()>maximum){
    //             maximum = visit_set.size();
    //         }
    //     }
    // }
    // std::cout<<"maximum can reach "<<maximum<<" points"<<std::endl;
    std::cout<<"have "<<labels_to_number_of_points[target_label]<<" points in total"<<std::endl;
    
}