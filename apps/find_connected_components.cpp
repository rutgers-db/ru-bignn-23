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

uint32_t query_dim=192, nd=10000000, degree=96;
std::vector<int> visit_set(nd,0);
std::vector<int> dfn(nd,0);
std::vector<int> low(nd,0);
std::vector<bool> onStack(nd,false);
std::vector<uint32_t> stack, trace;
std::vector<label_set> point_ids_to_labels;
std::string target_label;
std::vector<int> ids(nd,0);
uint32_t id=0, sccCount=0;

void dfs(uint32_t p, diskann::Index<float>* index){
    stack.push_back(p);
    visit_set[p] = 1;
    assert(p<1000000);
    ids[p] = id;
    low[p] = id;
    id++;
    std::vector<uint32_t> neighbor_list = index->get_neighbor(p);
    std::cout<<"Point "<<p<<" has "<<neighbor_list.size()<<" neighbors"<<std::endl;
    for (auto q:neighbor_list){
        assert(q<1000000);
        if (visit_set[q]==0){
            if (point_ids_to_labels[q].find(target_label)!=point_ids_to_labels[q].end()){
                dfs(q, index);
                low[p] = std::min(low[p],low[q]);
                if (low[p]>ids[q]){
                    sccCount++;
                    uint32_t pop_value = stack.back();
                    stack.pop_back();
                    while (pop_value!=p){
                        pop_value = stack.back();
                        stack.pop_back();
                    }
                }
            }
            else{
                visit_set[q] = 1;
            }
        }
        else if (point_ids_to_labels[q].find(target_label)!=point_ids_to_labels[q].end()){
                    low[p] = std::min(low[p],ids[q]);
                }
    }
    return;
}

int main(int argc, char **argv){
    std::string index_path="/data/local/big-ann-benchmarks-23/index/yfcc-10m/yfcc_R64_L120_SR96_stitched_index";
    std::string label_format_file = "/data/local/big-ann-benchmarks-23/index/yfcc-10m/yfcc_R64_L120_SR96_stitched_index_label_formatted.txt";
    std::string label_map_file = "/data/local/big-ann-benchmarks-23/index/yfcc-10m/yfcc_R64_L120_SR96_stitched_index_labels_map.txt";
    std::cout<<"index path: "<<index_path<<std::endl;
    std::string target_filter="18186";
    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);

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
    

    // std::vector<label_set> point_ids_to_labels;
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

    target_label = label_map[target_filter];
    // std::vector<int> visit_set(nd,0);
    // std::vector<int> ids(nd,0);
    // std::vector<int> low(nd,0);
    // std::vector<uint32_t> stack;
    // uint32_t sccCount=0;
    // uint32_t id=0;

    std::cout<<target_label<<std::endl;
    uint32_t total_size = 0;
    for (int i=0;i<nd;i++){
        if (point_ids_to_labels[i].find(target_label)!=point_ids_to_labels[i].end()){
            total_size++;
        }
    }
    std::cout<<"Total num: "<<total_size<<std::endl;
    

    for (uint32_t i=0;i<nd;i++){
        if (dfn[i]==0 && point_ids_to_labels[i].find(target_label)!=point_ids_to_labels[i].end()){
            stack.push_back(i);
            trace.push_back(i);
            uint32_t* parent = new uint32_t[nd];
            memset(parent,0,sizeof(uint32_t)*nd);
            parent[i] = i;
            ++id;
            dfn[i] = id;
            low[i] = id;
            onStack[i] =true;
            while (!stack.empty()){
                uint32_t v = stack.back();
                bool pop_flag = true;
                const std::vector<uint32_t>& neighbor = index->get_neighbor(v);
                for (auto w:neighbor){
                    if (point_ids_to_labels[w].find(target_label)!=point_ids_to_labels[w].end()){
                        if (dfn[w]==0){
                            stack.push_back(w);
                            trace.push_back(w);
                            parent[w] = v;
                            ++id;
                            dfn[w] = id;
                            low[w] = id;
                            onStack[w] =true;
                            pop_flag = false;
                            break;
                        }
                        else if (onStack[w]){
                            low[v] = std::min(low[v],dfn[w]);
                        }
                    }
                }
                if (pop_flag){
                    stack.pop_back();
                    low[parent[v]] = std::min(low[parent[v]],low[v]);
                    if (low[v] == dfn[v]){
                        uint32_t sccSize = 1;
                        uint32_t w = trace.back();
                        trace.pop_back();
                        onStack[w] = false;
                        std::vector<uint32_t> scc;
                        scc.push_back(w);
                        while (w!=v){
                            w = trace.back();
                            trace.pop_back();
                            onStack[w] = false;
                            if (std::find(scc.begin(),scc.end(),w)==scc.end()){
                                sccSize++;
                                scc.push_back(w);
                            }
                        }
                        sccCount++;
                        std::cout<<"New scc found of size "<<sccSize<<std::endl;
                    } 
                }
            }
            delete[] parent;
        }
    }

    std::cout<<"Number of SCCs: "<<sccCount<<std::endl;
    

    std::cout << "Index loaded" << std::endl;
    
}