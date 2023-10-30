mkdir -p $CONTEST_REPO_PATH/results/neurips23/filter/yfcc-10M/10/rubignn
cd /home/app/ru-bignn-23/build
./apps/search_contest --index_path_prefix /home/app/index_file/yfcc_R16_L80_SR80_stitched_index_label --query_file /home/app/data/yfcc100M/query.public.100K.u8bin --search_list 50 80 90 100 110 120 130 --query_filters_file /home/app/data/yfcc100M/query.metadata.public.100K.spmat --result_path_prefix /home/app/results/neurips23/filter/yfcc-10M/10/rubignn/rubignn


./apps/search_contest --index_path_prefix /home/ubuntu/built_index/index_file_96/yfcc_R16_L80_SR96_stitched_index_label --query_file /home/ubuntu/big-ann-benchmarks/data/yfcc100M/query.public.100K.u8bin --search_list 50 80 90 100 110 120 130 --query_filters_file /home/ubuntu/big-ann-benchmarks/data/yfcc100M/query.metadata.public.100K.spmat --result_path_prefix /home/ubuntu/big-ann-benchmarks/results/neurips23/filter/yfcc-10M/10/rubignn/rubignn --runs 1

 python3 ../contest-scripts/output_bin_to_hdf5.py ~/big-ann-benchmarks/results/neurips23/filter/yfcc-10M/10/rubignn/rubignn_search_metadata.txt /home/ubuntu/big-ann-benchmarks