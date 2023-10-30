docker exec mycontainer /bin/sh -c "cmd1;cmd2;...;cmdn"


./apps/search_contest --index_path_prefix /home/ubuntu/index_file/yfcc_R16_L80_SR80_stitched_index_label --query_file ~/big-ann-benchmarks/data/yfcc100M/query.public.100K.u8bin --recall_at 10 --search_list 50 80 90 100 110 120 130 --query_filters_file ~/big-ann-benchmarks/data/yfcc100M/query.metadata.public.100K.spmat --result_path_prefix ~/big-ann-benchmarks/results/neurips23/filter/yfcc-10M/10/rubignn/rubignn