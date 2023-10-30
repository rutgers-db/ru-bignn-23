CONTEST_REPO_PATH=/home/ubuntu/big-ann-benchmarks #path to big-ann-benchmarks directory
INDEX_FILE_PATH=/home/ubuntu/built_index #path to index_file directory

docker container run -it  --mount type=bind,src=$CONTEST_REPO_PATH/results,dst=/home/app/results --mount type=bind,src=$INDEX_FILE_PATH/index_file_docker_build,dst=/home/app/index_file --read-only --mount type=bind,src=$CONTEST_REPO_PATH/data,dst=/home/app/data  neurips23-filter-rubignn  /bin/bash -c 'mkdir -p /home/app/index_file/index_file_docker_build && 
cd /home/app/ru-bignn-23/build &&
./apps/base_label_to_label_file /home/app/data/random-filter100000/data_metadata_100000_50 /home/app/index_file/label_file_base_random_filter.txt &&
./apps/build_stitched_index --data_type uint8 --data_path /home/app/data/random-filter100000/data_100000_50  --index_path_prefix home/app/index_file/index_file_docker_build/random-filter100000_stitched_index_label -R 16 -L 80 --stitched_R 50 --alpha 1.2 --label_file /home/app/index_file/label_file_base_random_filter.txt --universal_label 0'