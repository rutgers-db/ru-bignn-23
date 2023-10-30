CONTEST_REPO_PATH=/home/ubuntu/big-ann-benchmarks #path to big-ann-benchmarks
INDEX_FILE_PATH=~/index_file
docker container run -it  --mount type=bind,src=$CONTEST_REPO_PATH/results,dst=/home/app/results type=bind,src=$INDEX_FILE_PATH,dst=/home/app/index_file --read-only --mount type=bind,src=$CONTEST_REPO_PATH/data,dst=/home/app/data  neurips23-filter-rubignn  /bin/bash