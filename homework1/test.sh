#!/bin/bash

make
rsync --progress  -e "ssh -p 12882" * hntfkb@realmia.tnhh.net:~/cuda-tmp
echo "cd cuda-tmp; make" | ssh -p 12882 hntfkb@realmia.tnhh.net
#echo "cd cuda-tmp; make; ./bench.sh | tee bench_data.txt" | ssh -p 12882 hntfkb@realmia.tnhh.net
rsync --progress  -e "ssh -p 12882" hntfkb@realmia.tnhh.net:~/cuda-tmp/bench_data.txt .
cat bench_data.txt | grep "TIMING" > bench_data.csv

