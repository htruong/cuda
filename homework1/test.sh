#!/bin/bash

rsync --recursive --progress --size-only -e "ssh -p 12882" * hntfkb@realmia.tnhh.net:~/cuda-tmp
echo "cd cuda-tmp; make; ./hw1 graphs/in/graph_10000_10000.in" | ssh -p 12882 hntfkb@realmia.tnhh.net
