#!/bin/bash

dataset_root=""
output_root="data/for_instancev2"
num_workers=1
chunk_points=2000000
semantic_map="1:0,2:1,3:2"
tree_segment_ids="1,2"
ignore_index=-1
max_scenes=0
overwrite=false

while getopts "d:o:n:c:s:t:i:m:w" opt; do
  case $opt in
    d) dataset_root=$OPTARG ;;
    o) output_root=$OPTARG ;;
    n) num_workers=$OPTARG ;;
    c) chunk_points=$OPTARG ;;
    s) semantic_map=$OPTARG ;;
    t) tree_segment_ids=$OPTARG ;;
    i) ignore_index=$OPTARG ;;
    m) max_scenes=$OPTARG ;;
    w) overwrite=true ;;
    *)
      echo "Usage: $0 -d <dataset_root> [-o <output_root>] [-n <num_workers>] [-c <chunk_points>] [-s <semantic_map>] [-t <tree_segment_ids>] [-i <ignore_index>] [-m <max_scenes>] [-w]"
      exit 1
      ;;
  esac
done

if [ -z "$dataset_root" ]; then
  echo "Usage: $0 -d <dataset_root> [-o <output_root>] [-n <num_workers>] [-c <chunk_points>] [-s <semantic_map>] [-t <tree_segment_ids>] [-i <ignore_index>] [-m <max_scenes>] [-w]"
  exit 1
fi

cmd="python pointcept/datasets/preprocessing/for_instancev2/preprocess_for_instancev2.py \
  --dataset_root ${dataset_root} \
  --output_root ${output_root} \
  --semantic_map ${semantic_map} \
  --ignore_index ${ignore_index} \
  --tree_segment_ids ${tree_segment_ids} \
  --chunk_points ${chunk_points} \
  --num_workers ${num_workers} \
  --max_scenes ${max_scenes}"

if $overwrite; then
  cmd="$cmd --overwrite"
fi

echo "$cmd"
eval "$cmd"
