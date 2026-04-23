#!/bin/bash

dataset_root=""
output_root="data/whu_stree"
num_workers=1
chunk_points=1000000
ignore_index=-1
max_scenes=0
test_label_source="reference"
intensity_scale=65535
keep_sparse_label=false
overwrite=false

while getopts "d:o:n:c:i:m:t:s:kw" opt; do
  case $opt in
    d) dataset_root=$OPTARG ;;
    o) output_root=$OPTARG ;;
    n) num_workers=$OPTARG ;;
    c) chunk_points=$OPTARG ;;
    i) ignore_index=$OPTARG ;;
    m) max_scenes=$OPTARG ;;
    t) test_label_source=$OPTARG ;;
    s) intensity_scale=$OPTARG ;;
    k) keep_sparse_label=true ;;
    w) overwrite=true ;;
    *)
      echo "Usage: $0 -d <dataset_root> [-o <output_root>] [-n <num_workers>] [-c <chunk_points>] [-i <ignore_index>] [-m <max_scenes>] [-t <test_label_source>] [-s <intensity_scale>] [-k] [-w]"
      exit 1
      ;;
  esac
done

if [ -z "$dataset_root" ]; then
  echo "Usage: $0 -d <dataset_root> [-o <output_root>] [-n <num_workers>] [-c <chunk_points>] [-i <ignore_index>] [-m <max_scenes>] [-t <test_label_source>] [-s <intensity_scale>] [-k] [-w]"
  exit 1
fi

cmd="python pointcept/datasets/preprocessing/whu_stree/preprocess_whu_stree.py \
  --dataset_root ${dataset_root} \
  --output_root ${output_root} \
  --ignore_index ${ignore_index} \
  --chunk_points ${chunk_points} \
  --num_workers ${num_workers} \
  --max_scenes ${max_scenes} \
  --test_label_source ${test_label_source} \
  --intensity_scale ${intensity_scale}"

if $keep_sparse_label; then
  cmd="$cmd --keep_sparse_label"
fi

if $overwrite; then
  cmd="$cmd --overwrite"
fi

echo "$cmd"
eval "$cmd"
