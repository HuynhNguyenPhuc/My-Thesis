#!/bin/bash

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

mkdir -p logs/extract_feature

for RANK in 0 1; do
  echo "Starting rank $RANK..."
  CUDA_VISIBLE_DEVICES=$RANK \
  python dataset_toolkits/extract_feature.py \
    --output_dir datasets/House3K \
    --rank $RANK \
    --world_size 2 \
    > logs/extract_feature/rank_$RANK.out 2>&1 &
done

wait

echo "✅ All feature extract jobs completed!"
