#!/bin/bash
# -*- coding: utf-8 -*-

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

mkdir -p logs/encode_latent

for RANK in 0 3; do
  echo "Starting rank $RANK..."
  CUDA_VISIBLE_DEVICES=$RANK \
  python dataset_toolkits/encode_latent.py \
    --output_dir datasets/House3K \
    --rank $RANK \
    --world_size 4 \
    > logs/extract_feature/rank_$RANK.out 2>&1 &
done

wait

echo "✅ All latent encode jobs completed!"
