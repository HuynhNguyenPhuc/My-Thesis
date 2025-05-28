#!/bin/bash

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

mkdir -p logs/render

for RANK in 0 1 2 3; do
  echo "Starting rank $RANK..."
  CUDA_VISIBLE_DEVICES=$RANK \
  python dataset_toolkits/render.py House3K \
    --output_dir datasets/House3K \
    --rank $RANK \
    --world_size 4 \
    > logs/render/rank_$RANK.out 2>&1 &
done

wait

echo "✅ All rendering jobs completed!"
