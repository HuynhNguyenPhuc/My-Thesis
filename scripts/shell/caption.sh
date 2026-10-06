#!/bin/bash
# Generate text captions for a dataset with Gemini (needs GEMINI_API_KEY in .env, see .env.example).
# Step 1 renders 4 views per shape (front/back/left/right); step 2 asks Gemini for one detailed
# description, one concise caption, then 10 progressively shorter captions per shape.
# Per-rank results land in datasets/House3K/captioned_<rank>.csv; merge_captions.py writes them into metadata.csv.

DATASET=${1:-House3K}
OUT=datasets/$DATASET
WORLD_SIZE=${2:-1}

python dataset_toolkits/render_caption.py $DATASET --output_dir $OUT

mkdir -p logs/caption
for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
  python dataset_toolkits/extract_caption.py $DATASET \
    --output_dir $OUT \
    --rank $RANK \
    --world_size $WORLD_SIZE \
    > logs/caption/rank_$RANK.out 2>&1 &
done
wait

python dataset_toolkits/merge_captions.py --output_dir $OUT
echo "✅ Captioning done"
