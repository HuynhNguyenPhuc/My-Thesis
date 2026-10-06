#!/bin/bash
# Full Houses3K preprocessing. Run from the repo root on a GPU machine, after
# scripts/shell/download_house3k.sh. Each stage writes per-rank CSVs that
# build_metadata.py merges into datasets/House3K/metadata.csv, so it is re-run after every stage.
#
# Stage 6 (captions) needs GEMINI_API_KEY in .env, see scripts/shell/caption.sh.

set -e
DATASET=House3K
OUT=datasets/$DATASET

python dataset_toolkits/build_metadata.py $DATASET --output_dir $OUT          # index + hash FBX files

bash scripts/shell/render.sh                                                  # 1. 150 views per shape (4 GPUs)
python dataset_toolkits/build_metadata.py $DATASET --output_dir $OUT

python dataset_toolkits/voxelize.py $DATASET --output_dir $OUT                # 2. 64^3 voxel grids
python dataset_toolkits/build_metadata.py $DATASET --output_dir $OUT

python dataset_toolkits/extract_feature.py --output_dir $OUT                  # 3. DINOv2 features
python dataset_toolkits/build_metadata.py $DATASET --output_dir $OUT

python dataset_toolkits/encode_ss_latent.py --output_dir $OUT                 # 4. sparse-structure latents
python dataset_toolkits/build_metadata.py $DATASET --output_dir $OUT

python dataset_toolkits/encode_latent.py --output_dir $OUT                    # 5. structured latents
python dataset_toolkits/build_metadata.py $DATASET --output_dir $OUT

bash scripts/shell/caption.sh $DATASET                                        # 6. Gemini captions -> metadata.csv
