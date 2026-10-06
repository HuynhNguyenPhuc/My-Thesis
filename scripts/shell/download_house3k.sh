#!/bin/bash
# Download Houses3K (FBX models, Google Drive) into datasets/House3K/raw/3dmodels/original/.
# Source: https://github.com/darylperalta/Houses3K
# gdown cannot list Drive folders with more than 50 files per folder and can hit Google's quota;
# if it fails, download the folder manually in a browser and unzip it to the same location
# (see the README, Dataset section).

OUT=${1:-datasets/House3K}
FOLDER_URL=https://drive.google.com/drive/folders/1fb5gGBxFIibvHrsJGquO6N8rqKSbkIZB

pip install -q gdown
mkdir -p "$OUT/raw/3dmodels/original"
gdown --folder "$FOLDER_URL" -O "$OUT/raw/3dmodels/original" --remaining-ok
find "$OUT/raw/3dmodels/original" -iname "*.fbx" | wc -l | xargs echo "FBX files downloaded (expected 3000):"
