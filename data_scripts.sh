#!/bin/bash
python dataset_toolkits/download.py ABO --output_dir datasets/ABO --world_size 1 --rank 0
python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO
python dataset_toolkits/render.py ABO --output_dir datasets/ABO 
python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO
python dataset_toolkits/voxelize.py House3K --output_dir datasets/House3K
python dataset_toolkits/build_metadata.py House3K --output_dir datasets/House3K
python dataset_toolkits/extract_feature.py --output_dir datasets/House3K
python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO
python dataset_toolkits/encode_ss_latent.py --output_dir datasets/House3K
python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO
python dataset_toolkits/encode_latent.py --output_dir datasets/House3K
python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO


