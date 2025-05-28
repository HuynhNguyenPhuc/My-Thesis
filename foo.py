import os
import shutil

# Your input and output directories
src_root = "./test"
dst_root = "./demo_distillation"

# Replace with your actual sha256 folder names (same across all subfolders)
sha256_folders = [
    '36064f2e9eebd52e42908f91172c85b128154916df625934c99afc7c381c0591', 'cfb7a1c47fa459d4e8c440daa807cf9924868f6e438e11ec650d11b91cb18d71', '9a8c49e69bca49bc2a254cdd926f60970f03d88bead81d5e10462dbe35234614', '5b66352e08345fa99ddeb18fcd5ef03f514ac939a5fc6b7c08e2b5010f2b060a', '03d7de5ee93a93a5180aec9e71aa8cf56d93a982cd074745833d3d0114f3f04a', 'b07a8147480ff9b93d51391f1849b6d564664abfc7d37885fe9eac292eb0dc1d'
]


for subfolder in os.listdir(src_root):
    sub_path = os.path.join(src_root, subfolder)
    if not os.path.isdir(sub_path):
        continue

    # Destination path
    dst_sub_path = os.path.join(dst_root, subfolder)
    os.makedirs(dst_sub_path, exist_ok=True)

    for i, sha in enumerate(sha256_folders):
        img_dir = os.path.join(sub_path, "renders", sha)
        if not os.path.isdir(img_dir):
            print(f"Missing: {img_dir}")
            continue

        imgs = sorted([
            f for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))  # <-- Only include files
               and f.lower().endswith(('.png', '.jpg', '.jpeg')) 
            ])  # Sort to ensure consistent naming
        for j, img in enumerate(imgs[:4]):  # Only take first 4 images
            src_img_path = os.path.join(img_dir, img)
            dst_img_name = f"prompt{i}_{j}.png"
            dst_img_path = os.path.join(dst_sub_path, dst_img_name)
            shutil.copyfile(src_img_path, dst_img_path)
