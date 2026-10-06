import argparse
import pandas as pd
import os
import torch
import ast
import shutil
from typing import *
from trellis.metrics import Metric, reconstruction_evaluate_set, generation_evaluate_set
from trellis.utils import render_utils, postprocessing_utils
from trellis.pipelines import TrellisImageTo3DPipeline
from diffusers import DiffusionPipeline

def _extract_glb(
    outputs,
    prompt,
    save_dir: str,
    save_file: str,
    mesh_simplify: float = 0.95,
    texture_size: int = 1024,
     
) -> str:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    os.makedirs(os.path.join(save_dir, 'shapes'), exist_ok=True)
    glb_path = os.path.join(save_dir, 'shapes', save_file + ".glb")
    print("postprocessing")
    glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    print("Exporting glb...")
    glb.export(glb_path)
    print("Export done")
    torch.cuda.empty_cache()
    return glb_path

def parse_args():
    parser = argparse.ArgumentParser(description="Text to 3D Inference Script")
    parser.add_argument('--data_dir', type=str, default="datasets/House3K", help="Directory for the dataset")
    parser.add_argument('--save_dir', type=str, default="inference/shapes", help="Directory to save the output glb files")
    return parser.parse_args()

def main():
    args = parse_args()

    data_dir = args.data_dir
    test_file = os.path.join(data_dir, "test.csv")
    assert os.path.isfile(test_file), "test file does not exist in data folder"
    save_dir = args.save_dir

  


    test_df = pd.read_csv(test_file)
    test_df["captions"] = test_df["captions"].apply(ast.literal_eval)
    test_df["captions"] = test_df["captions"].apply(lambda x: x[0])

    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("./TRELLIS-image-large")
    trellis_pipeline.cuda()
    diffuser_pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
    diffuser_pipeline.to("cuda")

    for _, row in test_df.iterrows():
        prompt = row["captions"]
        save_file = row["sha256"]
        glb_path = os.path.join(save_dir, 'shapes', save_file + ".glb")
        if os.path.exists(glb_path):
            print(f"GLB for {save_file} already exists, skipping...")
            continue
        image = diffuser_pipeline(prompt).images[0]
        outputs = trellis_pipeline.run(
            image,
            seed=1,
        )
        _extract_glb(outputs=outputs, prompt=prompt, save_dir=save_dir, save_file=save_file)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
