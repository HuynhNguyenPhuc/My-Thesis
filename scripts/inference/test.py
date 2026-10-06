import argparse
import pandas as pd
import os
import torch
import ast
import shutil
from typing import *
from trellis.metrics import Metric, reconstruction_evaluate_set, generation_evaluate_set
from trellis.utils import render_utils, postprocessing_utils
from trellis.pipelines import TrellisTextTo3DPipeline


def _extract_glb(
    outputs,
    prompt,
    save_dir: str,
    save_file: str,
    mesh_simplify: float = 0.95,
    texture_size: int = 1024,
     
) -> Tuple[str, str]:
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
    parser.add_argument('--posttrain', type=bool, default=False, help="Use model from postraining")
    parser.add_argument('--lora', type=bool, default=False, help="Use model from lora")
    parser.add_argument('--ss_model', type=str, required=True, help="Path to the ss folder")
    parser.add_argument('--slat_model', type=str, required=True, help="Path to the slat folder")
    
    
    return parser.parse_args()

def copy_checkpoint_file(model_path, ckpts_dir):
    ckpts_folder = os.path.join(model_path, 'ckpts')
    safetensors_files = [f for f in os.listdir(ckpts_folder) if f.endswith('.safetensors')]
    
    if len(safetensors_files) == 1:
        checkpoint_path = os.path.join(ckpts_folder, safetensors_files[0])
        shutil.copy(checkpoint_path, ckpts_dir)
        print(f"Copied {checkpoint_path} to {ckpts_dir}")
    else:
        print(f"Error: Expected 1 .safetensors file, but found {len(safetensors_files)} in {ckpts_folder}")

def main():
    args = parse_args()

    data_dir = args.data_dir
    test_file = os.path.join(data_dir, "test.csv")
    
    assert os.path.isfile(test_file), "test file does not exist in data folder"

    posttrain = args.posttrain
    lora = args.lora
    assert not(posttrain and lora), "Post-train and lora can't be both true"

    save_dir = args.save_dir
    ss_model_path = args.ss_model
    slat_model_path = args.slat_model

    if not os.path.isdir(ss_model_path):
        raise ValueError(f"SS model directory does not exist: {ss_model_path}")
    if not os.path.isdir(slat_model_path):
        raise ValueError(f"Slat model directory does not exist: {slat_model_path}")

    ckpts_dir = "./TRELLIS-text-inference/ckpts"
    os.makedirs(ckpts_dir, exist_ok=True)

    copy_checkpoint_file(ss_model_path, ckpts_dir)
    copy_checkpoint_file(slat_model_path, ckpts_dir)


    test_df = pd.read_csv(test_file)
    test_df = test_df[:50]
    test_df["captions"] = test_df["captions"].apply(ast.literal_eval)
    test_df["captions"] = test_df["captions"].apply(lambda x: x[0])
    
    pipeline = TrellisTextTo3DPipeline.from_pretrained("./TRELLIS-text-inference", posttrain=posttrain, lora=lora)
    pipeline.cuda()

    import time
    total_time = 0
    n_samples = 0
    for _, row in test_df.iterrows():
        
        prompt = row["captions"]
        save_file = row["sha256"]
        glb_path = os.path.join(save_dir, 'shapes', save_file + ".glb")
        if os.path.exists(glb_path):
            print(f"GLB for {save_file} already exists, skipping...")
            continue
        start_t = time.time()
        outputs = pipeline.run(
            prompt=prompt, 
            formats=['mesh', "gaussian"],
            # sparse_structure_sampler_params={
            #     "steps": 25,
            #     "cfg_strength": 7.5,
            # },
            # slat_sampler_params={
            #     "steps": 25,
            #     "cfg_strength": 7.5,
            # }
        )
        end_t = time.time()
        total_time += (end_t - start_t)
        n_samples += 1
        _extract_glb(outputs=outputs, prompt=prompt, save_dir=save_dir, save_file=save_file)
        torch.cuda.empty_cache()
    print("average gen time: ", total_time / n_samples)
if __name__ == "__main__":
    main()
