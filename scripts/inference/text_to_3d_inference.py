import os
# os.environ['SPCONV_ALGO'] = 'native' 
# os.environ['ATTN_BACKEND'] = 'xformers'
import shutil
from typing import *
import torch
import numpy as np
import imageio
import hashlib
import json
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


def hash_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
    
def extract_glb(
    outputs,
    prompt,
    save_dir: str = None,
    mesh_simplify: float = 0.95,
    texture_size: int = 1024,
    save_file: str = None
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
    if save_dir:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = "./TRELLIS-text/inference"
        os.makedirs(save_dir, exist_ok=True)
    shape_dir = os.path.join(save_dir, "shapes")
    os.makedirs(shape_dir, exist_ok=True)
    dict_path = os.path.join(save_dir, "dict.json")
    if save_file is None:
        save_file = hash_text_sha256(prompt)
    glb_path = os.path.join(shape_dir, save_file + ".glb")

    
    
    
    print("postprocessing")
    glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    print("Exporting glb...")
    glb.export(glb_path)
    print("Export done")
    torch.cuda.empty_cache()
    new_shape = {
        "text": prompt,
        "shape_path": save_file
    }

    if os.path.exists(dict_path):
        with open(dict_path, "r") as file:
            try:
                data = json.load(file)  # Load JSON list
                if not isinstance(data, list):
                    data = [] 
            except json.JSONDecodeError:
                data = [] 
    else:
        data = []  


    data.append(new_shape)

    with open(dict_path, "w") as file:
        json.dump(data, file, indent=4)
    return glb_path, glb_path

if __name__ == "__main__":
    pipeline = TrellisTextTo3DPipeline.from_pretrained("./TRELLIS-text-inference")
    pipeline.cuda()
    
    prompt = "A Japanese-style house" #test dataset
    
    outputs = pipeline.run(prompt=prompt, formats=['mesh', "gaussian"])
    extract_glb(outputs, prompt=prompt)
    torch.cuda.empty_cache()
    
