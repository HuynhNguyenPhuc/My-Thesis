import argparse
import pandas as pd
import os
import torch
import ast
import shutil
import json 
from typing import *
from trellis.metrics import Metric, reconstruction_evaluate_set, generation_evaluate_set
from trellis.metrics import compute_fid_kid, compute_dataset_clip_scores
from trellis.utils import render_utils, postprocessing_utils, render_all_shapes
from trellis.pipelines import TrellisTextTo3DPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Text to 3D Evaluation Script")
    parser.add_argument('--data_dir', type=str, default="datasets/House3K", help="Directory for the dataset")
    parser.add_argument('--test_dir', type=str, default="inference/shapes", help="Directory to save the output glb files")
    parser.add_argument('--kfid', action='store_true', help='Evaluate using FID, KID')
    parser.add_argument('--clip', action='store_true', help='Evaluate using CLIP Score')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    test_dir = args.test_dir
    shape_dir = os.path.join(test_dir, 'shapes')
    metric_log = os.path.join(test_dir, 'evaluate.json')
    assert os.path.isdir(data_dir), "data directory doesn't exist"
    assert os.path.isdir(test_dir), "test directory doesn't exist"
    assert os.path.isdir(shape_dir), "No shape directory inside test directory"
    eval_results = {}
    rendering = False
    if not os.path.isdir(os.path.join(test_dir, 'renders')):
        rendering = True
    if rendering:
        print("Rendering test shapes...")
        render_all_shapes(shape_dir=shape_dir, output_dir=os.path.join(test_dir, 'renders'))
    
    if args.kfid:
        gt_renders_path = os.path.join(data_dir, "test_renders")
        pr_renders_path = os.path.join(test_dir, "renders")
        
        inceptionV3_scores = compute_fid_kid(paths=[gt_renders_path, pr_renders_path], model_name="inceptionv3")
        eval_results["inception_score"] = {
            "inceptionv3": {
                "fid": inceptionV3_scores["fid"],
                "kid": inceptionV3_scores["kid"]
            }
        }
        
        dinoV2_scores = compute_fid_kid(paths=[gt_renders_path, pr_renders_path], model_name="dinov2")
        eval_results["inception_score"]["dinov2"] = {
            "fid": dinoV2_scores["fid"],
            "kid": dinoV2_scores["kid"]
        }
    
    if args.clip:
        clip_score = compute_dataset_clip_scores(data_dir=data_dir, test_dir=test_dir)
        eval_results["Clip Score"] = clip_score

    # Save the evaluation results to a JSON file
    with open(metric_log, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)
    print(f"Evaluation results saved to {metric_log}")

if __name__ == "__main__":
    main()
