"""Run the GaussianDreamer baseline on the House3K test captions.

Thin wrapper around the unmodified upstream `launch.py`, so
`extensions/GaussianDreamer` can stay a pristine submodule.
Run from the repo root: python -m baselines.gaussiandreamer.run
"""
import argparse
import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GD_DIR = ROOT / "extensions" / "GaussianDreamer"
DEFAULT_CAPTIONS = Path(__file__).parent / "data" / "test.csv"
DEFAULT_OUT = ROOT / "outputs" / "gaussiandreamer"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=100, help="Number of prompts to run")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    df = pd.read_csv(args.captions)
    prompts = df["captions"].apply(lambda x: ast.literal_eval(x)[0]).iloc[: args.limit]

    for idx, prompt in prompts.items():
        prompt = prompt.replace(":", " with")
        print(f"Running training for row {idx} with prompt: {prompt}")
        subprocess.run(
            [
                sys.executable, "launch.py",
                "--config", "configs/gaussiandreamer-sd.yaml",
                "--train", "--gpu", args.gpu,
                f"system.prompt_processor.prompt={prompt}",
                f"exp_root_dir={args.out_dir.as_posix()}",
            ],
            cwd=GD_DIR,
            check=True,
        )


if __name__ == "__main__":
    main()
