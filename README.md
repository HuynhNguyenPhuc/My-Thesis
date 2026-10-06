# Generate 3D Models from Multi-modal Descriptions

Graduation thesis (*Sinh mô hình 3D từ mô tả đa phương thức*), Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT, VNU-HCM), May 2025.

- **Authors:** Huỳnh Nguyên Phúc (2110451), Mai Phương Nhã (2111892)
- **Advisor:** Dr. Lê Thành Sách
- **Reviewer:** Dr. Nguyễn Đức Dũng

The thesis (in Vietnamese) and its defense slides are in [`docs/`](docs/).

## Summary

The project studies text-to-3D generation of houses on the **Houses3K** dataset, building on the text-conditioned variant of [TRELLIS](https://github.com/microsoft/TRELLIS). The contributions are:

1. **Fine-tuning** the pretrained text-to-3D model on Houses3K with **L2-SP** regularisation and **LoRA**.
2. **Knowledge distillation** to shorten inference time while keeping the original model's knowledge.
3. **Rectified Flow vs. Diffusion**: the generative models are compared by also training diffusion-based variants.
4. A **prompting technique** that automatically captions a 3D dataset (a by-product of the research).

[GaussianDreamer](extensions/GaussianDreamer/README.md) (optimization-based) and a Stable Diffusion → TRELLIS image-to-3D pipeline serve as comparison baselines.

## How it works

Generation is a two-stage pipeline (`trellis/pipelines/trellis_text_to_3d.py`):

1. **Sparse Structure (SS)** — a text-conditioned flow model produces the occupied voxel grid.
2. **Structured Latent (SLat)** — a text-conditioned sparse flow model produces features on those voxels, decoded to Gaussians, radiance field or mesh and exported as a textured GLB.

Houses3K has 3,000 textured house models (12 batches × 5 texture sets × 50 houses, FBX). It ships without captions, so each shape is rendered from four views and captioned with `gemini-2.0-flash`: one detailed caption, then ten progressively shorter versions (11 per shape, used as augmentation). Shapes are split 70% train / 20% validation / 10% test. Each processed shape has 150 renders, a voxel grid, DINOv2 features, and SS and SLat latents.

Each adaptation method trains the SS and SLat models separately; both are loaded together at inference.

## Repository layout

| Path | Contents |
| --- | --- |
| `trellis/` | Core library: models, trainers, samplers, renderers, metrics |
| `scripts/` | Entry points: `train/`, `finetune/`, `posttrain/`, `distill/`, `inference/`, `shell/` |
| `dataset_toolkits/` | Dataset download, rendering, voxelization, feature and latent extraction |
| `experiments/` | Training outputs: `{Diffusion,LoRA,Post-training,Distillation}/{ss,slat}/<run>` |
| `results/` | Generated shapes and `evaluate.json` results per run, including the GaussianDreamer baseline |
| `demo/` | Qualitative renders (4 views per prompt): `finetune_examples/` (3 prompts, base vs. fine-tuned vs. baselines) and `model_comparison/` (6 prompts, all models) |
| `models/pipeline.json` | Pipeline config (checkpoints, sampler steps, CFG) |
| `extensions/` | `vox2seq` CUDA extension and the GaussianDreamer submodule |
| `baselines/` | Scripts for the comparison baselines: GaussianDreamer wrappers (`gaussiandreamer/`) and text → image → 3D (`StableDiffusion.py`) |
| `docs/` | Thesis document and slides |

## Requirements

Linux with an NVIDIA GPU. The pinned stack is PyTorch 2.5.1 with CUDA 12.4. Windows is not supported for running the pipeline.

```bash
bash setup.sh                      # Python deps, flash-attn, kaolin, CUDA extensions
git submodule update --init        # FlexiCubes (pinned to f97beb0) and the GaussianDreamer baseline
```

The GaussianDreamer baseline (`extensions/GaussianDreamer`) is an unmodified clone of [hustvl/GaussianDreamer](https://github.com/hustvl/GaussianDreamer). To run it, follow its README (it needs `shap-e` cloned into its folder). The dependency pins used for this project are in `baselines/gaussiandreamer/requirements.txt`.

The following are not tracked and must be provided locally:

- `datasets/` (for example `datasets/House3K`)
- `TRELLIS-text-train/`, the pretrained pipeline folder every trainer loads
- `trellis/datasets/`, the dataset classes the training scripts import (`TextConditionedSparseStructure` and others). They were missing from the first commit because of a `.gitignore` rule that is now fixed, so copy them in from the original machine

## Dataset: Houses3K

[Houses3K](https://github.com/darylperalta/Houses3K) (Peralta et al., ECCV Workshops 2020) has 3,000 textured house models in FBX format: 12 batches × 50 house geometries × 5 texture sets (A–E). It ships without captions, so this project renders and captions every shape (thesis section 4.1.3). Check the dataset's license before redistributing it.

### 1. Download

```bash
bash scripts/shell/download_house3k.sh            # -> datasets/House3K/raw/3dmodels/original/
```

The script uses `gdown` on the dataset's [Google Drive folder](https://drive.google.com/drive/folders/1fb5gGBxFIibvHrsJGquO6N8rqKSbkIZB) and prints how many FBX files it found (expected 3000). `gdown` can fail on large folders because of Google's download quota or its 50-files-per-folder listing limit. In that case, download the folder in a browser and unzip it so the batch folders sit directly under `datasets/House3K/raw/3dmodels/original/`, for example `BATCH_1/Set_A/BAT1_SETA_HOUSE1.fbx`. Folder and file names can vary, because the loader (`dataset_toolkits/datasets/House3K.py`) finds every `.fbx` recursively.

### 2. Process

Needs Linux, an NVIDIA GPU, the dependencies from `setup.sh`, and Blender (`render.py` downloads it).

```bash
bash scripts/shell/prepare_house3k.sh
```

The script runs these stages and calls `dataset_toolkits/build_metadata.py` after each one to merge the per-rank CSVs into `datasets/House3K/metadata.csv`:

| Stage | Script | Output in `datasets/House3K/` |
| --- | --- | --- |
| Index | `build_metadata.py House3K` | `metadata.csv` (SHA-256 and path of every FBX) |
| Render | `scripts/shell/render.sh` → `render.py` | `renders/` (150 views per shape) |
| Voxelize | `voxelize.py` | `voxels/` |
| Features | `extract_feature.py` | `features/` (DINOv2) |
| SS latents | `encode_ss_latent.py` | `ss_latents/` |
| SLat latents | `encode_latent.py` | `latents/` |
| Captions | `scripts/shell/caption.sh` | `captions` column in `metadata.csv` |

`render.sh` is set up for 4 GPUs (ranks 0–3); change its rank list and `--world_size` for another setup. `extract_feature.sh`, `encode_ss_latent.sh` and `encode_latent.sh` are sharded versions of the matching stages.

**Captions.** Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. `caption.sh House3K [num_shards]` renders four views per shape (`render_caption.py`), then asks Gemini (`gemini-2.0-flash` by default) for one detailed description, one caption of at most 40 words, and 10 progressively shorter versions, from about 12 words down to 5 (`extract_caption.py`). It merges them into the `captions` column of `metadata.csv` as a Python list (`merge_captions.py`).

**Split.** The thesis uses a random 70% / 20% / 10% train / validation / test split. The training scripts read it through the dataset classes in `trellis/datasets/` (see Requirements).

After processing, `datasets/House3K/` holds `raw/`, `renders/`, `voxels/`, `features/`, `ss_latents/`, `latents/` and `metadata.csv`. The training scripts read this folder (`data_dir = "datasets/House3K"`).

## Usage

Run everything from the repository root as a module. Data preparation is described in the Houses3K section above.

**Train.** Hyperparameters are set inside each script, so edit the script before running.

```bash
python -m scripts.train.train_ss
torchrun --nproc_per_node=4 -m scripts.finetune.finetune_ss_ddp   # multi-GPU LoRA
```

**Generate and evaluate:**

```bash
python -m scripts.inference.test \
  --data_dir ./datasets/House3K --save_dir ./results/<run> \
  --ss_model ./experiments/<method>/ss/<run> \
  --slat_model ./experiments/<method>/slat/<run> [--lora True]

python -m scripts.inference.evaluate \
  --data_dir ./datasets/House3K --test_dir ./results/<run> --kfid --clip
```

Evaluation reports FID/KID (InceptionV3) and CLIP score, written to `<test_dir>/evaluate.json`.

**GaussianDreamer baseline** (wrappers in `baselines/gaussiandreamer/`; the submodule itself stays untouched):

```bash
python -m baselines.gaussiandreamer.run            # one run per test caption -> outputs/gaussiandreamer/
python -m baselines.gaussiandreamer.collect_gif    # gather shape.gif files
python -m baselines.gaussiandreamer.extract_frames # 4 frames per gif
```

**Single prompt / web demo:**

```bash
python -m scripts.inference.demo --prompt "..." --save_dir ./temp \
  --ss_model <ss_dir> --slat_model <slat_dir>
python -m scripts.inference.app    # Gradio
```

## Results

Evaluated on the 10% test split with CLIP score (↑) and FID/KID (↓, InceptionV3 and DINOv2 features). Tables below are from thesis chapter 4; per-run numbers are in `results/*/evaluate.json`. Training used 1× A100 40GB, AdamW, lr 1e-4 (1e-5 compared), with either 100 fixed epochs or early stopping (patience 5 or 10).

Baseline for all fine-tuning tables: TRELLIS-text without fine-tuning (CLIP 0.3047, FD Incep 345.66, KD Incep 1.2782, FD DINO 254.05, KD DINO 1.4435).

**Fine-tuning with the original loss (thesis Table 4.2)**

| Early stopping | CLIP ↑ | FD Incep ↓ | KD Incep ↓ | FD DINO ↓ | KD DINO ↓ |
| --- | --- | --- | --- | --- | --- |
| none (100 epochs) | 0.2999 | 198.54 | 0.8687 | 130.48 | 1.9758 |
| patience 5 | 0.2939 | 339.39 | 0.0593 | 197.44 | 11.34 |

**Fine-tuning with L2-SP (Table 4.3)**

| Early stopping | CLIP ↑ | FD Incep ↓ | KD Incep ↓ | FD DINO ↓ | KD DINO ↓ |
| --- | --- | --- | --- | --- | --- |
| none (100 epochs) | 0.2911 | 380.59 | 0.5713 | 234.35 | 0.0030 |
| patience 10 | 0.2883 | 433.84 | 0.3258 | 236.23 | 1.7890 |

**Fine-tuning with LoRA (Table 4.4)**

| Early stopping | LoRA config | LR | CLIP ↑ | FD Incep ↓ | KD Incep ↓ | FD DINO ↓ | KD DINO ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (100 epochs) | 1 | 1e-4 | 0.3056 | 246.03 | 2.5777 | 196.86 | 1.9200 |
| patience 5 | 1 | 1e-4 | 0.3053 | 303.55 | 1.2312 | 216.48 | 1.5325 |
| patience 10 | 2 | 1e-4 | 0.3012 | 313.48 | 1.1835 | 213.64 | 1.5018 |
| patience 10 | 2 | 1e-5 | 0.3056 | 352.15 | 1.2530 | 251.41 | 1.4385 |

**Rectified Flow vs. Diffusion (Table 4.5)**

| Generative model | CLIP ↑ | FD Incep ↓ | KD Incep ↓ | FD DINO ↓ | KD DINO ↓ |
| --- | --- | --- | --- | --- | --- |
| Rectified Flow (25 steps) | 0.3047 | 345.66 | 1.2782 | 254.05 | 1.4435 |
| Diffusion (700 steps) | 0.2104 | 1736.49 | 3.7442 | 539.85 | 0.5428 |

**Distillation (Table 4.6)**

| Model | Params SS | Params SLat | Avg. time | CLIP ↑ | FD Incep ↓ | KD Incep ↓ | FD DINO ↓ | KD DINO ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Teacher | 185,364,616 | 768,156,536 | 8.99 s | 0.3047 | 345.66 | 1.2782 | 245.05* | 1.4435 |
| Student 1 | 52,776,200 | 81,414,280 | 3.06 s | 0.2624 | 713.36 | 1.2811 | 338.77 | 4.2861 |
| Student 2 | 14,444,168 | 25,832,968 | 2.73 s | 0.2505 | 843.68 | 1.3040 | 373.65 | 0.0011 |

\* Printed as 245.05 in the thesis; every other table gives 254.05 for the same model, so this is likely a typo.

**Comparison with other approaches (Table 4.7)**

Row mapping is inferred from the values because the labels are garbled in the PDF text.

| Model | CLIP ↑ | FD Incep ↓ | KD Incep ↓ | FD DINO ↓ | KD DINO ↓ |
| --- | --- | --- | --- | --- | --- |
| TRELLIS-text | 0.3047 | 345.66 | 1.2782 | 254.05 | 1.4435 |
| Stable Diffusion + TRELLIS-image | 0.2568 | 498.33 | 1.2593 | 319.26 | 1.3966 |
| GaussianDreamer | 0.2582 | 1410.72 | 1.5405 | 571.34 | 1.5039 |


The thesis also proposes future work on image, multi-view image and video input, and on combining modalities (depth, sketch, ControlNet).

## Acknowledgements

Built on [TRELLIS](https://github.com/microsoft/TRELLIS), [FlexiCubes](https://github.com/MaxtirError/FlexiCubes), [nvdiffrast](https://github.com/NVlabs/nvdiffrast) and [mip-splatting](https://github.com/autonomousvision/mip-splatting). The baseline in `extensions/GaussianDreamer/` retains its own license.
