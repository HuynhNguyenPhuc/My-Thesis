import warnings
warnings.filterwarnings("ignore")

import sys
import os
os.environ['ATTN_BACKEND'] = 'xformers'

import torch.distributed as dist


def disable_logs():
    if dist.is_initialized() and dist.get_rank() != 0:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')


def main():
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        disable_logs()
        rank = dist.get_rank()
        print(f"Starting process with rank {rank}")
    else:
        rank = 0

    from trellis.datasets import TextConditionedStructuredLatent
    from trellis.trainer import TrellisTextDiffusionMultiprocessTrainer

    data_dir = "datasets/House3K"
    train_dataset = TextConditionedStructuredLatent(data_dir, "train", 2100)
    val_dataset   = TextConditionedStructuredLatent(data_dir, "val", 600)

    trainer = TrellisTextDiffusionMultiprocessTrainer.from_pretrained("./TRELLIS-text-train")

    trainer.train_slat_model(
        slat_train_dataset = train_dataset,
        slat_val_dataset   = val_dataset,
        batch_size         = 16,
        num_epochs         = 200,
        save_interval      = 10,
        early_stopping     = True,
        patience           = 10,
        smooth_window      = 10
    )

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

# Chạy với 4 GPU:
# torchrun --nproc_per_node=4 -m scripts.train.train_slat_diffusion
