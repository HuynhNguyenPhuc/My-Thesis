import warnings
warnings.filterwarnings("ignore")

import sys
import os
os.environ['ATTN_BACKEND'] = 'flash-attn'

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

    from trellis.datasets import TextConditionedSparseStructure
    from trellis.trainer import TrellisTextTrainer

    data_dir = "datasets/ABO"
    train_dataset = TextConditionedSparseStructure(data_dir, "train", 21)
    val_dataset   = TextConditionedSparseStructure(data_dir, "val", 6)

    trainer = TrellisTextTrainer.from_pretrained("./TRELLIS-text-train")

    trainer.train_ss_model(
        ss_train_dataset   = train_dataset,
        ss_val_dataset     = val_dataset,
        batch_size         = 4,
        learning_rate      = 1e-4,
        save_dir           = "TRELLIS-text/flow/ss",
        num_epochs         = 1000,
        save_interval      = 10,
        early_stopping     = False,
        patience           = 10,
        min_delta          = 0.0025,
        smooth_window      = 10
    )

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

# Chạy với 4 GPU:
# torchrun --nproc_per_node=4 train_ss.py