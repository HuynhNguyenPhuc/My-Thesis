import warnings
warnings.filterwarnings("ignore")

import sys
import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    from trellis.trainer import TrellisTextLoRAMultiprocessTrainer

    data_dir = "datasets/House3K"
    train_dataset = TextConditionedSparseStructure(data_dir, "train", 2100)
    val_dataset   = TextConditionedSparseStructure(data_dir, "val",   300)

    trainer = TrellisTextLoRAMultiprocessTrainer.from_pretrained("./TRELLIS-text-train")

    trainer.finetune_ss_model(
        ss_train_dataset = train_dataset,
        ss_val_dataset   = val_dataset,
        batch_size       = 64,
        learning_rate    = 1e-5,
        num_epochs       = 100,
        save_interval    = 10,
        save_dir         = "experiments/LoRA/ss",
        early_stopping   = True,
        patience         = 10,
        smooth_window    = 10
    )

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=4 finetune_ss_ddp.py