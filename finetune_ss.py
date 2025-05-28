import warnings
warnings.filterwarnings("ignore")

import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from trellis.datasets import TextConditionedSparseStructure
from trellis.trainer import TrellisTextLoRATrainer

data_dir = "datasets/ABO"
train_dataset = TextConditionedSparseStructure(data_dir, "train", 2100)
val_dataset = TextConditionedSparseStructure(data_dir, "val", 300)

trainer = TrellisTextLoRATrainer.from_pretrained("./TRELLIS-text-train")
trainer.cuda()

trainer.finetune_ss_model(
    ss_train_dataset = train_dataset, 
    ss_val_dataset = val_dataset, 
    batch_size = 64, 
    num_epochs = 100,
    save_interval = 10
)