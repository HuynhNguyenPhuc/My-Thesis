import warnings
warnings.filterwarnings("ignore")

import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from trellis.datasets import TextConditionedStructuredLatent
from trellis.trainer import TrellisTextLoRATrainer

data_dir = "datasets/ABO"
train_dataset = TextConditionedStructuredLatent(data_dir, "train", 32)
val_dataset = TextConditionedStructuredLatent(data_dir, "val", 32)

trainer = TrellisTextLoRATrainer.from_pretrained("./TRELLIS-text-train")
trainer.cuda()

trainer.finetune_slat_model(
    slat_train_dataset = train_dataset,
    slat_val_dataset = val_dataset, 
    batch_size = 4, 
    num_epochs = 1000,
    save_interval = 10
)
