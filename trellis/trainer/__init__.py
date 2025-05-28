from . import samplers
from .trellis_text_trainer import TrellisTextTrainer
from .trellis_text_trainer_diffusion import TrellisTextTrainerDiffusion
from .trellis_text_diffusion_multiprocess_trainer import TrellisTextDiffusionMultiprocessTrainer
from .trellis_text_post_trainer import TrellisTextPostTrainer
from .trellis_text_multiprocess_post_trainer import TrellisTextMultiprocessPostTrainer
from .trellis_text_lora_trainer import TrellisTextLoRATrainer
from .trellis_text_lora_multiprocess_trainer import TrellisTextLoRAMultiprocessTrainer
from .trellis_text_distillation_multiprocess_trainer import TrellisTextDistillationMultiprocessTrainer
from .lora_config import SS_LORA_CONFIG, SLAT_LORA_CONFIG

def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)
