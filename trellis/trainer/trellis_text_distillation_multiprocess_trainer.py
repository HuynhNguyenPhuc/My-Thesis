from typing import *

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import json
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, CLIPTextModel
from safetensors.torch import load_file, save_file

from . import samplers
from .base import Trainer
from .utils import sample_logit_normal, slat_collate_fn, get_custom_optimizer, l2sp_loss
from ..modules import sparse as sp
from ..models import SparseStructureFlowModel, SLatFlowModel


def compute_logit_normal_probs(timesteps, mu=0.0, sigma=1.0, device='cuda'):
    """
    Compute probabilities for discrete timesteps using a logit-normal distribution.
    
    Args:
        timesteps: Tensor of discrete timesteps (e.g., [0, 0.04, ..., 1.0]).
        mu: Mean of the underlying normal distribution.
        sigma: Standard deviation of the underlying normal distribution.
        device: Device for computations.
    
    Returns:
        probs: Tensor of probabilities for each timestep.
    """
    timesteps = timesteps.clamp(1e-6, 1.0 - 1e-6)
    logits = torch.log(timesteps / (1 - timesteps))
    normal_dist = torch.distributions.Normal(mu, sigma)
    log_probs = normal_dist.log_prob(logits)
    probs = torch.exp(log_probs)
    probs = probs / probs.sum()
    return probs.to(device)


class TrellisTextDistillationMultiprocessTrainer(Trainer):
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization

        self._init_distribution()
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextDistillationMultiprocessTrainer":
        pipeline = super(TrellisTextDistillationMultiprocessTrainer, TrellisTextDistillationMultiprocessTrainer).from_pretrained(path)
        new_pipeline = TrellisTextDistillationMultiprocessTrainer()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.path = path

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_distribution()
        new_pipeline._init_text_cond_model(args['text_cond_model'])
        return new_pipeline
    

    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # Load model
        model = CLIPTextModel.from_pretrained(name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(name)
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])


    def _init_distribution(self):
        """
        Initialize the distributed training environment.
        """
        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
            self.rank       = dist.get_rank()
    
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.world_size = 1
            self.rank       = 0
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        self.to(device)

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].to(self.device)
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        

    def compute_ss_loss(
        self,
        sparse_structure: torch.Tensor,
        text_features: torch.Tensor,
        cfg_dropout: float = 0.1,
        distill_weight: float = 1.0
    ):
        """
        Trả về distillation loss giữa teacher và student.
        - distill_weight: trọng số cho distillation term.
        """

        teacher = self.models['sparse_structure_flow_model_teacher']
        student = self.models['sparse_structure_flow_model_student']
        batch_size = sparse_structure.size(0)

        # Classifier-free guidance dropout
        mask = torch.bernoulli(
            torch.full((batch_size, 1, 1), 1 - cfg_dropout, device=self.device)
        )
        text_features = text_features * mask

        # Sample noise schedule t
        timesteps = torch.arange(0.04, 1.01, 0.04, device=self.device)
        probs = compute_logit_normal_probs(timesteps, mu=1.0, sigma=1.0, device=self.device)
        indices = torch.multinomial(probs, batch_size, replacement=True)
        t = timesteps[indices]
        t_scaled = (t * 1000).to(torch.float32)

        sigma_min = self.sparse_structure_sampler.sigma_min
        t_exp = t.view(batch_size, *([1] * (sparse_structure.ndim - 1)))
        coeff_sparse = 1 - t_exp
        coeff_noise = sigma_min + (1 - sigma_min) * t_exp
        noise = torch.randn_like(sparse_structure, device=self.device)
        noised = sparse_structure * coeff_sparse + noise * coeff_noise

        gt_v = (1 - sigma_min) * noise - sparse_structure
        with torch.no_grad():
            teacher_pred_v = teacher(noised, t_scaled, cond=text_features)
        student_pred_v = student(noised, t_scaled, cond=text_features)

        # Distillation loss (MSE)
        loss_distill = 0.8 * F.mse_loss(teacher_pred_v, student_pred_v) + 0.2 * F.mse_loss(student_pred_v, gt_v)

        return distill_weight * loss_distill
    

    def compute_slat_loss(
        self, 
        structured_latent: sp.SparseTensor, 
        text_features: torch.Tensor,
        cfg_dropout: float = 0.1,
        distill_weight: float = 1.0
    ):
        teacher_flow_model = self.models['slat_flow_model_teacher']
        student_flow_model = self.models['slat_flow_model_student']
        
        # Compute the normalized SLat
        std = torch.tensor(self.slat_normalization['std'], device=self.device).view(1, -1)
        mean = torch.tensor(self.slat_normalization['mean'], device=self.device).view(1, -1)
        normalized_slat = structured_latent.replace(structured_latent.feats.clone(), structured_latent.coords.clone())
        normalized_slat.feats.sub_(mean).div_(std)
        
        feats = normalized_slat.feats
        coords = normalized_slat.coords
        batch_size = normalized_slat.shape[0]
        
        # Timestep t
        timesteps = torch.arange(0.04, 1.01, 0.04, device=self.device)
        probs = compute_logit_normal_probs(timesteps, mu=1.0, sigma=1.0, device=self.device)
        indices = torch.multinomial(probs, batch_size, replacement=True)
        t = timesteps[indices]
        t_scaled = (t * 1000).to(torch.float32)

        # Classifier Free Guidance 
        mask = torch.bernoulli(
            torch.full((text_features.shape[0], 1, 1), 1 - cfg_dropout, device=self.device)
        ).to(torch.float32)
        text_features = text_features * mask

        noise_feats = torch.randn(coords.shape[0], teacher_flow_model.in_channels, device=self.device)

        sigma_min = self.slat_sampler.sigma_min

        gt_v = (1 - sigma_min) * noise_feats - feats
        
        batch_indices = normalized_slat.coords[:, 0]
        multiplier = 1 - t[batch_indices]
        normalized_slat.feats.mul_(multiplier.unsqueeze(1))
        noise_multiplier = sigma_min + (1 - sigma_min) * t[batch_indices]
        noise_feats.mul_(noise_multiplier.unsqueeze(1))
        normalized_slat.feats.add_(noise_feats)
        noised_slat = normalized_slat
        
        with torch.no_grad():
            teacher_pred_v = teacher_flow_model(noised_slat, t_scaled, cond=text_features)
        student_pred_v = student_flow_model(noised_slat, t_scaled, cond=text_features)
        
        # MSE loss
        loss_distill = 0.8 * F.mse_loss(teacher_pred_v.feats, student_pred_v.feats) + 0.2 * F.mse_loss(student_pred_v.feats, gt_v)
        
        return distill_weight * loss_distill
    

    def distill_ss_model(
        self,
        ss_train_dataset: torch.utils.data.Dataset,
        ss_val_dataset: torch.utils.data.Dataset,
        num_epochs: int = 2000,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        save_dir: Optional[str] = None,
        save_interval: int = 10,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        smooth_window: int = 5,
        plot_loss: bool = True
    ):
        self.ss_dim = 16**3 * 8
        self.timesteps = torch.arange(0.04, 1.01, 0.04, device=self.device)
        self.probs = compute_logit_normal_probs(self.timesteps, mu=1.0, sigma=1.0, device=self.device)

        # Initialize save directory
        if not save_dir:
            save_dir = "TRELLIS-text/ss"
        
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            folders = os.listdir(save_dir)
            run_indices = sorted(int(re.search(r"run_(\d+)", folder).group(1)) for folder in folders if re.match(r"run_\d+", folder))
            max_idx = run_indices[-1] if run_indices else -1
            save_dir = os.path.join(save_dir, f"run_{max_idx + 1}")
            ckpt_dir = os.path.join(save_dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            ss_model_path = os.path.join(ckpt_dir, "ss_flow_txt_dit_B_16l8.safetensors")
            log_file_path = os.path.join(save_dir, "ss_logs.csv")
            with open(log_file_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Epoch", "Train Distill Loss", "Val Distill Loss", "Smoothed Val Loss"])
            
            # Early stopping initialization
            best_val_loss = float("inf")
            counter = 0

        # Load teacher and student models
        try:
            with open(f"{self.path}/ckpts/ss_flow_txt_dit_B_16l8.json", "r") as fp:
                ss_kwargs = json.load(fp)
            teacher_ss_model = SparseStructureFlowModel(**ss_kwargs['args']).to(self.device)
            teacher_ss_model.load_state_dict(load_file(f"{self.path}/ckpts/ss_flow_txt_dit_B_16l8.safetensors"))
            teacher_ss_model.eval()

            with open(f"{self.path}/ckpts/ss_flow_txt_dit_B_16l8_student.json", "r") as fp:
                ss_kwargs = json.load(fp)
            student_ss_model = SparseStructureFlowModel(**ss_kwargs['args']).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

        if self.is_distributed:
            student_ss_model = DDP(student_ss_model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        
        self.models['sparse_structure_flow_model_teacher'] = teacher_ss_model
        self.models['sparse_structure_flow_model_student'] = student_ss_model

        # Optimizer and Scaler
        optimizer_ss = get_custom_optimizer(self.models['sparse_structure_flow_model_student'], lr=learning_rate, weight_decay=1e-5)
        scheduler_ss = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ss,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        scaler_ss = torch.amp.GradScaler(device=self.device.type)

        # DataLoader
        train_sampler = DistributedSampler(ss_train_dataset, shuffle=True) if self.is_distributed else None
        val_sampler = DistributedSampler(ss_val_dataset, shuffle=False) if self.is_distributed else None
        train_loader = torch.utils.data.DataLoader(
            ss_train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, 
            persistent_workers=True, sampler=train_sampler, shuffle=(not self.is_distributed)
        )
        val_loader = torch.utils.data.DataLoader(
            ss_val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=False, 
            persistent_workers=True, sampler=val_sampler
        )

        # Early stopping flag
        continue_training = torch.tensor([1], device=self.device)

        if self.rank == 0:
            epoch_bar = tqdm(range(num_epochs), desc="Epochs", unit="epoch")
        else:
            epoch_bar = range(num_epochs)

        train_loss_values = []
        val_loss_values = []
        smoothed_val_values = []

        for epoch in epoch_bar:
            if continue_training.item() == 0:
                break  # Stop training if early stopping is triggered

            if self.is_distributed:
                train_sampler.set_epoch(epoch)

            # Training
            self.models['sparse_structure_flow_model_student'].train()
            total_distill_loss = torch.tensor(0.0, device=self.device)
            num_train_batches = torch.tensor(0, device=self.device)

            for text, sparse_structure in train_loader:
                sparse_structure = sparse_structure.to(self.device)
                text_features = self.encode_text(text).to(self.device)

                with torch.amp.autocast(device_type=self.device.type):
                    distill_loss = self.compute_ss_loss(sparse_structure, text_features)

                optimizer_ss.zero_grad()
                scaler_ss.scale(distill_loss).backward()
                scaler_ss.step(optimizer_ss)
                scaler_ss.update()

                total_distill_loss += distill_loss.detach()
                num_train_batches += 1

            # Compute global training loss
            if self.is_distributed:
                dist.all_reduce(total_distill_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_train_batches, op=dist.ReduceOp.SUM)
            global_avg_distill_loss = total_distill_loss.item() / num_train_batches.item() if num_train_batches.item() > 0 else 0

            # Validation
            self.models['sparse_structure_flow_model_student'].eval()
            total_val_loss = torch.tensor(0.0, device=self.device)
            num_val_batches = torch.tensor(0, device=self.device)
            with torch.no_grad():
                for text, sparse_structure in val_loader:
                    sparse_structure = sparse_structure.to(self.device)
                    text_features = self.encode_text(text).to(self.device)

                    with torch.amp.autocast(device_type=self.device.type):
                        distill_loss = self.compute_ss_loss(sparse_structure, text_features)

                    total_val_loss += distill_loss.detach()
                    num_val_batches += 1

            # Compute global validation loss
            if self.is_distributed:
                dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_val_batches, op=dist.ReduceOp.SUM)
            global_avg_val_loss = total_val_loss.item() / num_val_batches.item() if num_val_batches.item() > 0 else 0

            if self.rank == 0:
                train_loss_values.append(global_avg_distill_loss)
                val_loss_values.append(global_avg_val_loss)

                # Smoothing validation loss
                window = min(len(val_loss_values), smooth_window)
                smoothed = sum(val_loss_values[-window:]) / window
                smoothed_val_values.append(smoothed)

                epoch_bar.set_postfix({"Train Loss": f"{global_avg_distill_loss:.4f}", "Val Loss": f"{global_avg_val_loss:.4f}"})
                
                with open(log_file_path, "a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([epoch + 1, f"{global_avg_distill_loss:.4f}", f"{global_avg_val_loss:.4f}", f"{smoothed:.4f}"])

                scheduler_ss.step(smoothed)

                # Early stopping logic
                if early_stopping:
                    if smoothed < best_val_loss - min_delta:
                        best_val_loss = smoothed
                        counter = 0
                        try:
                            save_file(self.models['sparse_structure_flow_model_student'].module.state_dict(), ss_model_path)
                        except Exception as e:
                            print(f"Failed to save model at epoch {epoch + 1}: {e}")
                    else:
                        counter += 1
                    
                    if counter >= patience:
                        continue_training = torch.tensor([0], device=self.device)
                        num_epochs = epoch + 1
                        print(f"Early stopping at epoch {epoch + 1}")
                    else:
                        continue_training = torch.tensor([1], device=self.device)
                else:
                    # Save model if no early stopping or if it's the best so far
                    if smoothed < best_val_loss:
                        best_val_loss = smoothed
                        try:
                            save_file(self.models['sparse_structure_flow_model_student'].module.state_dict(), ss_model_path)
                        except Exception as e:
                            print(f"Failed to save model at epoch {epoch + 1}: {e}")

            # Broadcast continue_training flag
            if self.is_distributed:
                dist.broadcast(continue_training, src=0)

        # Plotting
        if plot_loss and self.rank == 0:
            epochs = range(1, len(train_loss_values)+1)

            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_loss_values, label="Train Loss", color="blue")
            plt.plot(epochs, val_loss_values, label="Val Loss", alpha=0.3)
            plt.plot(epochs, smoothed_val_values, label=f"Smoothed Val Loss (w={smooth_window})")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Sparse Structure Flow Model - Training & Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "ss_loss.png"), dpi=300, bbox_inches="tight")


    def distill_slat_model(
        self,
        slat_train_dataset: torch.utils.data.Dataset,
        slat_val_dataset: torch.utils.data.Dataset,
        num_epochs: int = 2000,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        save_dir: Optional[str] = None,
        save_interval: int = 10,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        smooth_window: int = 5,
        plot_loss: bool = True
    ):
        self.slat_dim = 64**3 * 8
        self.timesteps = torch.arange(0.04, 1.01, 0.04, device=self.device)
        self.probs = compute_logit_normal_probs(self.timesteps, mu=1.0, sigma=1.0, device=self.device)
        
        # Initialize save directory
        if not save_dir:
            save_dir = "TRELLIS-text/slat"
        
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            folders = os.listdir(save_dir)
            run_indices = sorted(int(re.search(r"run_(\d+)", folder).group(1)) for folder in folders if re.match(r"run_\d+", folder))
            max_idx = run_indices[-1] if run_indices else -1
            save_dir = os.path.join(save_dir, f"run_{max_idx + 1}")
            ckpt_dir = os.path.join(save_dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            slat_model_path = os.path.join(ckpt_dir, "slat_flow_txt_dit_B_64l8p2.safetensors")
            log_file_path = os.path.join(save_dir, "slat_logs.csv")
            with open(log_file_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Epoch", "Train Distill Loss", "Val Distill Loss", "Smoothed Val Loss"])
            
            # Early stopping initialization
            best_val_loss = float("inf")
            counter = 0

        # Load teacher and student models
        try:
            with open(f"{self.path}/ckpts/slat_flow_txt_dit_B_64l8p2.json", "r") as fp:
                slat_kwargs = json.load(fp)
            teacher_slat_model = SLatFlowModel(**slat_kwargs['args']).to(self.device)
            teacher_slat_model.load_state_dict(load_file(f"{self.path}/ckpts/slat_flow_txt_dit_B_64l8p2.safetensors"))
            teacher_slat_model.eval()

            with open(f"{self.path}/ckpts/slat_flow_txt_dit_B_64l8p2_student.json", "r") as fp:
                slat_kwargs = json.load(fp)
            student_slat_model = SLatFlowModel(**slat_kwargs['args']).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

        if self.is_distributed:
            student_slat_model = DDP(student_slat_model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        
        self.models['slat_flow_model_teacher'] = teacher_slat_model
        self.models['slat_flow_model_student'] = student_slat_model

        # Optimizer and Scaler
        optimizer_slat = get_custom_optimizer(self.models['slat_flow_model_student'], lr=learning_rate, weight_decay=1e-5)
        scheduler_slat = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_slat,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        scaler_slat = torch.amp.GradScaler(device=self.device.type)

        # DataLoader
        train_sampler = DistributedSampler(slat_train_dataset, shuffle=True) if self.is_distributed else None
        val_sampler = DistributedSampler(slat_val_dataset, shuffle=False) if self.is_distributed else None
        train_loader = torch.utils.data.DataLoader(
            slat_train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, 
            persistent_workers=True, sampler=train_sampler, collate_fn=slat_collate_fn, shuffle=(not self.is_distributed)
        )
        val_loader = torch.utils.data.DataLoader(
            slat_val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=False, 
            persistent_workers=True, sampler=val_sampler, collate_fn=slat_collate_fn
        )

        # Early stopping flag
        continue_training = torch.tensor([1], device=self.device)

        if self.rank == 0:
            epoch_bar = tqdm(range(num_epochs), desc="Epochs", unit="epoch")
        else:
            epoch_bar = range(num_epochs)

        train_loss_values = []
        val_loss_values = []
        smoothed_val_values = []

        for epoch in epoch_bar:
            if continue_training.item() == 0:
                break  # Stop training if early stopping is triggered

            if self.is_distributed:
                train_sampler.set_epoch(epoch)

            # Training
            self.models['slat_flow_model_student'].train()
            total_distill_loss = torch.tensor(0.0, device=self.device)
            num_train_batches = torch.tensor(0, device=self.device)

            for text, structured_latent in train_loader:
                structured_latent = sp.sparse_cat(structured_latent, dim=0).to(self.device)
                text_features = self.encode_text(text).to(self.device)

                with torch.amp.autocast(device_type=self.device.type):
                    distill_loss = self.compute_slat_loss(structured_latent, text_features)

                optimizer_slat.zero_grad()
                scaler_slat.scale(distill_loss).backward()
                scaler_slat.step(optimizer_slat)
                scaler_slat.update()

                total_distill_loss += distill_loss.detach()
                num_train_batches += 1

            # Compute global training loss
            if self.is_distributed:
                dist.all_reduce(total_distill_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_train_batches, op=dist.ReduceOp.SUM)
            global_avg_distill_loss = total_distill_loss.item() / num_train_batches.item() if num_train_batches.item() > 0 else 0

            # Validation
            self.models['slat_flow_model_student'].eval()
            total_val_loss = torch.tensor(0.0, device=self.device)
            num_val_batches = torch.tensor(0, device=self.device)
            with torch.no_grad():
                for text, structured_latent in val_loader:
                    structured_latent = sp.sparse_cat(structured_latent, dim=0).to(self.device)
                    text_features = self.encode_text(text).to(self.device)

                    with torch.amp.autocast(device_type=self.device.type):
                        distill_loss = self.compute_slat_loss(structured_latent, text_features)

                    total_val_loss += distill_loss.detach()
                    num_val_batches += 1

            # Compute global validation loss
            if self.is_distributed:
                dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_val_batches, op=dist.ReduceOp.SUM)
            global_avg_val_loss = total_val_loss.item() / num_val_batches.item() if num_val_batches.item() > 0 else 0

            if self.rank == 0:
                train_loss_values.append(global_avg_distill_loss)
                val_loss_values.append(global_avg_val_loss)

                # Smoothing validation loss
                window = min(len(val_loss_values), smooth_window)
                smoothed = sum(val_loss_values[-window:]) / window
                smoothed_val_values.append(smoothed)

                epoch_bar.set_postfix({"Train Loss": f"{global_avg_distill_loss:.4f}", "Val Loss": f"{global_avg_val_loss:.4f}"})
                
                with open(log_file_path, "a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([epoch + 1, f"{global_avg_distill_loss:.4f}", f"{global_avg_val_loss:.4f}", f"{smoothed:.4f}"])

                scheduler_slat.step(smoothed)

                # Early stopping logic
                if early_stopping:
                    if smoothed < best_val_loss - min_delta:
                        best_val_loss = smoothed
                        counter = 0
                        try:
                            save_file(self.models['slat_flow_model_student'].module.state_dict(), slat_model_path)
                        except Exception as e:
                            print(f"Failed to save model at epoch {epoch + 1}: {e}")
                    else:
                        counter += 1
                    
                    if counter >= patience:
                        continue_training = torch.tensor([0], device=self.device)
                        num_epochs = epoch + 1
                        print(f"Early stopping at epoch {epoch + 1}")
                    else:
                        continue_training = torch.tensor([1], device=self.device)
                else:
                    # Save model if no early stopping or if it's the best so far
                    if smoothed < best_val_loss:
                        best_val_loss = smoothed
                        try:
                            save_file(self.models['slat_flow_model_student'].module.state_dict(), slat_model_path)
                        except Exception as e:
                            print(f"Failed to save model at epoch {epoch + 1}: {e}")

            # Broadcast continue_training flag
            if self.is_distributed:
                dist.broadcast(continue_training, src=0)

        # Plotting
        if plot_loss and self.rank == 0:
            epochs = range(1, len(train_loss_values)+1)

            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_loss_values, label="Train Loss", color="blue")
            plt.plot(epochs, val_loss_values, label="Val Loss", alpha=0.3)
            plt.plot(epochs, smoothed_val_values, label=f"Smoothed Val Loss (w={smooth_window})")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("SLat Flow Model - Training & Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "slat_loss.png"), dpi=300, bbox_inches="tight")