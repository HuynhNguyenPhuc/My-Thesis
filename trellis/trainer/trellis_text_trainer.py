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
from .utils import sample_logit_normal, slat_collate_fn
from ..modules import sparse as sp
from ..models import SparseStructureFlowModel, SLatFlowModel


class TrellisTextTrainer(Trainer):
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
    def from_pretrained(path: str) -> "TrellisTextTrainer":
        pipeline = super(TrellisTextTrainer, TrellisTextTrainer).from_pretrained(path)
        new_pipeline = TrellisTextTrainer()
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
        

    def compute_loss_ss(
        self, 
        sparse_structure: torch.Tensor, 
        text_features: torch.Tensor,
        cfg_dropout: float = 0.1
    ):
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.module.resolution

        batch_size = sparse_structure.shape[0]

        # Classifier Free Guidance 
        mask = torch.bernoulli(
            torch.full((text_features.shape[0], 1, 1), 1 - cfg_dropout, device=self.device)
        ).to(torch.float32)
        text_features = text_features * mask

        t = sample_logit_normal(batch_size, device=self.device)
        t_expanded = t.view(batch_size, *([1] * (sparse_structure.ndim - 1)))

        noise = torch.randn(batch_size, flow_model.module.in_channels, reso, reso, reso, device=self.device)
        clone_noise = noise.clone()
        

        sigma_min = self.sparse_structure_sampler.sigma_min
        coef_sparse = 1 - t_expanded
        coef_noise = sigma_min + (1 - sigma_min) * t_expanded
        
        noised_sparse_structure = sparse_structure.clone().mul_(coef_sparse)
        clone_noise.mul_(coef_noise)
        noised_sparse_structure.add_(clone_noise)
        

        pred_v = flow_model(noised_sparse_structure, (t * 1000).to(torch.float32), cond=text_features)
        
        gt_v = (1 - sigma_min) * noise - sparse_structure
        
        # MSE Loss
        loss = F.mse_loss(pred_v, gt_v)
        
        return loss
    

    def compute_loss_slat(
        self, 
        structured_latent: sp.SparseTensor, 
        text_features: torch.Tensor,
        cfg_dropout: float = 0.1
    ):
        flow_model = self.models['slat_flow_model']
        
        # Compute the normalized SLat
        std = torch.tensor(self.slat_normalization['std'], device=self.device).view(1, -1)
        mean = torch.tensor(self.slat_normalization['mean'], device=self.device).view(1, -1)
        normalized_slat = structured_latent.replace(structured_latent.feats.clone(), structured_latent.coords.clone())
        normalized_slat.feats.sub_(mean).div_(std)
        
        feats = normalized_slat.feats
        coords = normalized_slat.coords
        batch_size = normalized_slat.shape[0]
        
        # Timestep t
        t = sample_logit_normal(batch_size, device=self.device)

        # Classifier Free Guidance 
        mask = torch.bernoulli(
            torch.full((text_features.shape[0], 1, 1), 1 - cfg_dropout, device=self.device)
        ).to(torch.float32)
        text_features = text_features * mask

        noise_feats = torch.randn(coords.shape[0], flow_model.module.in_channels, device=self.device)

        sigma_min = self.slat_sampler.sigma_min

        gt_v = (1 - sigma_min) * noise_feats - feats
        
        batch_indices = normalized_slat.coords[:, 0]
        multiplier = 1 - t[batch_indices]
        normalized_slat.feats.mul_(multiplier.unsqueeze(1))
        noise_multiplier = sigma_min + (1 - sigma_min) * t[batch_indices]
        noise_feats.mul_(noise_multiplier.unsqueeze(1))
        normalized_slat.feats.add_(noise_feats)
        noised_slat = normalized_slat
        
        pred_v = flow_model(noised_slat, (t * 1000).to(torch.float32), cond=text_features)
        
        # MSE loss
        loss = F.mse_loss(pred_v.feats, gt_v)
        
        return loss
    

    def train_ss_model(
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
        smooth_window: int = 10,
        plot_loss: bool = True
    ):
        self.ss_dim = 16**3 * 8
        
        if not save_dir:
            save_dir = "TRELLIS-text/ss"
        
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            folders = os.listdir(save_dir)
            run_indices = sorted(int(re.search(r"run_(\d+)", folder).group(1)) for folder in folders if re.match(r"run_\d+", folder))
            max_idx = run_indices[-1] if run_indices else -1
            save_dir = os.path.join(save_dir, f"run_{max_idx+1}")
            ckpt_dir = os.path.join(save_dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            ss_model_path = os.path.join(ckpt_dir, "ss_flow_txt_dit_B_16l8.safetensors")
            log_file_path = os.path.join(save_dir, "ss_logs.csv")
            with open(log_file_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Epoch", "Train Loss", "Val Loss", "Smoothed Val Loss"])
            
            # Early stopping initialization
            best_val_loss = float("inf")
            counter = 0

        # Load model, convert to PEFT, then wrap in DDP for distributed training
        with open(f"{self.path}/ckpts/ss_flow_txt_dit_B_16l8.json", "r") as fp:
            ss_kwargs = json.load(fp)
        ss_model = SparseStructureFlowModel(**ss_kwargs['args']).to(self.device)
        # ss_model.load_state_dict(load_file(f"{self.path}/ckpts/ss_flow_txt_dit_B_16l8.safetensors"))
        if self.is_distributed:
            ss_model = DDP(ss_model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        self.models['sparse_structure_flow_model'] = ss_model

        # Optimizer and Scaler
        optimizer_ss = torch.optim.AdamW(self.models['sparse_structure_flow_model'].parameters(), lr=learning_rate)
        warmup_epochs = 10
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer_ss,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
        )
        scheduler_ss = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ss,
            mode='min',
            factor=0.5,
            patience=5
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
        continue_training = torch.tensor([1], device = self.device)

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
            self.models['sparse_structure_flow_model'].train()
            total_loss_ss = torch.tensor(0.0, device=self.device)
            num_train_batches = torch.tensor(0, device=self.device)

            for text, sparse_structure in train_loader:
                sparse_structure = sparse_structure.to(self.device)
                text_features = self.encode_text(text).to(self.device)

                with torch.amp.autocast(device_type=self.device.type):
                    loss_ss = self.compute_loss_ss(sparse_structure, text_features)

                optimizer_ss.zero_grad()
                scaler_ss.scale(loss_ss).backward()
                # grad_norm = torch.nn.utils.clip_grad_norm_(
                #     self.models['sparse_structure_flow_model'].module.parameters() if self.is_distributed 
                #     else self.models['sparse_structure_flow_model'].parameters(), max_norm=1.0)
                scaler_ss.step(optimizer_ss)
                scaler_ss.update()

                total_loss_ss += loss_ss.detach()
                num_train_batches += 1

            # Compute global training loss
            if self.is_distributed:
                dist.all_reduce(total_loss_ss, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_train_batches, op=dist.ReduceOp.SUM)
            global_avg_train_loss = total_loss_ss.item() / num_train_batches.item() if num_train_batches.item() > 0 else 0

            # Validation
            self.models['sparse_structure_flow_model'].eval()
            total_val_loss = torch.tensor(0.0, device=self.device)
            num_val_batches = torch.tensor(0, device=self.device)
            with torch.no_grad():
                for text, sparse_structure in val_loader:
                    sparse_structure = sparse_structure.to(self.device)
                    text_features = self.encode_text(text).to(self.device)

                    with torch.amp.autocast(device_type=self.device.type):
                        loss_ss = self.compute_loss_ss(sparse_structure, text_features)

                    total_val_loss += loss_ss.detach()
                    num_val_batches += 1

            # Compute global validation loss
            if self.is_distributed:
                dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_val_batches, op=dist.ReduceOp.SUM)
            global_avg_val_loss = total_val_loss.item() / num_val_batches.item() if num_val_batches.item() > 0 else 0

            if self.rank == 0:
                train_loss_values.append(global_avg_train_loss)
                val_loss_values.append(global_avg_val_loss)

                # Smoothing validation loss
                window = min(len(val_loss_values), smooth_window)
                smoothed = sum(val_loss_values[-window:]) / window
                smoothed_val_values.append(smoothed)

                epoch_bar.set_postfix({"Train Loss": f"{global_avg_train_loss:.4f}", "Val Loss": f"{global_avg_val_loss:.4f}", "Smoothed Val Loss": f"{smoothed:.4f}"})
                
                with open(log_file_path, "a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([epoch+1, f"{global_avg_train_loss:.4f}", f"{global_avg_val_loss:.4f}", f"{smoothed:.4f}"])

                # Learning rate scheduling
                if epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    scheduler_ss.step(smoothed)

                # Early stopping logic using smoothed validation loss
                if early_stopping:
                    if smoothed < best_val_loss - min_delta:
                        best_val_loss = smoothed
                        counter = 0
                        save_file(self.models['sparse_structure_flow_model'].state_dict(), ss_model_path)
                    else:
                        counter += 1
                    
                    if counter >= patience:
                        continue_training = torch.tensor([0], device = self.device)
                        num_epochs = epoch + 1
                        print(f"Early stopping at epoch {epoch+1}")
                    else:
                        continue_training = torch.tensor([1], device = self.device)
                else:
                    # Save model if no early stopping or if it's the best so far
                    if smoothed < best_val_loss:
                        best_val_loss = smoothed
                        save_file(self.models['sparse_structure_flow_model'].state_dict(), ss_model_path)

            # Broadcast continue_training flag
            if self.is_distributed:
                dist.broadcast(continue_training, src=0)

        # Plotting
        if plot_loss and self.rank == 0:
            epochs = range(1, len(train_loss_values)+1)
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_loss_values, label="Train Loss", color="blue")
            plt.plot(epochs, val_loss_values, label="Val Loss", color="red", alpha=0.5)
            plt.plot(epochs, smoothed_val_values, label=f"Smoothed Val Loss (w={smooth_window})", color="green")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Sparse Structure Flow Model - Training & Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "ss_loss.png"), dpi=300, bbox_inches="tight")


    def train_slat_model(
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
        smooth_window: int = 10,
        plot_loss: bool = True
    ):
        self.slat_dim = 64**3 * 8
        
        if not save_dir:
            save_dir = "TRELLIS-text/slat"
        
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            folders = os.listdir(save_dir)
            run_indices = sorted(int(re.search(r"run_(\d+)", folder).group(1)) for folder in folders if re.match(r"run_\d+", folder))
            max_idx = run_indices[-1] if run_indices else -1
            save_dir = os.path.join(save_dir, f"run_{max_idx+1}")
            ckpt_dir = os.path.join(save_dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            slat_model_path = os.path.join(ckpt_dir, "slat_flow_txt_dit_B_64l8p2.safetensors")
            log_file_path = os.path.join(save_dir, "slat_logs.csv")
            with open(log_file_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Epoch", "Train Loss", "Val Loss", "Smoothed Val Loss"])
            
            # Early stopping initialization
            best_val_loss = float("inf")
            counter = 0

        # Load model, convert to PEFT, then wrap in DDP for distributed training
        with open(f"{self.path}/ckpts/slat_flow_txt_dit_B_64l8p2.json", "r") as fp:
            slat_kwargs = json.load(fp)
        slat_model = SLatFlowModel(**slat_kwargs['args']).to(self.device)
        if self.is_distributed:
            slat_model = DDP(slat_model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        self.models['slat_flow_model'] = slat_model

        # Optimizer and Scaler
        optimizer_slat = torch.optim.AdamW(self.models['slat_flow_model'].parameters(), lr=learning_rate)
        warmup_epochs = 10
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer_slat,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
        )
        scheduler_slat = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_slat,
            mode='min',
            factor=0.5,
            patience=5
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
        continue_training = torch.tensor([1], device = self.device)

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
            self.models['slat_flow_model'].train()
            total_loss_slat = torch.tensor(0.0, device=self.device)
            num_train_batches = torch.tensor(0, device=self.device)

            for text, structured_latent in train_loader:
                structured_latent = sp.sparse_cat(structured_latent, dim=0).to(self.device)
                text_features = self.encode_text(text).to(self.device)

                with torch.amp.autocast(device_type=self.device.type):
                    loss_slat = self.compute_loss_slat(structured_latent, text_features)

                optimizer_slat.zero_grad()
                scaler_slat.scale(loss_slat).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.models['slat_flow_model'].module.parameters() if self.is_distributed 
                    else self.models['slat_flow_model'].parameters(), max_norm=1.0
                )
                scaler_slat.step(optimizer_slat)
                scaler_slat.update()

                total_loss_slat += loss_slat.detach()
                num_train_batches += 1

            # Compute global training loss
            if self.is_distributed:
                dist.all_reduce(total_loss_slat, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_train_batches, op=dist.ReduceOp.SUM)
            global_avg_train_loss = total_loss_slat.item() / num_train_batches.item() if num_train_batches.item() > 0 else 0

            # Validation
            self.models['slat_flow_model'].eval()
            total_val_loss = torch.tensor(0.0, device=self.device)
            num_val_batches = torch.tensor(0, device=self.device)
            with torch.no_grad():
                for text, structured_latent in val_loader:
                    structured_latent = sp.sparse_cat(structured_latent, dim=0).to(self.device)
                    text_features = self.encode_text(text).to(self.device)

                    with torch.amp.autocast(device_type=self.device.type):
                        loss_slat = self.compute_loss_slat(structured_latent, text_features)

                    total_val_loss += loss_slat.detach()
                    num_val_batches += 1

            # Compute global validation loss
            if self.is_distributed:
                dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_val_batches, op=dist.ReduceOp.SUM)
            global_avg_val_loss = total_val_loss.item() / num_val_batches.item() if num_val_batches.item() > 0 else 0

            if self.rank == 0:
                train_loss_values.append(global_avg_train_loss)
                val_loss_values.append(global_avg_val_loss)

                # Smoothing validation loss
                window = min(len(val_loss_values), smooth_window)
                smoothed = sum(val_loss_values[-window:]) / window
                smoothed_val_values.append(smoothed)
    
                epoch_bar.set_postfix({"Train Loss": f"{global_avg_train_loss:.4f}", "Val Loss": f"{global_avg_val_loss:.4f}", "Smoothed Val Loss": f"{smoothed:.4f}"})
                
                with open(log_file_path, "a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([epoch+1, f"{global_avg_train_loss:.4f}", f"{global_avg_val_loss:.4f}", f"{smoothed:.4f}"])
    
                # Learning rate scheduling
                if epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    scheduler_slat.step(smoothed)
    
                # Early stopping logic using smoothed validation loss
                if early_stopping:
                    if smoothed < best_val_loss - min_delta:
                        best_val_loss = smoothed
                        counter = 0
                        save_file(self.models['slat_flow_model'].state_dict() if self.is_distributed else self.models['slat_flow_model'].state_dict(), slat_model_path)
                    else:
                        counter += 1
                    
                    if counter >= patience:
                        continue_training = torch.tensor([0], device=self.device)
                        num_epochs = epoch + 1
                        print(f"Early stopping at epoch {epoch+1}")
                    else:
                        continue_training = torch.tensor([1], device=self.device)
                else:
                    # Save model if no early stopping or if it's the best so far
                    if smoothed < best_val_loss:
                        best_val_loss = smoothed
                        save_file(self.models['slat_flow_model'].state_dict() if self.is_distributed else self.models['slat_flow_model'].state_dict(), slat_model_path)

            # Broadcast continue_training flag
            if self.is_distributed:
                dist.broadcast(continue_training, src=0)

        # Plotting
        if plot_loss and self.rank == 0:
            epochs = range(1, len(train_loss_values)+1)
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_loss_values, label="Train Loss", color="blue")
            plt.plot(epochs, val_loss_values, label="Val Loss", color="red", alpha=0.5)
            plt.plot(epochs, smoothed_val_values, label=f"Smoothed Val Loss (w={smooth_window})", color="green")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("SLat Flow Model - Training & Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "slat_loss.png"), dpi=300, bbox_inches="tight")