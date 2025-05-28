import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from typing import *
from .utils import gather
from ...modules import sparse as sp

class DDPM_Sampler:
    def __init__(self, n_steps: int, device):
        super().__init__()
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            device = torch.device(f"cuda:{local_rank}")
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta
        self.device = device

    def q_xt_x0(self, x0, t):
        mean = (gather(self.alpha_bar.to(t.device), t, x0.ndim) ** 0.5) * x0
        var = 1 - gather(self.alpha_bar.to(t.device), t, x0.ndim)
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, model, x_t, cond, t):
        eps_theta = model(x_t, t, cond)
        assert eps_theta.shape == x_t.shape, f"Shape mismatch: eps_theta {eps_theta.shape}, x_t {x_t.shape}"
        alpha_bar = gather(self.alpha_bar.to(t.device), t, x_t.ndim)
        alpha = gather(self.alpha.to(t.device), t, x_t.ndim)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (x_t - eps_coef * eps_theta)
        var = gather(self.sigma2.to(t.device), t, x_t.ndim)
        eps = torch.randn_like(x_t)
        return mean + (var ** 0.5) * eps

    def sample(self, model, noise: torch.Tensor, cond: Optional[Any] = None, steps: int = 50, rescale_t: float = None, verbose: bool = True):
        x_t = noise
        if rescale_t is None:
            rescale_t = self.n_steps / steps

        for step in reversed(range(steps)):
            t_val = int(step * rescale_t)
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.long)
            if verbose:
                print(f"[Step {step}/{steps}] t = {t_val}")
            x_t = self.p_sample(model, x_t, cond=cond, t=t)
        return x_t

    def loss(self, model, x0, cond, noise=None):
        batch_size = x0.shape[0]
        x0 = x0.to(self.device)
        if noise is not None:
            noise = noise.to(self.device)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps=noise)
        eps_theta = model(x_t, t, cond)
        if isinstance(x_t, sp.SparseTensor):
            assert eps_theta.feats.shape == noise.feats.shape, f"Feats shape mismatch: eps_theta {eps_theta.feats.shape}, noise {noise.feats.shape}"
            return F.mse_loss(noise.feats, eps_theta.feats)
        return F.mse_loss(noise, eps_theta)