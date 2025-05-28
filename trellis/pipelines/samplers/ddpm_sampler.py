import torch
import torch.nn.functional as F
from torch import nn
from typing import *
from .utils import gather
from .base import Sampler
from ...modules import sparse as sp
class DDPM_Sampler(Sampler):
    def __init__(self, n_steps: int, device: torch.device = 'cuda'):
        super().__init__()
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = (gather(self.alpha_bar, t, x0.ndim) ** 0.5) * x0
        var = 1 - gather(self.alpha_bar, t, x0.ndim)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, model, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        
        eps_theta = model(x_t, t, cond)
        assert eps_theta.shape == x_t.shape, f"Shape mismatch: eps_theta {eps_theta.shape}, x_t {x_t.shape}"
        alpha_bar = gather(self.alpha_bar, t, x_t.ndim)
        alpha = gather(self.alpha, t, x_t.ndim)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (x_t - eps_coef * eps_theta)
        var = gather(self.sigma2, t, x_t.ndim)
        if isinstance(x_t, sp.SparseTensor):
            eps = x_t.replace(torch.randn_like(x_t.feats))
        else:
            eps = torch.randn_like(x_t)
        return mean + (var ** 0.5) * eps
    def sample(
        self,
        model,
        noise: torch.Tensor,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = None,
        verbose: bool = True,
) -> torch.Tensor:
        x_t = noise
        if rescale_t is None:
            rescale_t = self.n_steps / steps
        for step in reversed(range(steps)):
            t_val = int(step * rescale_t)
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.long)
            if verbose:
                # print(f"[Step {step}/{steps}] t = {t_val}")
                pass
            x_t = self.p_sample(model, x_t, cond=cond, t=t)
        print('steps', steps)
        return x_t

    # def sample(
    #     self,
    #     model,
    #     noise,
    #     cond: Optional[Any] = None,
    #     steps: int = 50,
    #     eta: float = 0.0,  # 0 = deterministic DDIM
    #     verbose: bool = True,
    # ) -> torch.Tensor:
    #     x_t = noise
    #     device = x_t.device
    #     total_steps = self.n_steps

    #     times = torch.linspace(0, total_steps - 1, steps + 1, dtype=torch.long, device=device)
    #     times_next = torch.cat([times[1:], torch.tensor([0], device=device)])

    #     for i in range(steps):
    #         t = times[i]
    #         t_next = times_next[i]

    #         t_batch = torch.full((x_t.shape[0],), t.item(), device=device, dtype=torch.long)
    #         t_next_batch = torch.full_like(t_batch, t_next.item())

    #         alpha_t = gather(self.alpha_bar, t_batch, x_t.ndim)
    #         alpha_t_next = gather(self.alpha_bar, t_next_batch, x_t.ndim)

    #         eps_theta = model(x_t, t_batch, cond)

    #         # Predict x0
    #         x0_pred = (x_t - (1 - alpha_t).sqrt() * eps_theta) / alpha_t.sqrt()

    #         # DDIM update
    #         sigma = eta * ((1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)).sqrt()
    #         # noise = torch.randn_like(x_t) if eta > 0 else 0
    #         if eta > 0:
    #             if isinstance(x_t, sp.SparseTensor):
    #                 noise = x_t.replace(torch.randn_like(x_t.feats))
    #             else:
    #                 noise = torch.randn_like(x_t)
    #         else:
    #             noise = 0

    #         x_t = alpha_t_next.sqrt() * x0_pred + (1 - alpha_t_next - sigma**2).sqrt() * eps_theta + sigma * noise

    #         if verbose:
    #             # print(f"[DDIM Step {i}/{steps}] t = {t.item()} -> {t_next.item()}")
    #             pass

    #     return x_t
        
    def loss(self, model, x0: torch.Tensor, cond: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps=noise)
        eps_theta = model(x_t, t, cond)
        return F.mse_loss(noise, eps_theta)