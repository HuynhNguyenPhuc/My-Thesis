import torch
import torch.nn as nn
import torch.distributions as dist


def sample_logit_normal(batch_size, mu=1.0, sigma=1.0, device='cuda'):
    """
    Samples from a logit-normal distribution.

    Args:
        batch_size (int): The number of samples to generate.
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.
        device (str): The device on which to perform the computation.
    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing samples from the logit-normal distribution.
    """
    mu_tensor = torch.full((batch_size,), mu, device=device)
    sigma_tensor = torch.full((batch_size,), sigma, device=device)
    
    normal_dist = dist.Normal(mu_tensor, sigma_tensor)
    samples = normal_dist.sample()
    
    logit_normal_samples = torch.sigmoid(samples)
    
    return logit_normal_samples


def slat_collate_fn(batch):
    texts, structured_latents = zip(*batch)

    texts = list(texts)
    structured_latents = list(structured_latents)

    return texts, structured_latents


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def get_custom_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """
    AdamW optimizer that excludes bias and norm/embedding parameters from weight decay.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude bias, LayerNorm/BatchNorm, and embeddings from weight decay
        if (
            name.endswith('.bias')
            or 'bias' in name
            or 'LayerNorm' in name
            or 'layernorm' in name.lower()
            or 'ln' in name.lower()
            or 'norm' in name.lower()
            or 'embedding' in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    grouped = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(grouped, lr=lr, betas=(0.9, 0.95))


def l2sp_loss(model: torch.nn.Module, init_state: dict, alpha: float) -> torch.Tensor:
    """
    Compute L2-SP regularization loss only on parameters with requires_grad=True.
    """
    loss = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad:
            loss = loss + torch.sum((p - init_state[name]) ** 2)
    return alpha * loss