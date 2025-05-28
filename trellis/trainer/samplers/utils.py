import torch.utils.data
def gather(consts: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Gather consts for t and reshape to match input tensor's dimensions"""
    c = consts.gather(-1, t)
    return c.reshape(-1, *[1 for _ in range(ndim - 1)])