from torch import nn


def param_count(module: nn.Module):
    return sum(p.numel() for p in module.parameters())
