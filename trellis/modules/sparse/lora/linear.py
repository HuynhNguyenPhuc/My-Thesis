
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter

from ..basic import SparseTensor
from ..linear import SparseLinear


class LoRASparseLinear(nn.Module):
    def __init__(self, layer: SparseLinear, r=4, alpha=1.0):
        super().__init__()
        self.linear = layer
        self.r = r
        self.alpha = alpha
        self.in_features = layer.in_features
        self.out_features = layer.out_features

        # Freeze original parameters
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA low-rank matrices
        self.lora_A = Parameter(torch.zeros(r, self.in_features))
        self.lora_B = Parameter(torch.zeros(self.out_features, r))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: SparseTensor):
        base_out = self.linear(x)
        lora_feats = (self.alpha / self.r) * ((x.feats @ self.lora_A.t()) @ self.lora_B.t())
        combined_feats = base_out.feats + lora_feats
        return base_out.replace(combined_feats)