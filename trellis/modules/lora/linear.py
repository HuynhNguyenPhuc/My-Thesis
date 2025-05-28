import torch
import torch.nn as nn
import numpy as np

class LoRALinear(nn.Module):
    def __init__(self, layer: nn.Linear, r=4, alpha=1.0):
        """
        Wraps a pretrained nn.Linear layer with LoRA.
        
        Args:
            linear_layer: An existing nn.Linear layer to be wrapped.
            r: The low-rank dimension for the LoRA adjustment.
            alpha: The scaling factor for the LoRA adjustment.
        """
        super().__init__()
        self.linear = layer
        self.r = r
        self.alpha = alpha

        self.in_features = layer.in_features
        self.out_features = layer.out_features

        # Freeze the original linear layer parameters
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (self.alpha / self.r) * (x @ self.lora_A.t() @ self.lora_B.t())
        return base_out + lora_out