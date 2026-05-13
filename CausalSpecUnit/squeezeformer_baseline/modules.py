# Copyright (c) 2022, Sangchun Ha. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adapted for SqueezeFormer XS replication - Lorea project

import math
import torch
import torch.nn as nn
from torch import Tensor

from .activation import Swish


class FeedForwardModule(nn.Module):
    """Feed Forward Module with pre-norm residual units, Swish activation, and dropout."""

    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding with positive and negative position embeddings."""

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb


class ResidualConnectionModule(nn.Module):
    """Residual connection: outputs = module(inputs) * module_factor + inputs."""

    def __init__(self, module: nn.Module, module_factor: float = 1.0) -> None:
        super().__init__()
        self.module = module
        self.module_factor = module_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + inputs


class ScaleBias(nn.Module):
    """
    Learnable per-channel affine scaling used after post-LayerNorm.

    SqueezeFormer replaces the following module's pre-LayerNorm with this
    cheaper transform: Scaling(x) = gamma * x + beta.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.scale + self.bias


class Transpose(nn.Module):
    """Sequential-compatible wrapper for torch.transpose."""

    def __init__(self, shape: tuple) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


def recover_resolution(inputs: Tensor) -> Tensor:
    """Double sequence length by duplicating each timestep (nearest-neighbor ×2 upsample)."""
    return inputs.repeat_interleave(2, dim=1)
