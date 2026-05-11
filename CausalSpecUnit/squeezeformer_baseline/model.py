# Copyright (c) 2022, Sangchun Ha. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adapted for SqueezeFormer XS replication - Lorea project

from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder import SqueezeformerEncoder


class Squeezeformer(nn.Module):
    """
    SqueezeFormer ASR model.

    Combines a Temporal U-Net encoder (SqueezeformerEncoder) with a linear
    classification head and log-softmax output for CTC training.

    Args:
        num_classes: Vocabulary size (number of output tokens).
        input_dim: Mel feature dimension (default 80).
        encoder_dim: Model dimension.
        num_encoder_layers: Number of SqueezeformerBlocks.
        reduce_layer_index: Layer index where temporal downsampling occurs.
        recover_layer_index: Layer index where temporal upsampling occurs.
        num_attention_heads: MHA heads.
        feed_forward_expansion_factor: FFN expansion factor.
        conv_expansion_factor: Conv module expansion (must be 2).
        input_dropout_p: Dropout on input projection.
        feed_forward_dropout_p: FFN dropout.
        attention_dropout_p: Attention dropout.
        conv_dropout_p: Conv module dropout.
        conv_kernel_size: Depthwise conv kernel size.
        half_step_residual: Use 0.5 scaling on FFN residuals.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_encoder_layers: int = 16,
        reduce_layer_index: int = 7,
        recover_layer_index: int = 15,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = False,
        adaptive_scale: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SqueezeformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            reduce_layer_index=reduce_layer_index,
            recover_layer_index=recover_layer_index,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            adaptive_scale=adaptive_scale,
        )
        self.fc = nn.Linear(encoder_dim, num_classes, bias=True)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs)
        outputs = F.log_softmax(outputs, dim=-1)
        return outputs, encoder_output_lengths
