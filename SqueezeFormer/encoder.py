# Copyright (c) 2022, Sangchun Ha. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adapted for SqueezeFormer XS replication - Lorea project

from typing import Tuple, Optional

import torch.nn as nn
from torch import Tensor
from .attention import MultiHeadedSelfAttentionModule
from .convolution import ConvModule, DepthwiseConv2dSubsampling, TimeReductionLayer
from .modules import FeedForwardModule, ResidualConnectionModule, AdaptiveScale, recover_resolution


class SqueezeformerBlock(nn.Module):
    """
    SqueezeformerBlock: MHA → norm → FFN → norm → Conv → norm → FFN → norm.

    Simpler than Conformer's Macaron structure; each attention/conv module
    is directly followed by a single feed-forward module.

    adaptive_scale=True uses learnable per-channel scalars (paper micro-arch contribution)
    instead of LayerNorm, matching the official TF implementation.

    half_step_residual=True (default, matching paper) scales FFN residuals by 0.5.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        adaptive_scale: bool = True,
    ):
        super().__init__()
        ff_factor = 0.5 if half_step_residual else 1.0

        def norm_layer():
            return AdaptiveScale(encoder_dim) if adaptive_scale else nn.LayerNorm(encoder_dim)

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            norm_layer(),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=ff_factor,
            ),
            norm_layer(),
            ResidualConnectionModule(
                module=ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            norm_layer(),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=ff_factor,
            ),
            norm_layer(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class SqueezeformerEncoder(nn.Module):
    """
    SqueezeformerEncoder with Temporal U-Net structure.

    Processes input via convolution subsampling, then stacked SqueezeformerBlocks
    with time reduction at reduce_layer_index and recovery at recover_layer_index.

    Args:
        input_dim: Number of input mel features (default 80).
        encoder_dim: Model dimension (XS=144, S=256, M=512, L=512).
        num_layers: Number of SqueezeformerBlocks (default 16).
        reduce_layer_index: Layer at which to halve temporal resolution (default 7).
        recover_layer_index: Layer at which to restore temporal resolution (default 15).
        num_attention_heads: Number of MHA heads (XS=4).
        feed_forward_expansion_factor: FFN expansion (default 4).
        conv_expansion_factor: Conv expansion for GLU (must be 2).
        input_dropout_p: Dropout after input projection.
        feed_forward_dropout_p: FFN dropout.
        attention_dropout_p: Attention dropout.
        conv_dropout_p: Convolution module dropout.
        conv_kernel_size: Depthwise conv kernel size (default 31).
        half_step_residual: Use 0.5 scaling on FFN residuals.
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
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
        half_step_residual: bool = True,
        adaptive_scale: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.reduce_layer_index = reduce_layer_index
        self.recover_layer_index = recover_layer_index

        self.conv_subsample = DepthwiseConv2dSubsampling(in_channels=1, out_channels=encoder_dim)
        # With SAME-style padding (pad+1 before each stride-2 conv): freq_out = ceil(input_dim / 4)
        subsampled_freq = (input_dim + 3) // 4
        self.input_proj = nn.Sequential(
            nn.Linear(encoder_dim * subsampled_freq, encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )

        self.time_reduction_layer = TimeReductionLayer(encoder_dim=encoder_dim)
        # time_reduction_proj no longer needed: TimeReductionLayer now has its own pw_conv
        self.time_recover_layer = nn.Linear(encoder_dim, encoder_dim)

        block_kwargs = dict(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            adaptive_scale=adaptive_scale,
        )

        self.layers = nn.ModuleList([SqueezeformerBlock(**block_kwargs) for _ in range(num_layers)])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_proj(outputs)

        recover_tensor: Optional[Tensor] = None

        for idx, layer in enumerate(self.layers):
            if idx == self.reduce_layer_index:
                recover_tensor = outputs
                outputs, output_lengths = self.time_reduction_layer(outputs, output_lengths)

            if idx == self.recover_layer_index:
                outputs = recover_resolution(outputs)
                length = outputs.size(1)
                outputs = self.time_recover_layer(outputs)
                outputs = outputs + recover_tensor[:, :length, :]
                output_lengths = output_lengths * 2

            outputs = layer(outputs)

        # Clamp final output_lengths to actual tensor size to prevent CTC assertion errors
        output_lengths = output_lengths.clamp(max=outputs.size(1))
        return outputs, output_lengths
