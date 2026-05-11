# Copyright (c) 2022, Sangchun Ha. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adapted for SqueezeFormer XS replication - Lorea project

from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activation import Swish
from .modules import Transpose


class DepthwiseConv2dSubsampling(nn.Module):
    """
    Depthwise 2D convolution subsampling to 1/4 sequence length.

    Matches the official TF implementation (kssteven418/Squeezeformer, ds=True):
    - Pads input by (0,1) in time and freq before each conv (SAME-style)
    - Conv1: standard Conv2d, stride=2, kernel=3
    - Conv2: depthwise Conv2d (groups=out_channels), stride=2, kernel=3
    - ReLU after each conv
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.conv2_dw = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=0, groups=out_channels
        )
        self.conv2_pw = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        # inputs: (B, T, F)
        outputs = inputs.unsqueeze(1)                        # (B, 1, T, F)
        outputs = F.pad(outputs, (0, 1, 0, 1))               # pad freq+1, time+1
        outputs = F.relu(self.conv1(outputs))                # (B, C, T/2, F/2)
        outputs = F.pad(outputs, (0, 1, 0, 1))               # pad freq+1, time+1
        outputs = self.conv2_dw(outputs)                     # (B, C, T/4, F/4) depthwise
        outputs = F.relu(self.conv2_pw(outputs))             # (B, C, T/4, F/4) pointwise

        batch_size, channels, subsampled_lengths, subsampled_dim = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)

        # Clamp to actual tensor length — formula approximations diverge for some input lengths
        output_lengths = ((input_lengths + 3) // 4).clamp(max=subsampled_lengths)
        return outputs, output_lengths


class DepthwiseConv2d(nn.Module):
    """2D depthwise convolution where groups == in_channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super().__init__()
        assert out_channels % in_channels == 0, "out_channels must be a multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class DepthwiseConv1d(nn.Module):
    """1D depthwise convolution where groups == in_channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert out_channels % in_channels == 0, "out_channels must be a multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """Pointwise (1x1) 1D convolution for dimension matching."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    """
    Convolution module: pointwise -> Swish -> depthwise -> BatchNorm -> Swish -> pointwise.
    Input/output shape: (batch, time, dim).
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd for 'SAME' padding"
        assert expansion_factor == 2, "Only expansion_factor=2 is supported"
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            Swish(),
            DepthwiseConv1d(
                in_channels * expansion_factor,
                in_channels * expansion_factor,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(in_channels * expansion_factor),
            Swish(),
            PointwiseConv1d(in_channels * expansion_factor, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor, pad_mask: Tensor = None) -> Tensor:
        outputs = inputs.transpose(1, 2)
        outputs = self.sequential[1](outputs)
        outputs = self.sequential[2](outputs)
        if pad_mask is not None:
            outputs = outputs * pad_mask.unsqueeze(1).to(dtype=outputs.dtype)
        outputs = self.sequential[3](outputs)
        outputs = self.sequential[4](outputs)
        outputs = self.sequential[5](outputs)
        outputs = self.sequential[6](outputs)
        outputs = self.sequential[7](outputs)
        return outputs.transpose(1, 2)


class TimeReductionLayer(nn.Module):
    """
    Temporal downsampling matching the official TF implementation.

    Applies a depthwise conv only along the time axis (kernel=(kernel_size, 1)),
    then a pointwise conv to restore the feature dimension.
    This matches kssteven418/Squeezeformer: dw_conv kernel=(5,1) + pw_conv kernel=1.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        kernel_size: int = 5,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        # Depthwise conv along time only: input reshaped to (B, T, 1, E)
        # kernel=(kernel_size, 1) operates on (T, 1), groups=1 → depthwise over channels
        self.dw_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=0,
            groups=1,
        )
        # Pointwise projection back to encoder_dim
        self.pw_conv = nn.Conv1d(
            in_channels=encoder_dim,
            out_channels=encoder_dim,
            kernel_size=1,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, E = inputs.size()

        # Pad time dimension: padding = max(0, kernel_size - stride)
        pad = max(0, self.kernel_size - self.stride)
        outputs = inputs  # (B, T, E)
        if pad > 0:
            outputs = F.pad(outputs, (0, 0, 0, pad))  # pad time dim

        # Reshape to (B, 1, T+pad, E) for 2D conv along time only
        outputs = outputs.unsqueeze(1)                          # (B, 1, T+pad, E)
        outputs = self.dw_conv(outputs)                         # (B, 1, T', E)
        outputs = outputs.squeeze(1)                            # (B, T', E)

        # Pointwise projection: (B, T', E) → (B, E, T') → conv → (B, E, T') → (B, T', E)
        outputs = outputs.transpose(1, 2)                       # (B, E, T')
        outputs = self.pw_conv(outputs)                         # (B, E, T')
        outputs = outputs.transpose(1, 2)                       # (B, T', E)

        T_out = outputs.size(1)
        output_lengths = ((input_lengths + pad - self.kernel_size) // self.stride + 1).clamp(max=T_out)

        return outputs, output_lengths
