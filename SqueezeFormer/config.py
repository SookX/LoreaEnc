# SqueezeFormer model variant configurations
# Source: Table 1 in "SqueezeFormer: An Efficient Transformer for Automatic Speech Recognition"
# https://arxiv.org/abs/2206.00888

from dataclasses import dataclass


@dataclass
class SqueezeFormerConfig:
    # Model architecture
    encoder_dim: int
    num_encoder_layers: int
    num_attention_heads: int
    feed_forward_expansion_factor: int
    conv_expansion_factor: int
    conv_kernel_size: int
    reduce_layer_index: int
    recover_layer_index: int
    # half_step_residual=True: FFN residuals scaled by 0.5 (paper default, TF fc_factor=0.5)
    half_step_residual: bool = True
    # adaptive_scale=True: learnable per-channel scalars replace LayerNorm (paper micro-arch)
    adaptive_scale: bool = True

    # Input
    input_dim: int = 80          # 80-dim log-mel spectrogram

    # Dropout
    input_dropout_p: float = 0.1
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1


# Paper Table 1 variants
SQUEEZEFORMER_XS = SqueezeFormerConfig(
    encoder_dim=144,
    num_encoder_layers=16,
    num_attention_heads=4,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    reduce_layer_index=7,
    recover_layer_index=15,
)

SQUEEZEFORMER_S = SqueezeFormerConfig(
    encoder_dim=256,
    num_encoder_layers=16,
    num_attention_heads=4,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    reduce_layer_index=7,
    recover_layer_index=15,
)

SQUEEZEFORMER_M = SqueezeFormerConfig(
    encoder_dim=512,
    num_encoder_layers=16,
    num_attention_heads=8,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    reduce_layer_index=7,
    recover_layer_index=15,
)

SQUEEZEFORMER_L = SqueezeFormerConfig(
    encoder_dim=512,
    num_encoder_layers=16,
    num_attention_heads=8,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    reduce_layer_index=7,
    recover_layer_index=15,
)

VARIANTS = {
    "xs": SQUEEZEFORMER_XS,
    "s": SQUEEZEFORMER_S,
    "m": SQUEEZEFORMER_M,
    "l": SQUEEZEFORMER_L,
}


def get_config(variant: str = "xs") -> SqueezeFormerConfig:
    variant = variant.lower()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANTS.keys())}")
    return VARIANTS[variant]
