# SqueezeFormer model variant configurations.
# Source: Table 2 in "SqueezeFormer: An Efficient Transformer for Automatic Speech Recognition"
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
    # half_step_residual=False matches the official SqueezeFormer configs (encoder_fc_factor=1.0).
    half_step_residual: bool = False
    # adaptive_scale=True: apply learnable scale/bias after each post-LayerNorm.
    adaptive_scale: bool = True

    # Input
    input_dim: int = 80          # 80-dim log-mel spectrogram

    # Dropout
    input_dropout_p: float = 0.1
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1


# Paper Table 2 variants.
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
    encoder_dim=196,
    num_encoder_layers=18,
    num_attention_heads=4,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    reduce_layer_index=8,
    recover_layer_index=17,
)

SQUEEZEFORMER_SM = SqueezeFormerConfig(
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
    encoder_dim=324,
    num_encoder_layers=20,
    num_attention_heads=4,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    reduce_layer_index=9,
    recover_layer_index=19,
)

SQUEEZEFORMER_ML = SqueezeFormerConfig(
    encoder_dim=512,
    num_encoder_layers=18,
    num_attention_heads=8,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    reduce_layer_index=8,
    recover_layer_index=17,
)

SQUEEZEFORMER_L = SqueezeFormerConfig(
    encoder_dim=640,
    num_encoder_layers=22,
    num_attention_heads=8,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    reduce_layer_index=10,
    recover_layer_index=21,
)

VARIANTS = {
    "xs": SQUEEZEFORMER_XS,
    "s": SQUEEZEFORMER_S,
    "sm": SQUEEZEFORMER_SM,
    "m": SQUEEZEFORMER_M,
    "ml": SQUEEZEFORMER_ML,
    "l": SQUEEZEFORMER_L,
}


def get_config(variant: str = "xs") -> SqueezeFormerConfig:
    variant = variant.lower()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANTS.keys())}")
    return VARIANTS[variant]
