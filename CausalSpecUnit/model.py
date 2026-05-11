import torch
import torch.nn as nn

from CausalSpecUnit.squeezeformer_baseline import Squeezeformer, get_config


def build_copied_squeezeformer(variant, num_classes):
    """Build the copied SqueezeFormer baseline from CausalSpecUnit/squeezeformer_baseline."""
    cfg = get_config(variant)
    return Squeezeformer(
        num_classes=num_classes,
        input_dim=cfg.input_dim,
        encoder_dim=cfg.encoder_dim,
        num_encoder_layers=cfg.num_encoder_layers,
        reduce_layer_index=cfg.reduce_layer_index,
        recover_layer_index=cfg.recover_layer_index,
        num_attention_heads=cfg.num_attention_heads,
        feed_forward_expansion_factor=cfg.feed_forward_expansion_factor,
        conv_expansion_factor=cfg.conv_expansion_factor,
        input_dropout_p=cfg.input_dropout_p,
        feed_forward_dropout_p=cfg.feed_forward_dropout_p,
        attention_dropout_p=cfg.attention_dropout_p,
        conv_dropout_p=cfg.conv_dropout_p,
        conv_kernel_size=cfg.conv_kernel_size,
        half_step_residual=cfg.half_step_residual,
        adaptive_scale=cfg.adaptive_scale,
    )


class CausalSpecUnitSSL(nn.Module):
    """
    SSL wrapper around a copied SqueezeFormer encoder.

    The architecture lives in CausalSpecUnit/squeezeformer_baseline, copied from
    the repo baseline so experiments can diverge without editing SqueezeFormer/.
    """

    def __init__(self, variant="xs", k_coarse=100, k_fine=500):
        super().__init__()
        cfg = get_config(variant)
        backbone = build_copied_squeezeformer(variant, num_classes=k_fine)
        self.variant = variant
        self.encoder_dim = cfg.encoder_dim
        self.encoder = backbone.encoder
        self.head_coarse = nn.Linear(cfg.encoder_dim, k_coarse)
        self.head_fine = nn.Linear(cfg.encoder_dim, k_fine)

    def forward(self, mel, lengths):
        encoded, out_lengths = self.encoder(mel, lengths)
        return self.head_coarse(encoded), self.head_fine(encoded), out_lengths


class CausalSpecUnitCTC(nn.Module):
    """CTC wrapper around the copied SqueezeFormer baseline."""

    def __init__(self, vocab_size, variant="xs"):
        super().__init__()
        self.variant = variant
        self.model = build_copied_squeezeformer(variant, num_classes=vocab_size)
        self.encoder = self.model.encoder

    def forward(self, mel, lengths):
        return self.model(mel, lengths)

    def load_ssl_encoder(self, checkpoint_path, map_location="cpu"):
        state = torch.load(checkpoint_path, map_location=map_location)
        model_state = state["model"] if "model" in state else state
        encoder_state = {}
        for key, value in model_state.items():
            key = key.removeprefix("module.").removeprefix("_orig_mod.")
            if key.startswith("encoder."):
                encoder_state[key[len("encoder."):]] = value
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        return missing, unexpected

