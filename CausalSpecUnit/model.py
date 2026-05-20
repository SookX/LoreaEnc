import torch
import torch.nn as nn

from CausalSpecUnit.squeezeformer_baseline import Squeezeformer, get_config


def build_copied_squeezeformer(variant, num_classes, layer_drop_p: float = 0.0):
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
        layer_drop_p=layer_drop_p,
    )


class CausalSpecUnitSSL(nn.Module):
    """
    SSL wrapper around a copied SqueezeFormer encoder.

    Includes a learnable mask token (n_mels-dim vector) to mark masked
    spectrogram frames during pretraining, distinguishable from both real
    spectral values and the zero-padded tail.

    aux_layer_indices: 0-indexed encoder block indices at which to attach
    auxiliary k-means prediction heads. Forces intermediate layers to be
    directly discriminative on the same z100/z500 targets, which (a) gives
    the lower stack a strong gradient signal and (b) makes the bottleneck
    representation transferable for downstream CTC. Default 8 = U-net
    bottleneck (one layer past time-reduction at idx 7 in XS).
    """

    def __init__(
        self,
        variant="xs",
        k_coarse=100,
        k_fine=500,
        n_mels=80,
        layer_drop_p: float = 0.0,
        aux_layer_indices=None,
    ):
        super().__init__()
        cfg = get_config(variant)
        backbone = build_copied_squeezeformer(variant, num_classes=k_fine, layer_drop_p=layer_drop_p)
        self.variant = variant
        self.encoder_dim = cfg.encoder_dim
        self.encoder = backbone.encoder
        self.head_coarse = nn.Linear(cfg.encoder_dim, k_coarse)
        self.head_fine = nn.Linear(cfg.encoder_dim, k_fine)
        # Learnable mask vector applied to masked mel frames during SSL.
        # Initialized small so early gradients flow naturally.
        self.mask_emb = nn.Parameter(torch.empty(n_mels).normal_(mean=0.0, std=0.1))
        # Auxiliary SSL heads at intermediate encoder layers. Keyed by str(idx)
        # so the dict round-trips through state_dict cleanly.
        self.aux_layer_indices = tuple(int(i) for i in (aux_layer_indices or ()))
        self.aux_head_coarse: nn.ModuleDict = nn.ModuleDict()
        self.aux_head_fine: nn.ModuleDict = nn.ModuleDict()
        for idx in self.aux_layer_indices:
            key = str(int(idx))
            self.aux_head_coarse[key] = nn.Linear(cfg.encoder_dim, k_coarse)
            self.aux_head_fine[key] = nn.Linear(cfg.encoder_dim, k_fine)

    def forward(self, mel, lengths):
        wanted = sorted(self.aux_layer_indices) if self.aux_layer_indices else None
        encoded, out_lengths, intermediates = self.encoder(mel, lengths, intermediate_layers=wanted)
        coarse = self.head_coarse(encoded)
        fine = self.head_fine(encoded)
        # Optional intermediate predictions, returned as a dict keyed by encoder
        # block index. Each value is (coarse_logits, fine_logits, lengths).
        aux = None
        if intermediates is not None:
            aux = {}
            for idx, (feat, feat_lengths) in intermediates.items():
                key = str(int(idx))
                aux[idx] = (
                    self.aux_head_coarse[key](feat),
                    self.aux_head_fine[key](feat),
                    feat_lengths,
                )
        return coarse, fine, out_lengths, aux


class CausalSpecUnitCTC(nn.Module):
    """CTC wrapper around the copied SqueezeFormer baseline.

    inter_ctc_layers: 0-indexed encoder block indices at which to attach an
    auxiliary CTC head. Empty/None disables InterCTC.
    """

    def __init__(self, vocab_size, variant="xs", inter_ctc_layers=None, layer_drop_p: float = 0.0):
        super().__init__()
        self.variant = variant
        self.model = build_copied_squeezeformer(variant, num_classes=vocab_size, layer_drop_p=layer_drop_p)
        self.encoder = self.model.encoder
        self.inter_ctc_layers = tuple(int(i) for i in (inter_ctc_layers or ()))
        if self.inter_ctc_layers:
            self.model.add_inter_ctc_heads(self.inter_ctc_layers)

    def forward(self, mel, lengths, return_inter: bool = False):
        return self.model(mel, lengths, return_inter=return_inter and bool(self.inter_ctc_layers))

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

