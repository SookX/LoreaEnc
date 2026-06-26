import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from CausalSpecUnit.squeezeformer_baseline import Squeezeformer, get_config
from CausalSpecUnit.squeezeformer_baseline.convolution import DepthwiseConv2dSubsampling


SQUEEZEFORMER_VARIANTS = ("xs", "s", "sm", "m", "m95", "ml", "l")
MELHUBERT_TRANSFORMER_VARIANTS = ("mh9m",)
MODEL_VARIANTS = SQUEEZEFORMER_VARIANTS + MELHUBERT_TRANSFORMER_VARIANTS


@dataclass(frozen=True)
class MelHuBERTTransformerConfig:
    input_dim: int = 80
    encoder_dim: int = 240
    num_encoder_layers: int = 10
    num_attention_heads: int = 4
    feed_forward_dim: int = 960
    dropout_p: float = 0.1
    conv_pos_kernel: int = 128
    conv_pos_groups: int = 16


MELHUBERT_TRANSFORMER_CONFIGS = {
    # Roughly 9M parameters with the CTC head, depending on vocab size.
    # This is intentionally a compact Transformer baseline, not the original
    # MelHuBERT Base-size architecture.
    "mh9m": MelHuBERTTransformerConfig(),
}


def is_melhubert_transformer_variant(variant: str) -> bool:
    return variant.lower() in MELHUBERT_TRANSFORMER_CONFIGS


def get_encoder_dim(variant: str) -> int:
    variant = variant.lower()
    if is_melhubert_transformer_variant(variant):
        return MELHUBERT_TRANSFORMER_CONFIGS[variant].encoder_dim
    return get_config(variant).encoder_dim


class ConvPositionalEmbedding(nn.Module):
    """HuBERT/MelHuBERT-style convolutional positional embedding."""

    def __init__(self, dim: int, kernel_size: int = 128, groups: int = 16, dropout_p: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        pos = self.conv(inputs.transpose(1, 2)).transpose(1, 2)
        pos = pos[:, : inputs.size(1), :]
        return inputs + self.dropout(self.activation(pos))


class MelHuBERTTransformerEncoder(nn.Module):
    """Compact mel-input Transformer encoder for the MelHuBERT-style baseline.

    The 2-D convolutional subsampler keeps the output rate aligned with the
    existing chunk-size/stride 8/4 targets. Above that, the stack is deliberately
    plain Transformer: linear mel projection, convolutional positional
    embedding, and self-attention/FFN blocks.
    """

    def __init__(
        self,
        config: MelHuBERTTransformerConfig,
        layer_drop_p: float = 0.0,
    ):
        super().__init__()
        self.num_layers = config.num_encoder_layers
        self.layer_drop_p = float(layer_drop_p)
        self.encoder_dim = config.encoder_dim
        self.reduce_layer_index = -1
        self.recover_layer_index = -1

        self.conv_subsample = DepthwiseConv2dSubsampling(
            in_channels=1,
            out_channels=config.encoder_dim,
        )
        subsampled_freq = (config.input_dim + 3) // 4
        self.input_proj = nn.Sequential(
            nn.Linear(config.encoder_dim * subsampled_freq, config.encoder_dim),
            nn.Dropout(config.dropout_p),
        )
        self.input_layer_norm = nn.LayerNorm(config.encoder_dim)
        self.pos_conv = ConvPositionalEmbedding(
            config.encoder_dim,
            kernel_size=config.conv_pos_kernel,
            groups=config.conv_pos_groups,
            dropout_p=config.dropout_p,
        )
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.encoder_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.feed_forward_dim,
                dropout=config.dropout_p,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            for _ in range(config.num_encoder_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.encoder_dim)

    @staticmethod
    def _padding_mask(lengths: Tensor, max_length: int) -> Tensor:
        steps = torch.arange(max_length, device=lengths.device)
        return steps.unsqueeze(0) >= lengths.unsqueeze(1)

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        intermediate_layers: Optional[Iterable[int]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Dict[int, Tuple[Tensor, Tensor]]]]:
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_layer_norm(self.input_proj(outputs))
        outputs = self.pos_conv(outputs)

        wanted = set(intermediate_layers) if intermediate_layers else None
        intermediates: Optional[Dict[int, Tuple[Tensor, Tensor]]] = (
            {} if wanted is not None else None
        )

        for idx, layer in enumerate(self.layers):
            mask = self._padding_mask(output_lengths, outputs.size(1))
            drop_this_layer = (
                self.training
                and self.layer_drop_p > 0.0
                and torch.rand((), device=outputs.device).item() < self.layer_drop_p
            )
            if not drop_this_layer:
                outputs = layer(outputs, src_key_padding_mask=mask)
            outputs = outputs.masked_fill(mask.unsqueeze(-1), 0.0)
            if wanted is not None and idx in wanted:
                intermediates[idx] = (outputs.clone(), output_lengths.clamp(max=outputs.size(1)))

        outputs = self.final_layer_norm(outputs)
        output_lengths = output_lengths.clamp(max=outputs.size(1))
        return outputs, output_lengths, intermediates


def build_melhubert_transformer_encoder(variant: str, layer_drop_p: float = 0.0):
    variant = variant.lower()
    if variant not in MELHUBERT_TRANSFORMER_CONFIGS:
        raise ValueError(f"Unknown MelHuBERT-style Transformer variant: {variant}")
    return MelHuBERTTransformerEncoder(
        MELHUBERT_TRANSFORMER_CONFIGS[variant],
        layer_drop_p=layer_drop_p,
    )


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


def build_encoder(variant: str, layer_drop_p: float = 0.0):
    variant = variant.lower()
    if is_melhubert_transformer_variant(variant):
        return build_melhubert_transformer_encoder(variant, layer_drop_p=layer_drop_p)
    return build_copied_squeezeformer(variant, num_classes=1, layer_drop_p=layer_drop_p).encoder


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
        teacher_momentum: float = 0.0,
    ):
        super().__init__()
        self.variant = variant
        self.encoder_dim = get_encoder_dim(variant)
        self.encoder = build_encoder(variant, layer_drop_p=layer_drop_p)
        self.head_coarse = nn.Linear(self.encoder_dim, k_coarse)
        self.head_fine = nn.Linear(self.encoder_dim, k_fine)
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
            self.aux_head_coarse[key] = nn.Linear(self.encoder_dim, k_coarse)
            self.aux_head_fine[key] = nn.Linear(self.encoder_dim, k_fine)
        # EMA teacher for data2vec-style continuous-feature distillation. The
        # teacher is a (non-trainable) deepcopy of the encoder updated each
        # optimizer step toward the student with momentum ``teacher_momentum``.
        # The student is trained to predict the teacher's encoder features at
        # masked positions, in addition to the cluster prediction objective.
        # teacher_momentum = 0.0 disables (no teacher built, no extra memory).
        self.teacher_momentum = float(teacher_momentum)
        self.teacher: nn.Module = None  # type: ignore[assignment]
        if self.teacher_momentum > 0.0:
            self._build_teacher()

    def _build_teacher(self):
        """Materialize the EMA teacher as a frozen deepcopy of the encoder.

        Registered as a regular submodule so the teacher state participates in
        save_checkpoint / load_checkpoint round-trips — preserving the EMA
        history across resumes. Teacher params have ``requires_grad=False`` so
        they never receive gradients; the optimizer never sees them."""
        if self.teacher is not None:
            return
        teacher = copy.deepcopy(self.encoder)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        self.teacher = teacher

    @torch.no_grad()
    def teacher_encode(self, mel, lengths):
        """Run the EMA teacher on (clean) mel, return (features, lengths).

        Always invoked under no_grad. Caller passes the **unmasked** mel so the
        teacher sees the ground-truth audio while the student sees mask_emb at
        the same positions — the asymmetry is what makes the distillation a
        meaningful self-supervisory signal."""
        if self.teacher is None:
            raise RuntimeError("teacher not initialized; pass teacher_momentum > 0 at __init__")
        feat, out_lengths, _ = self.teacher(mel, lengths)
        return feat, out_lengths

    @torch.no_grad()
    def update_teacher_ema(self):
        """EMA update: teacher = momentum * teacher + (1 - momentum) * student.

        Called once per optimizer step. Skipped if teacher is disabled or
        momentum is 0/1 (in which case the teacher would be either ignored or
        a hard copy)."""
        if self.teacher is None or self.teacher_momentum <= 0.0:
            return
        m = self.teacher_momentum
        for src, dst in zip(self.encoder.parameters(), self.teacher.parameters()):
            dst.data.mul_(m).add_(src.data, alpha=1.0 - m)
        # Buffers (BatchNorm running stats etc.) are copied directly — they
        # don't have a "rate of change" interpretation for EMA.
        for src, dst in zip(self.encoder.buffers(), self.teacher.buffers()):
            dst.data.copy_(src.data)

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
        # ``encoded`` is exposed so the training loop can use it as the source
        # for data2vec-style EMA-teacher distillation. None of the existing
        # cluster-prediction code paths depend on it; ignore it if you don't.
        return coarse, fine, out_lengths, aux, encoded


class CausalSpecUnitCTC(nn.Module):
    """CTC wrapper around the copied SqueezeFormer baseline.

    inter_ctc_layers: 0-indexed encoder block indices at which to attach an
    auxiliary CTC head. Empty/None disables InterCTC.

    ssl_anchor: when True, adds K_c=100 and K_f=500 prediction heads (matching
    the SSL pretraining objective). The training loop can then anchor the
    encoder to its SSL feature space by predicting cluster IDs during CTC
    fine-tuning. Heads are randomly initialized — they get optimized alongside
    the CTC head. This costs ~92k extra params (Linear(144,100) + Linear(144,500)),
    well under 1% of the encoder.
    """

    def __init__(
        self,
        vocab_size,
        variant="xs",
        inter_ctc_layers=None,
        layer_drop_p: float = 0.0,
        ssl_anchor: bool = False,
        ssl_k_coarse: int = 100,
        ssl_k_fine: int = 500,
    ):
        super().__init__()
        self.variant = variant
        self.inter_ctc_layers = tuple(int(i) for i in (inter_ctc_layers or ()))
        self.ssl_anchor = bool(ssl_anchor)
        self.is_transformer_baseline = is_melhubert_transformer_variant(variant)
        self.inter_ctc_heads = nn.ModuleDict()
        self.fc = None
        if self.is_transformer_baseline:
            self.model = None
            self.encoder = build_melhubert_transformer_encoder(variant, layer_drop_p=layer_drop_p)
            self.fc = nn.Linear(get_encoder_dim(variant), vocab_size, bias=True)
            if self.inter_ctc_layers:
                self.add_inter_ctc_heads(self.inter_ctc_layers)
        else:
            self.model = build_copied_squeezeformer(variant, num_classes=vocab_size, layer_drop_p=layer_drop_p)
            self.encoder = self.model.encoder
            if self.inter_ctc_layers:
                self.model.add_inter_ctc_heads(self.inter_ctc_layers)
        if self.ssl_anchor:
            encoder_dim = get_encoder_dim(variant)
            self.ssl_head_coarse = nn.Linear(encoder_dim, ssl_k_coarse)
            self.ssl_head_fine = nn.Linear(encoder_dim, ssl_k_fine)

    def add_inter_ctc_heads(self, layer_indices):
        if not self.is_transformer_baseline:
            self.model.add_inter_ctc_heads(layer_indices)
            return
        encoder_dim = get_encoder_dim(self.variant)
        for idx in layer_indices:
            key = str(int(idx))
            if key not in self.inter_ctc_heads:
                self.inter_ctc_heads[key] = nn.Linear(encoder_dim, self.fc.out_features, bias=True)

    def forward(self, mel, lengths, return_inter: bool = False, return_ssl: bool = False):
        if self.is_transformer_baseline:
            wanted = sorted(int(k) for k in self.inter_ctc_heads.keys()) if return_inter else None
            encoder_outputs, out_lengths, intermediates = self.encoder(
                mel,
                lengths,
                intermediate_layers=wanted,
            )
            log_probs = F.log_softmax(self.fc(encoder_outputs), dim=-1)
            inter_outputs = None
            if intermediates is not None:
                inter_outputs = {}
                for idx, (feat, feat_lengths) in intermediates.items():
                    head = self.inter_ctc_heads[str(idx)]
                    inter_outputs[idx] = (F.log_softmax(head(feat), dim=-1), feat_lengths)
            ssl_outputs = None
            if return_ssl and self.ssl_anchor:
                ssl_outputs = (
                    self.ssl_head_coarse(encoder_outputs),
                    self.ssl_head_fine(encoder_outputs),
                )
            return log_probs, out_lengths, inter_outputs, ssl_outputs

        return_features = return_ssl and self.ssl_anchor
        log_probs, out_lengths, inter_outputs, encoder_features = self.model(
            mel, lengths,
            return_inter=return_inter and bool(self.inter_ctc_layers),
            return_features=return_features,
        )
        ssl_outputs = None
        if return_features:
            ssl_outputs = (
                self.ssl_head_coarse(encoder_features),
                self.ssl_head_fine(encoder_features),
            )
        return log_probs, out_lengths, inter_outputs, ssl_outputs

    def load_ssl_encoder(self, checkpoint_path, map_location="cpu", load_ssl_heads: bool = False):
        """Load the encoder weights from an SSL checkpoint.

        When ``load_ssl_heads`` is True (default False), also transfer the SSL
        cluster prediction heads (``head_coarse`` / ``head_fine``) into this
        model's ``ssl_head_coarse`` / ``ssl_head_fine``. This warm-starts the
        anchor objective with the SSL-trained head weights instead of random
        init, so the auxiliary loss is meaningful from step 1.
        """
        state = torch.load(checkpoint_path, map_location=map_location)
        model_state = state["model"] if "model" in state else state
        encoder_state = {}
        ssl_head_coarse_state = {}
        ssl_head_fine_state = {}
        for key, value in model_state.items():
            key = key.removeprefix("module.").removeprefix("_orig_mod.")
            if key.startswith("encoder."):
                encoder_state[key[len("encoder."):]] = value
            elif key.startswith("head_coarse."):
                ssl_head_coarse_state[key[len("head_coarse."):]] = value
            elif key.startswith("head_fine."):
                ssl_head_fine_state[key[len("head_fine."):]] = value
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if load_ssl_heads and self.ssl_anchor:
            if ssl_head_coarse_state:
                self.ssl_head_coarse.load_state_dict(ssl_head_coarse_state, strict=True)
            if ssl_head_fine_state:
                self.ssl_head_fine.load_state_dict(ssl_head_fine_state, strict=True)
        return missing, unexpected
