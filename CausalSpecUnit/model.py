import copy

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
        teacher_momentum: float = 0.0,
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
        self.model = build_copied_squeezeformer(variant, num_classes=vocab_size, layer_drop_p=layer_drop_p)
        self.encoder = self.model.encoder
        self.inter_ctc_layers = tuple(int(i) for i in (inter_ctc_layers or ()))
        if self.inter_ctc_layers:
            self.model.add_inter_ctc_heads(self.inter_ctc_layers)
        self.ssl_anchor = bool(ssl_anchor)
        if self.ssl_anchor:
            encoder_dim = self.model.fc.in_features
            self.ssl_head_coarse = nn.Linear(encoder_dim, ssl_k_coarse)
            self.ssl_head_fine = nn.Linear(encoder_dim, ssl_k_fine)

    def forward(self, mel, lengths, return_inter: bool = False, return_ssl: bool = False):
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

