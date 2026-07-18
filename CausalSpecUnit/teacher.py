"""
Frozen teacher wrapper for DistilHuBERT-style knowledge distillation.

Loads a released HuBERT Base or wav2vec 2.0 Base checkpoint via torchaudio
pipelines and exposes a forward() that returns frame-level features from one
or more chosen transformer layers.

Frame-rate note
---------------
HuBERT Base / wav2vec 2.0 Base output at ~50 fps (20 ms hop). SqueezeFormer-XS
outputs at ~25 fps after its 4x conv subsampling. With ``downsample=True`` we
2x mean-pool the teacher time axis so its feature tensors align (up to a
1-frame boundary) with the student output for the distillation loss.
"""

import torch
import torch.nn as nn
import torchaudio

SUPPORTED_TEACHERS = ("hubert_base", "wav2vec2_base")

# Transformer hidden dim for each Base pipeline.
TEACHER_DIMS = {
    "hubert_base": 768,
    "wav2vec2_base": 768,
}

_PIPELINES = {
    "hubert_base": "HUBERT_BASE",
    "wav2vec2_base": "WAV2VEC2_BASE",
}


class TeacherEncoder(nn.Module):
    """Frozen HuBERT / wav2vec2 teacher returning intermediate layer features.

    Args:
        name: One of ``SUPPORTED_TEACHERS``.
        layers: 0-indexed transformer layers whose outputs to use as
            distillation targets. DistilHuBERT predicts HuBERT layers 4/8/12
            (1-indexed), i.e. indices (3, 7, 11). Default (3, 7, 11).
        downsample: If True (default), 2x mean-pool the time axis so the
            teacher output rate matches the student's 4x-subsampled rate.
    """

    def __init__(self, name="hubert_base", layers=(3, 7, 11), downsample=True, pretrained=True):
        super().__init__()
        if name not in SUPPORTED_TEACHERS:
            raise ValueError(f"Unknown teacher '{name}'. Choose from: {SUPPORTED_TEACHERS}")
        self.name = name
        self.layers = tuple(int(l) for l in layers)
        self.downsample = downsample
        self.output_dim = TEACHER_DIMS[name]
        self.pretrained = bool(pretrained)

        pipeline = getattr(torchaudio.pipelines, _PIPELINES[name])
        self.sample_rate = pipeline.sample_rate
        model = pipeline.get_model()
        if not self.pretrained:
            # Random-teacher ablation: identical architecture, weights re-drawn
            # from each module's default init so the teacher carries NONE of its
            # pretraining. Distilling into this isolates how much of KD's benefit
            # comes from the teacher's pretraining vs the distillation mechanism.
            n_reset = 0
            for m in model.modules():
                if m is not model and hasattr(m, "reset_parameters"):
                    m.reset_parameters()
                    n_reset += 1
            print(f"[teacher] RANDOM-INIT {name}: re-initialized {n_reset} submodules (no pretrained weights)")
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        self.model = model

    def train(self, mode: bool = True):
        # The teacher is permanently frozen; never switch it into train mode
        # (would toggle dropout / running stats). Keep it in eval regardless.
        return super().train(False)

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, wav_lengths: torch.Tensor):
        """Extract teacher features for the requested layers.

        Args:
            waveforms: [B, T_wav] float waveforms at ``self.sample_rate``,
                in [-1, 1] (no per-utterance normalization — Base checkpoints
                are trained on un-normalized input).
            wav_lengths: [B] valid sample counts before padding.

        Returns:
            feats: list of [B, T_feat, output_dim], one per requested layer.
            feat_lengths: [B] valid frame counts (shared across layers).
        """
        all_layers, out_lengths = self.model.extract_features(waveforms, lengths=wav_lengths)
        n_available = len(all_layers)
        for l in self.layers:
            if l < 0 or l >= n_available:
                raise ValueError(
                    f"Requested teacher layer {l} but model exposes {n_available} layers (0..{n_available - 1})."
                )
        feats = [all_layers[l] for l in self.layers]  # each [B, T_feat, D]

        if out_lengths is None:
            # Fallback: Base output rate is sample_rate/320.
            out_lengths = (wav_lengths // 320).clamp(min=1, max=feats[0].size(1))
        feat_lengths = out_lengths.clone()

        if self.downsample:
            aligned = []
            T = feats[0].size(1)
            keep = T - (T % 2)
            for f in feats:
                f = f[:, :keep, :]
                f = f.view(f.size(0), keep // 2, 2, f.size(2)).mean(dim=2)
                aligned.append(f)
            feats = aligned
            feat_lengths = (feat_lengths // 2).clamp(min=1, max=feats[0].size(1))

        feat_lengths = feat_lengths.clamp(max=feats[0].size(1))
        return feats, feat_lengths


def build_teacher(name="hubert_base", layers=(3, 7, 11), downsample=True, pretrained=True) -> TeacherEncoder:
    return TeacherEncoder(name=name, layers=layers, downsample=downsample, pretrained=pretrained)
