import importlib.util
import os
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Re-export shared utilities from wav2vec2.0/utils.py
# (the directory name contains a dot so it can't be a regular Python package)
# ---------------------------------------------------------------------------
_wav2vec2_utils_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "wav2vec2.0", "utils.py"
)
_spec = importlib.util.spec_from_file_location("wav2vec2_utils", _wav2vec2_utils_path)
_wav2vec2_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wav2vec2_utils)

compute_span_mask        = _wav2vec2_utils.compute_span_mask
sample_negative_indices  = _wav2vec2_utils.sample_negative_indices


# ---------------------------------------------------------------------------
# Spec2Vec config
# ---------------------------------------------------------------------------
@dataclass
class Spec2VecConfig:

    ### MEL SPECTROGRAM CONFIG ###
    n_fft: int        = 400
    hop_length: int   = 160       # 10 ms at 16 kHz -> 100 frames/sec
    n_mels: int       = 80
    sampling_rate: int = 16000

    ### TRANSFORMER CONFIG ###
    num_transformer_layers: int  = 12
    num_attention_heads: int     = 12
    embedding_dimension: int     = 768
    mlp_ratio: int               = 4
    mlp_dropout_p: float         = 0.0
    attention_dropout_p: float   = 0.0
    transformer_encoder_dropout: float = 0.0
    layer_norm_eps: float        = 1e-5
    initializer_range: float     = 0.02

    ### MASKING CONFIG ###
    masking_probability: float   = 0.065
    masking_span_length: int     = 10
    minimum_spans: int           = 2

    ### CONTRASTIVE LOSS CONFIG ###
    contrastive_logits_temperature: float = 0.1
    diversity_loss_weight: float          = 0.1
    num_negatives: int                    = 100

    ### QUANTIZER CONFIG ###
    num_codevector_groups: int      = 2
    num_codevectors_per_group: int  = 320
    codevector_dim: int             = 256
    pre_quantizer_dropout: float    = 0.0

    def to_dict(self):
        return asdict(self)
