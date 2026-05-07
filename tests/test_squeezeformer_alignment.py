import os
import re
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from SqueezeFormer import Squeezeformer, get_config
from SqueezeFormer.attention import MultiHeadedSelfAttentionModule
from SqueezeFormer.convolution import ConvModule, DepthwiseConv2dSubsampling, TimeReductionLayer
from SqueezeFormer.encoder import SqueezeformerBlock
from SqueezeFormer.modules import FeedForwardModule, ResidualConnectionModule, ScaleBias
from SqueezeFormer.train import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    NUM_PEAK_EPOCHS,
    NUM_WARMUP_EPOCHS,
    WEIGHT_DECAY,
    paper_specaugment_time_masks,
)


OFFICIAL = Path(os.environ.get("OFFICIAL_SQUEEZEFORMER_PATH", ROOT / ".official_squeezeformer"))


def _official_config(variant):
    path = OFFICIAL / "examples" / "squeezeformer" / "configs" / f"squeezeformer-{variant.upper()}.yml"
    if not path.exists():
        raise unittest.SkipTest(f"official config not found: {path}")

    data = {}
    last_key = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":"):
            last_key = line[:-1]
            continue
        if line.startswith("-") and last_key in {"encoder_time_reduce_idx", "encoder_time_recover_idx"}:
            data[last_key] = int(line.split("-", 1)[1].strip())
            continue
        match = re.match(r"([A-Za-z0-9_]+):\s*(.+)", line)
        if match:
            key, value = match.groups()
            last_key = key
            value = value.split("#", 1)[0].strip()
            if value.lower() in {"true", "false"}:
                data[key] = value.lower() == "true"
            elif re.fullmatch(r"-?\d+", value):
                data[key] = int(value)
            elif re.fullmatch(r"-?\d+(\.\d*)?", value):
                data[key] = float(value)
            else:
                data[key] = value
    return data


class SqueezeFormerAlignmentTest(unittest.TestCase):
    def test_variant_configs_match_official_tensorflow_yamls(self):
        for variant in ("xs", "s", "sm", "m", "ml", "l"):
            official = _official_config(variant)
            local = get_config(variant)
            with self.subTest(variant=variant):
                self.assertEqual(local.encoder_dim, official["encoder_dmodel"])
                self.assertEqual(local.num_encoder_layers, official["encoder_num_blocks"])
                self.assertEqual(local.num_attention_heads, official["encoder_num_heads"])
                self.assertEqual(local.encoder_dim // local.num_attention_heads, official["encoder_head_size"])
                self.assertEqual(local.conv_kernel_size, official["encoder_kernel_size"])
                self.assertEqual(local.reduce_layer_index, official["encoder_time_reduce_idx"])
                self.assertEqual(local.recover_layer_index, official["encoder_time_recover_idx"])
                self.assertFalse(official["encoder_conv_use_glu"])
                self.assertTrue(official["encoder_ds_subsample"])
                self.assertTrue(official["encoder_adaptive_scale"])
                self.assertEqual(float(official["encoder_fc_factor"]), 1.0)
                self.assertFalse(local.half_step_residual)

    def test_block_order_matches_official_m_s_c_s_wrappers(self):
        block = SqueezeformerBlock(encoder_dim=144, num_attention_heads=4, adaptive_scale=True)
        modules = list(block.sequential)
        expected = [
            ScaleBias,
            ResidualConnectionModule,
            nn.LayerNorm,
            ScaleBias,
            ResidualConnectionModule,
            nn.LayerNorm,
            ScaleBias,
            ResidualConnectionModule,
            nn.LayerNorm,
            ScaleBias,
            ResidualConnectionModule,
            nn.LayerNorm,
        ]
        self.assertEqual([type(m) for m in modules], expected)
        self.assertIsInstance(modules[1].module, MultiHeadedSelfAttentionModule)
        self.assertIsInstance(modules[4].module, FeedForwardModule)
        self.assertIsInstance(modules[7].module, ConvModule)
        self.assertIsInstance(modules[10].module, FeedForwardModule)
        self.assertEqual(modules[4].module_factor, 1.0)
        self.assertEqual(modules[10].module_factor, 1.0)

    def test_subsampling_and_time_reduction_match_official_switches(self):
        encoder = Squeezeformer(num_classes=128, **get_config("xs").__dict__).encoder
        self.assertIsInstance(encoder.conv_subsample, DepthwiseConv2dSubsampling)
        self.assertIsInstance(encoder.time_reduction_layer, TimeReductionLayer)
        self.assertEqual(encoder.time_reduction_layer.kernel_size, 5)
        self.assertEqual(encoder.time_reduction_layer.stride, 2)

    def test_training_recipe_matches_official_configs_and_paper_appendix(self):
        self.assertEqual((ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON), (0.9, 0.98, 1e-9))
        self.assertEqual(WEIGHT_DECAY, 5e-4)
        self.assertEqual(NUM_WARMUP_EPOCHS, 20)
        self.assertEqual(NUM_PEAK_EPOCHS, 160)
        expected_masks = {"xs": 5, "s": 5, "sm": 5, "m": 7, "ml": 10, "l": 10}
        self.assertEqual({k: paper_specaugment_time_masks(k) for k in expected_masks}, expected_masks)

    def test_forward_lengths_and_parameter_scales(self):
        expected_millions = {"xs": 9.0, "s": 18.6, "sm": 28.2, "m": 55.6, "ml": 125.1, "l": 236.3}
        for variant, expected in expected_millions.items():
            cfg = get_config(variant)
            model = Squeezeformer(num_classes=128, **cfg.__dict__)
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            with self.subTest(variant=variant):
                self.assertAlmostEqual(params_m, expected, delta=0.2)

        model = Squeezeformer(num_classes=128, **get_config("xs").__dict__).eval()
        inputs = torch.randn(2, 400, 80)
        lengths = torch.tensor([400, 200])
        with torch.no_grad():
            outputs, output_lengths = model(inputs, lengths)
        self.assertEqual(outputs.shape[:2], (2, 100))
        self.assertEqual(output_lengths.tolist(), [100, 50])


if __name__ == "__main__":
    unittest.main()
