import argparse
import contextlib
import json
import math
import os
import random
import shutil
import socket
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from SqueezeFormer.train import build_tokenizer
from CausalSpecUnit.common import (
    DEV_SPLIT,
    TRAIN_SPLITS,
    barrier,
    build_extended_noam_scheduler,
    build_lpft_scheduler,
    cleanup_distributed,
    is_main_process,
    print0,
    save_checkpoint,
    setup_distributed,
)
from CausalSpecUnit.data import BatchSpecAugment, CTCSpecDataset, collate_ctc, collate_eval
from CausalSpecUnit.model import (
    CausalSpecUnitCTC,
    MODEL_VARIANTS,
    is_melhubert_transformer_variant,
)


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def set_reproducibility_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    target = model.module if hasattr(model, "module") else model
    total = sum(p.numel() for p in target.parameters())
    trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in target.encoder.parameters())
    return {"total": total, "trainable": trainable, "encoder": encoder}


def current_lrs(optimizer):
    lrs = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{idx}")
        lrs[name] = group["lr"]
    return lrs


def current_group_grad_norms(optimizer):
    norms = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{idx}")
        total_sq = 0.0
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total_sq += float(grad.pow(2).sum().item())
        norms[name] = math.sqrt(total_sq)
    return norms


def unwrap_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def clear_encoder_grads(model):
    for param in unwrap_model(model).encoder.parameters():
        param.grad = None


def encoder_layer_depth(model, name):
    num_layers = int(getattr(model.encoder, "num_layers", 0))
    for marker in ("encoder.layers.", "model.encoder.layers."):
        if marker in name:
            rest = name.split(marker, 1)[1]
            try:
                return int(rest.split(".", 1)[0]) + 1
            except (IndexError, ValueError):
                return num_layers
    if "encoder.time_reduction_layer." in name or "model.encoder.time_reduction_layer." in name:
        return min(num_layers, int(getattr(model.encoder, "reduce_layer_index", 0)) + 1)
    if "encoder.time_recover_layer." in name or "model.encoder.time_recover_layer." in name:
        return min(num_layers, int(getattr(model.encoder, "recover_layer_index", num_layers - 1)) + 1)
    return 0


def make_adamw_param_groups(model, encoder_lr, head_lr, encoder_layer_lr_decay=1.0, split_no_decay=True):
    if encoder_layer_lr_decay <= 0.0:
        raise ValueError("--encoder-layer-lr-decay must be positive.")
    no_decay_terms = ("bias", "norm", "layer_norm", "ln")
    groups = {}
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    num_layers = int(getattr(model.encoder, "num_layers", 0))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = id(param) in encoder_param_ids
        if is_encoder:
            depth = encoder_layer_depth(model, name)
            lr_scale = float(encoder_layer_lr_decay) ** max(num_layers - depth, 0)
            lr = encoder_lr * lr_scale
            prefix = f"encoder_l{depth:02d}"
        else:
            lr = head_lr
            prefix = "head"
        is_no_decay = split_no_decay and (param.ndim <= 1 or any(term in name.lower() for term in no_decay_terms))
        suffix = "no_decay" if is_no_decay else "decay"
        weight_decay = 0.0 if is_no_decay else None
        key = (prefix, suffix, lr, weight_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "weight_decay": weight_decay, "name": f"{prefix}_{suffix}"}
        groups[key]["params"].append(param)
    return [group for group in groups.values() if group["params"]]


def load_ssl_mask_embedding(checkpoint_path, expected_dim=80, map_location="cpu"):
    if not checkpoint_path:
        return None
    state = torch.load(checkpoint_path, map_location=map_location)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    for key, value in model_state.items():
        key = key.removeprefix("module.").removeprefix("_orig_mod.")
        if key == "mask_emb":
            value = value.detach().float()
            if value.numel() != expected_dim:
                raise ValueError(
                    f"SSL mask_emb has {value.numel()} values, expected {expected_dim}. "
                    "Check that the checkpoint and mel feature config match."
                )
            return value
    return None


def reduce_train_average(total_loss, n_batches, device):
    stats = torch.tensor([float(total_loss), float(n_batches)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item() / max(stats[1].item(), 1.0))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--cmvn-path", type=str, default="outputs/causal_specunit/targets/cmvn.pt")
    p.add_argument("--ssl-checkpoint", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="outputs/causal_specunit/ctc")
    p.add_argument("--tokenizer-path", type=str, default="dataset/bpe128.model")
    p.add_argument("--train-splits", nargs="+", default=TRAIN_SPLITS,
                   help="Training splits to use. Defaults to full LibriSpeech train-960.")
    p.add_argument("--train-subset-hours", type=float, default=None,
                   help="Use a reproducible random subset with approximately this many audio hours.")
    p.add_argument("--train-subset-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for model initialization and DataLoader/DistributedSampler shuffling.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataloader-timeout", type=int, default=120)
    p.add_argument("--variant", type=str, default="xs", choices=list(MODEL_VARIANTS))
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--encoder-lr", type=float, default=None,
                   help="Optional peak LR for encoder parameters, useful for SSL fine-tuning.")
    p.add_argument("--head-lr", type=float, default=None,
                   help="Optional peak LR for non-encoder parameters, including the CTC head.")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--no-decay-norm-and-bias", action="store_true",
                   help="Exclude normalization weights and biases from AdamW weight decay.")
    p.add_argument("--encoder-layer-lr-decay", type=float, default=1.0,
                   help="Layer-wise encoder LR decay. 1.0 uses one encoder LR; values below 1.0 adapt upper layers more than lower layers.")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--freeze-encoder-epochs", type=int, default=0,
                   help="Keep the encoder frozen for the first N epochs. Mainly useful when fine-tuning SSL encoders with a fresh CTC head.")
    p.add_argument("--encoder-rewarmup-epochs", type=int, default=0,
                   help="After --freeze-encoder-epochs, linearly re-warmup the encoder LR over this many epochs. "
                        "The head schedule is unaffected. Enables LP-FT (linear probe then fine-tune): the head settles "
                        "on SSL features before the encoder starts moving, preventing destruction of the pretrained representation.")
    p.add_argument("--inter-ctc-layers", type=int, nargs="*", default=None,
                   help="Encoder block indices (0-based) at which to attach auxiliary CTC heads (InterCTC). "
                        "For SqueezeFormer XS (16 blocks, reduce@7, recover@15), 7 is a strong default: the deepest "
                        "shared point of the upper U-net half. Injects a CTC gradient halfway down the encoder so the "
                        "lower stack gets direct task supervision instead of one diluted by the rest of the network.")
    p.add_argument("--inter-ctc-weight", type=float, default=0.3,
                   help="Total weight on InterCTC losses; final loss = (1-w)*main_ctc + w*mean(inter_ctc).")
    # ---- SSL-target anchored fine-tuning ----
    p.add_argument("--ssl-anchor-weight", type=float, default=0.0,
                   help="Weight on the auxiliary K=100/K=500 cluster prediction loss during fine-tuning. "
                        "Anchors the encoder to its SSL feature space and prevents the CTC head from "
                        "rewriting useful pretrained features. 0.0 disables (no targets loaded, no heads). "
                        "Recommended starting weight: 0.1.")
    p.add_argument("--ssl-anchor-targets-dir", type=str, default=None,
                   help="Directory containing the SSL cluster targets (targets.pt + shards). When "
                        "--ssl-anchor-weight > 0, the dataset loads these targets and the model "
                        "predicts cluster IDs at every encoder output position alongside CTC. "
                        "Items whose UID is not in the targets file are filtered.")
    p.add_argument("--ssl-anchor-load-heads", action="store_true",
                   help="Warm-start the SSL anchor heads from the SSL checkpoint's head_coarse/head_fine "
                        "weights, so the auxiliary loss is meaningful from step 1 rather than random.")
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--peak-epochs", type=int, default=160,
                   help="Number of epochs to hold the peak LR after warmup before Noam decay.")
    p.add_argument("--noam-decay-rate", type=float, default=1.0)
    p.add_argument("--eval-split", type=str, default=DEV_SPLIT)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10,
                   help="Save periodic checkpoint_epNNN snapshots every N epochs. Set 0 to disable periodic snapshots.")
    p.add_argument("--keep-checkpoints", type=int, default=5)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--progress", choices=["on", "off"], default="on")
    p.add_argument("--specaug", action="store_true",
                   help="Apply SpecAugment to training mels after CMVN and before the model.")
    p.add_argument("--specaug-time-mask-param", type=int, default=40)
    p.add_argument("--specaug-freq-mask-param", type=int, default=30)
    p.add_argument("--specaug-time-masks", type=int, default=2)
    p.add_argument("--specaug-freq-masks", type=int, default=2)
    p.add_argument("--specaug-disable-last-epochs", type=int, default=0,
                   help="Disable SpecAugment for the final N epochs for clean fine-tuning.")
    p.add_argument("--specaug-mask-source", choices=["zero", "ssl-mask"], default="zero",
                   help="Use zero masks or the SSL checkpoint's learned mask token for SpecAugment.")
    return p.parse_args()


def cleanup_epoch_checkpoints(output_dir, keep):
    if keep <= 0 or not os.path.isdir(output_dir):
        return
    ckpts = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint_ep"):
            try:
                epoch = int(name.split("_ep")[-1])
            except ValueError:
                continue
            ckpts.append((epoch, name))
    ckpts.sort()
    for _, name in ckpts[:-keep]:
        shutil.rmtree(os.path.join(output_dir, name), ignore_errors=True)


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def word_error_counts(ref_words, hyp_words):
    """Return substitutions, insertions, deletions for ref -> hyp."""
    rows, cols = len(ref_words), len(hyp_words)
    dp = [[(0, 0, 0, 0) for _ in range(cols + 1)] for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost, sub, ins, dele = dp[i - 1][0]
        dp[i][0] = (cost + 1, sub, ins, dele + 1)
    for j in range(1, cols + 1):
        cost, sub, ins, dele = dp[0][j - 1]
        dp[0][j] = (cost + 1, sub, ins + 1, dele)

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            cost, sub, ins, dele = dp[i - 1][j - 1]
            substitute = (cost + 1, sub + 1, ins, dele)
            cost, sub, ins, dele = dp[i][j - 1]
            insert = (cost + 1, sub, ins + 1, dele)
            cost, sub, ins, dele = dp[i - 1][j]
            delete = (cost + 1, sub, ins, dele + 1)
            dp[i][j] = min((substitute, insert, delete), key=lambda x: (x[0], x[1], x[2], x[3]))
    _, sub, ins, dele = dp[rows][cols]
    return sub, ins, dele


def greedy_decode(log_probs, lengths, blank_id):
    preds = log_probs.argmax(dim=-1)
    decoded = []
    for b in range(preds.size(0)):
        seq, prev = [], None
        for tok in preds[b, :lengths[b]].tolist():
            if tok != prev:
                seq.append(tok)
            prev = tok
        decoded.append([t for t in seq if t != blank_id])
    return decoded


@torch.no_grad()
def evaluate(model, loader, tokenizer, blank_id, ctc_loss, device, rank, show_progress=True):
    model.eval()
    total_word_errors = 0
    total_ref_words = 0
    total_loss = 0.0
    total_batches = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    total_hyp_words = 0
    total_char_errors = 0
    total_ref_chars = 0
    empty_hypotheses = 0
    total_utterances = 0
    example = None
    bar = tqdm(loader, desc="eval", leave=False, disable=not (show_progress and is_main_process(rank)))
    for mel, lengths, labels, label_lengths, transcripts in bar:
        mel = mel.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # Eval always uses only the main CTC head; InterCTC and SSL anchor are training-time auxiliaries.
            log_probs, output_lengths, _, _ = model(mel, lengths, return_inter=False, return_ssl=False)
            loss = ctc_loss(log_probs.transpose(0, 1), labels, output_lengths, label_lengths)
        total_loss += loss.detach().float().item()
        total_batches += 1
        hyps = [tokenizer.decode(x).lower().strip() for x in greedy_decode(log_probs, output_lengths, blank_id)]
        refs = [x.lower().strip() for x in transcripts]
        for hyp, ref in zip(hyps, refs):
            hw, rw = hyp.split(), ref.split()
            substitutions, insertions, deletions = word_error_counts(rw, hw)
            total_substitutions += substitutions
            total_insertions += insertions
            total_deletions += deletions
            total_word_errors += substitutions + insertions + deletions
            total_ref_words += max(len(rw), 1)
            total_hyp_words += len(hw)
            empty_hypotheses += int(len(hw) == 0)
            ref_chars = list(ref.replace(" ", ""))
            hyp_chars = list(hyp.replace(" ", ""))
            total_char_errors += edit_distance(hyp_chars, ref_chars)
            total_ref_chars += max(len(ref_chars), 1)
            total_utterances += 1
            if example is None:
                example = (hyp, ref)

    stats = torch.tensor(
        [
            total_word_errors,
            total_ref_words,
            total_loss,
            total_batches,
            total_substitutions,
            total_insertions,
            total_deletions,
            total_hyp_words,
            total_char_errors,
            total_ref_chars,
            empty_hypotheses,
            total_utterances,
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_word_errors = float(stats[0].item())
    total_ref_words = float(stats[1].item())
    total_loss = float(stats[2].item())
    total_batches = float(stats[3].item())
    total_substitutions = float(stats[4].item())
    total_insertions = float(stats[5].item())
    total_deletions = float(stats[6].item())
    total_hyp_words = float(stats[7].item())
    total_char_errors = float(stats[8].item())
    total_ref_chars = float(stats[9].item())
    empty_hypotheses = float(stats[10].item())
    total_utterances = float(stats[11].item())

    return {
        "loss": total_loss / max(total_batches, 1.0),
        "wer": total_word_errors / max(total_ref_words, 1.0),
        "cer": total_char_errors / max(total_ref_chars, 1.0),
        "substitution_rate": total_substitutions / max(total_ref_words, 1.0),
        "insertion_rate": total_insertions / max(total_ref_words, 1.0),
        "deletion_rate": total_deletions / max(total_ref_words, 1.0),
        "hyp_words_per_ref_word": total_hyp_words / max(total_ref_words, 1.0),
        "empty_hypothesis_rate": empty_hypotheses / max(total_utterances, 1.0),
        "eval_utterances": int(total_utterances),
        "example": example,
    }


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    set_reproducibility_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    tokenizer = build_tokenizer(args.tokenizer_path)
    blank_id = tokenizer.pad_token_id
    cmvn_path = args.cmvn_path
    if cmvn_path is not None and cmvn_path.lower() in {"", "none", "null"}:
        cmvn_path = None
    ssl_anchor_active = args.ssl_anchor_weight > 0.0
    ssl_targets_path = args.ssl_anchor_targets_dir if ssl_anchor_active else None
    train_dataset = CTCSpecDataset(
        args.data_root,
        args.train_splits,
        tokenizer,
        cmvn_path=cmvn_path,
        train_split=True,
        max_hours=args.train_subset_hours,
        subset_seed=args.train_subset_seed,
        ssl_targets_path=ssl_targets_path,
    )
    dev_dataset = CTCSpecDataset(args.data_root, [args.eval_split], tokenizer, cmvn_path=cmvn_path, train_split=False)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if world_size > 1 else None
    )
    dev_sampler = (
        DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
        if world_size > 1 else None
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    worker_kwargs = {"persistent_workers": True, "prefetch_factor": 4} if args.workers > 0 else {}
    # PyTorch's single-process DataLoaderIter requires timeout=0; the
    # --dataloader-timeout flag only applies to worker-based loaders.
    effective_timeout = args.dataloader_timeout if args.workers > 0 else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=collate_ctc,
        pin_memory=True,
        drop_last=True,
        timeout=effective_timeout,
        generator=loader_generator,
        **worker_kwargs,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=args.workers,
        collate_fn=collate_eval,
        pin_memory=True,
        timeout=effective_timeout,
        **worker_kwargs,
    )

    model = CausalSpecUnitCTC(
        vocab_size=tokenizer.vocab_size,
        variant=args.variant,
        inter_ctc_layers=args.inter_ctc_layers,
        ssl_anchor=ssl_anchor_active,
    )
    if args.ssl_checkpoint:
        missing, unexpected = model.load_ssl_encoder(
            args.ssl_checkpoint,
            map_location="cpu",
            load_ssl_heads=bool(args.ssl_anchor_load_heads and ssl_anchor_active),
        )
        print0(rank, f"Loaded SSL encoder from {args.ssl_checkpoint} | missing={len(missing)} unexpected={len(unexpected)}")
        if ssl_anchor_active and args.ssl_anchor_load_heads:
            print0(rank, "  SSL anchor heads warm-started from SSL checkpoint")
    model.to(device)
    if world_size > 1:
        # nn.TransformerEncoderLayer can produce unused-parameter gradients
        # under certain attention mask patterns, which deadlocks plain DDP's
        # all_reduce. Enable find_unused_parameters for the MelHuBERT-style
        # Transformer variant; SqueezeFormer keeps the default fast path.
        ddp_find_unused = is_melhubert_transformer_variant(args.variant)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
        )

    metrics_path = os.path.join(args.output_dir, "ctc_metrics.jsonl")
    run_info_path = os.path.join(args.output_dir, "ctc_run_info.json")
    if is_main_process(rank):
        run_info = {
            "event": "run_start",
            "argv": sys.argv,
            "args": vars(args),
            "world_size": world_size,
            "device": str(device),
            "hostname": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameter_counts": count_parameters(model),
            "train_utterances": len(train_dataset),
            "dev_utterances": len(dev_dataset),
            "effective_batch": args.batch_size * world_size * args.grad_accum_steps,
            "ssl_initialized": bool(args.ssl_checkpoint),
            "train_splits": args.train_splits,
            "train_audio_hours": getattr(train_dataset, "audio_hours", None),
        }
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, sort_keys=True)
        append_jsonl(metrics_path, run_info)

    opt_model = unwrap_model(model)
    if args.encoder_lr is not None or args.head_lr is not None:
        encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
        head_lr = args.head_lr if args.head_lr is not None else args.lr
        use_detailed_groups = args.no_decay_norm_and_bias or args.encoder_layer_lr_decay != 1.0
        if not use_detailed_groups:
            encoder_param_ids = {id(p) for p in opt_model.encoder.parameters()}
            head_params = [p for p in opt_model.parameters() if id(p) not in encoder_param_ids]
            param_groups = [
                {"params": opt_model.encoder.parameters(), "lr": encoder_lr, "name": "encoder"},
                {"params": head_params, "lr": head_lr, "name": "head"},
            ]
        else:
            param_groups = make_adamw_param_groups(
                opt_model,
                encoder_lr,
                head_lr,
                encoder_layer_lr_decay=args.encoder_layer_lr_decay,
                split_no_decay=args.no_decay_norm_and_bias,
            )
            for group in param_groups:
                if group["weight_decay"] is None:
                    group["weight_decay"] = args.weight_decay
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=args.weight_decay,
        )
        print0(
            rank,
            f"LR groups: encoder={encoder_lr:g} | head={head_lr:g} "
            f"| layer_decay={args.encoder_layer_lr_decay:g} | no_decay_norm_bias={args.no_decay_norm_and_bias}",
        )
    else:
        use_detailed_groups = args.no_decay_norm_and_bias or args.encoder_layer_lr_decay != 1.0
        if not use_detailed_groups:
            param_groups = model.parameters()
        else:
            param_groups = make_adamw_param_groups(
                opt_model,
                args.lr,
                args.lr,
                encoder_layer_lr_decay=args.encoder_layer_lr_decay,
                split_no_decay=args.no_decay_norm_and_bias,
            )
            for group in param_groups:
                if group["weight_decay"] is None:
                    group["weight_decay"] = args.weight_decay
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=args.weight_decay,
        )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    use_lpft = args.encoder_rewarmup_epochs > 0 or args.freeze_encoder_epochs > 0
    if use_lpft:
        scheduler = build_lpft_scheduler(
            optimizer,
            steps_per_epoch,
            warmup_epochs=args.warmup_epochs,
            peak_epochs=args.peak_epochs,
            decay_rate=args.noam_decay_rate,
            encoder_freeze_epochs=args.freeze_encoder_epochs,
            encoder_rewarmup_epochs=args.encoder_rewarmup_epochs,
        )
        print0(
            rank,
            f"LP-FT scheduler: encoder freeze={args.freeze_encoder_epochs}ep "
            f"+ rewarmup={args.encoder_rewarmup_epochs}ep | head schedule unchanged",
        )
    else:
        scheduler = build_extended_noam_scheduler(
            optimizer,
            steps_per_epoch,
            warmup_epochs=args.warmup_epochs,
            peak_epochs=args.peak_epochs,
            decay_rate=args.noam_decay_rate,
        )
    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    specaug_mask_value = 0.0
    specaug_mask_source = args.specaug_mask_source
    if args.specaug_mask_source == "ssl-mask":
        specaug_mask_value = load_ssl_mask_embedding(args.ssl_checkpoint, expected_dim=80, map_location="cpu")
        if specaug_mask_value is None:
            raise ValueError("--specaug-mask-source ssl-mask requires an SSL checkpoint with mask_emb")
        print0(rank, "SpecAugment mask source: SSL learned mask_emb")
    else:
        print0(rank, "SpecAugment mask source: zero")
    specaugment = BatchSpecAugment(
        time_mask_param=args.specaug_time_mask_param,
        freq_mask_param=args.specaug_freq_mask_param,
        num_time_masks=args.specaug_time_masks,
        num_freq_masks=args.specaug_freq_masks,
        mask_value=specaug_mask_value,
    ).to(device)
    best_wer = float("inf")
    optimizer_steps = 0
    run_start = time.time()
    hours_note = ""
    if getattr(train_dataset, "audio_hours", None) is not None:
        hours_note = f" train_hours={train_dataset.audio_hours:.2f}"
    print0(
        rank,
        f"CausalSpecUnit CTC | train={len(train_dataset)} dev={len(dev_dataset)} "
        f"world={world_size} effective_batch={args.batch_size * world_size * args.grad_accum_steps}"
        f"{hours_note} | warmup={args.warmup_epochs} hold={args.peak_epochs} decay={args.noam_decay_rate:g} "
        f"| specaug={args.specaug} disable_last={args.specaug_disable_last_epochs}",
    )

    try:
        for epoch in range(1, args.epochs + 1):
            specaug_enabled = bool(
                args.specaug
                and (
                    args.specaug_disable_last_epochs <= 0
                    or epoch <= args.epochs - args.specaug_disable_last_epochs
                )
            )
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            encoder_should_train = epoch > args.freeze_encoder_epochs
            if args.freeze_encoder_epochs > 0 and epoch in (1, args.freeze_encoder_epochs + 1):
                state = "trainable" if encoder_should_train else "update-frozen"
                print0(rank, f"[ctc] epoch={epoch:03d} encoder {state}")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_main_loss = 0.0
            total_inter_loss = 0.0
            total_ssl_anchor_loss = 0.0
            n_batches = 0
            n_ssl_anchor_batches = 0
            grad_steps = 0
            clipped_steps = 0
            grad_norm_sum = 0.0
            grad_norm_max = 0.0
            group_grad_norm_sums = {}
            group_grad_norm_max = {}
            show = args.progress == "on" and is_main_process(rank)
            bar = tqdm(train_loader, desc=f"CTC {epoch:03d}", leave=False, disable=not show)
            effective_batches = len(train_loader)
            if args.max_train_batches is not None:
                effective_batches = min(effective_batches, args.max_train_batches)
            for step, batch in enumerate(bar, start=1):
                if step > effective_batches:
                    break
                accum_index = (step - 1) % max(1, args.grad_accum_steps)
                remaining = effective_batches - step + 1
                actual_accum_steps = min(max(1, args.grad_accum_steps), accum_index + remaining)
                sync_step = accum_index + 1 >= actual_accum_steps
                mel, lengths, labels, label_lengths, z100, z500, _ssl_target_lengths = batch
                mel = mel.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)
                if z100 is not None:
                    z100 = z100.to(device, non_blocking=True)
                    z500 = z500.to(device, non_blocking=True)
                if specaug_enabled:
                    mel = specaugment(mel, lengths)
                sync_context = (
                    model.no_sync
                    if isinstance(model, DDP) and not sync_step
                    else contextlib.nullcontext
                )
                with sync_context():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        log_probs, output_lengths, inter_outputs, ssl_outputs = model(
                            mel, lengths,
                            return_inter=bool(args.inter_ctc_layers),
                            return_ssl=ssl_anchor_active,
                        )
                        main_loss = ctc_loss(log_probs.transpose(0, 1), labels, output_lengths, label_lengths)
                        if inter_outputs:
                            inter_losses = []
                            for _idx, (inter_log_probs, inter_lengths) in inter_outputs.items():
                                inter_losses.append(
                                    ctc_loss(
                                        inter_log_probs.transpose(0, 1),
                                        labels,
                                        inter_lengths,
                                        label_lengths,
                                    )
                                )
                            inter_loss = torch.stack(inter_losses).mean()
                            w = args.inter_ctc_weight
                            combined = (1.0 - w) * main_loss + w * inter_loss
                        else:
                            inter_loss = None
                            combined = main_loss

                        # SSL-target anchored fine-tuning: predict K=100/K=500
                        # cluster IDs at every encoder output position with the
                        # auxiliary heads, alongside CTC. Targets are padded
                        # with -100 so CE ignores invalid positions naturally.
                        ssl_anchor_loss = None
                        if ssl_anchor_active and ssl_outputs is not None and z100 is not None:
                            ssl_coarse_logits, ssl_fine_logits = ssl_outputs
                            t = min(ssl_coarse_logits.size(1), z100.size(1))
                            sc = ssl_coarse_logits[:, :t]
                            sf = ssl_fine_logits[:, :t]
                            z100_a = z100[:, :t]
                            z500_a = z500[:, :t]
                            ce_coarse = nn.functional.cross_entropy(
                                sc.reshape(-1, sc.size(-1)),
                                z100_a.reshape(-1),
                                ignore_index=-100,
                            )
                            ce_fine = nn.functional.cross_entropy(
                                sf.reshape(-1, sf.size(-1)),
                                z500_a.reshape(-1),
                                ignore_index=-100,
                            )
                            ssl_anchor_loss = ce_coarse + ce_fine
                            combined = combined + args.ssl_anchor_weight * ssl_anchor_loss

                        loss = combined / actual_accum_steps
                    loss.backward()
                grad_norm_value = None
                group_grad_norms = None
                if sync_step:
                    if not encoder_should_train:
                        clear_encoder_grads(model)
                    group_grad_norms = current_group_grad_norms(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    grad_norm_value = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    grad_steps += 1
                    grad_norm_sum += grad_norm_value
                    grad_norm_max = max(grad_norm_max, grad_norm_value)
                    clipped_steps += int(grad_norm_value > args.max_grad_norm)
                    for name, value in group_grad_norms.items():
                        group_grad_norm_sums[name] = group_grad_norm_sums.get(name, 0.0) + value
                        group_grad_norm_max[name] = max(group_grad_norm_max.get(name, 0.0), value)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                loss_val = loss.detach().float().item() * actual_accum_steps
                main_loss_val = main_loss.detach().float().item()
                inter_loss_val = inter_loss.detach().float().item() if inter_loss is not None else None
                ssl_anchor_val = ssl_anchor_loss.detach().float().item() if ssl_anchor_loss is not None else None
                total_loss += loss_val
                total_main_loss += main_loss_val
                if inter_loss_val is not None:
                    total_inter_loss += inter_loss_val
                if ssl_anchor_val is not None:
                    total_ssl_anchor_loss += ssl_anchor_val
                    n_ssl_anchor_batches += 1
                n_batches += 1
                if show:
                    postfix = dict(
                        loss=f"{loss_val:.3f}",
                        avg=f"{total_loss/max(n_batches,1):.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.1e}",
                    )
                    if inter_loss_val is not None:
                        postfix["main"] = f"{main_loss_val:.3f}"
                        postfix["inter"] = f"{inter_loss_val:.3f}"
                    if ssl_anchor_val is not None:
                        postfix["anchor"] = f"{ssl_anchor_val:.3f}"
                    bar.set_postfix(**postfix, refresh=False)
                if args.log_every > 0 and is_main_process(rank) and step % args.log_every == 0:
                    append_jsonl(metrics_path, {
                        "event": "train_step",
                        "epoch": epoch,
                        "batch": step,
                        "batches_per_epoch": effective_batches,
                        "optimizer_step": optimizer_steps,
                        "train_loss": loss_val,
                        "train_loss_avg": total_loss / max(n_batches, 1),
                        "lr": scheduler.get_last_lr()[0],
                        "lrs": current_lrs(optimizer),
                        "grad_norm": grad_norm_value,
                        "group_grad_norms": group_grad_norms,
                        "specaug_enabled": specaug_enabled,
                        "specaug_mask_source": specaug_mask_source,
                        "encoder_trainable": encoder_should_train,
                        "elapsed_hours": (time.time() - run_start) / 3600,
                    })

            avg_loss = reduce_train_average(total_loss, n_batches, device)
            avg_main_loss = reduce_train_average(total_main_loss, n_batches, device)
            avg_inter_loss = (
                reduce_train_average(total_inter_loss, n_batches, device)
                if args.inter_ctc_layers else None
            )
            avg_ssl_anchor_loss = (
                reduce_train_average(total_ssl_anchor_loss, max(n_ssl_anchor_batches, 1), device)
                if ssl_anchor_active else None
            )
            grad_norm_avg = grad_norm_sum / max(grad_steps, 1)
            group_grad_norm_avg = {
                name: value / max(grad_steps, 1)
                for name, value in group_grad_norm_sums.items()
            }
            clip_fraction = clipped_steps / max(grad_steps, 1)
            should_eval = args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs)
            if should_eval and dev_sampler is not None:
                dev_sampler.set_epoch(epoch)
            if should_eval:
                metrics = evaluate(model, dev_loader, tokenizer, blank_id, ctc_loss, device, rank, show_progress=show)
                if is_main_process(rank):
                    best_wer = min(best_wer, metrics["wer"])
                    hyp, ref = metrics["example"] or ("", "")
                    tqdm.write(
                        f"[ctc] epoch={epoch:03d} train_loss={avg_loss:.4f} dev_loss={metrics['loss']:.4f} "
                        f"wer={metrics['wer']:.2%} cer={metrics['cer']:.2%} best={best_wer:.2%} "
                        f"del={metrics['deletion_rate']:.2%} ins={metrics['insertion_rate']:.2%} "
                        f"clip={clip_fraction:.1%}\nREF: {ref}\nHYP: {hyp}"
                    )
                    append_jsonl(metrics_path, {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "optimizer_step": optimizer_steps,
                        "train_loss": avg_loss,
                        "train_main_loss": avg_main_loss,
                        "train_inter_loss": avg_inter_loss,
                        "train_ssl_anchor_loss": avg_ssl_anchor_loss,
                        "inter_ctc_layers": list(args.inter_ctc_layers) if args.inter_ctc_layers else None,
                        "inter_ctc_weight": args.inter_ctc_weight if args.inter_ctc_layers else None,
                        "ssl_anchor_weight": args.ssl_anchor_weight if ssl_anchor_active else None,
                        "dev_loss": metrics["loss"],
                        "wer": metrics["wer"],
                        "cer": metrics["cer"],
                        "substitution_rate": metrics["substitution_rate"],
                        "insertion_rate": metrics["insertion_rate"],
                        "deletion_rate": metrics["deletion_rate"],
                        "hyp_words_per_ref_word": metrics["hyp_words_per_ref_word"],
                        "empty_hypothesis_rate": metrics["empty_hypothesis_rate"],
                        "eval_utterances": metrics["eval_utterances"],
                        "best_wer": best_wer,
                        "lr": scheduler.get_last_lr()[0],
                        "lrs": current_lrs(optimizer),
                        "specaug_enabled": specaug_enabled,
                        "specaug": {
                            "time_mask_param": args.specaug_time_mask_param,
                            "freq_mask_param": args.specaug_freq_mask_param,
                            "time_masks": args.specaug_time_masks,
                            "freq_masks": args.specaug_freq_masks,
                            "disable_last_epochs": args.specaug_disable_last_epochs,
                            "mask_source": specaug_mask_source,
                        },
                        "encoder_trainable": encoder_should_train,
                        "freeze_encoder_epochs": args.freeze_encoder_epochs,
                        "grad_norm_avg": grad_norm_avg,
                        "grad_norm_max": grad_norm_max,
                        "clip_fraction": clip_fraction,
                        "group_grad_norm_avg": group_grad_norm_avg,
                        "group_grad_norm_max": group_grad_norm_max,
                        "elapsed_hours": (time.time() - run_start) / 3600,
                        "example_ref": ref,
                        "example_hyp": hyp,
                    })
                    if metrics["wer"] <= best_wer:
                        save_checkpoint(
                            os.path.join(args.output_dir, "checkpoint_best"),
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            extra={"best_wer": best_wer, "optimizer_steps": optimizer_steps},
                        )
            else:
                print0(rank, f"[ctc] epoch={epoch:03d} train_loss={avg_loss:.4f}")
                if is_main_process(rank):
                    append_jsonl(metrics_path, {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "optimizer_step": optimizer_steps,
                        "train_loss": avg_loss,
                        "train_main_loss": avg_main_loss,
                        "train_inter_loss": avg_inter_loss,
                        "train_ssl_anchor_loss": avg_ssl_anchor_loss,
                        "inter_ctc_layers": list(args.inter_ctc_layers) if args.inter_ctc_layers else None,
                        "inter_ctc_weight": args.inter_ctc_weight if args.inter_ctc_layers else None,
                        "ssl_anchor_weight": args.ssl_anchor_weight if ssl_anchor_active else None,
                        "dev_loss": None,
                        "wer": None,
                        "best_wer": best_wer if math.isfinite(best_wer) else None,
                        "lr": scheduler.get_last_lr()[0],
                        "lrs": current_lrs(optimizer),
                        "specaug_enabled": specaug_enabled,
                        "grad_norm_avg": grad_norm_avg,
                        "grad_norm_max": grad_norm_max,
                        "clip_fraction": clip_fraction,
                        "group_grad_norm_avg": group_grad_norm_avg,
                        "group_grad_norm_max": group_grad_norm_max,
                        "elapsed_hours": (time.time() - run_start) / 3600,
                    })
            barrier()
            if args.save_every > 0 and epoch % args.save_every == 0:
                if is_main_process(rank):
                    save_checkpoint(
                        os.path.join(args.output_dir, f"checkpoint_ep{epoch:03d}"),
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        extra={"best_wer": best_wer, "optimizer_steps": optimizer_steps},
                    )
                    cleanup_epoch_checkpoints(args.output_dir, args.keep_checkpoints)
                barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
