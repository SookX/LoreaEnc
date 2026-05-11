import math
import os
import socket
import time

import torch
import torch.distributed as dist


TRAIN_SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]
DEV_SPLIT = "dev-other"


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
        return 0, 0, 1, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return rank, local_rank, world_size, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def print0(rank, *args, **kwargs):
    if is_main_process(rank):
        print(*args, **kwargs, flush=True)


def debug_print(enabled, rank, message):
    if enabled:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        host = socket.gethostname()
        print(f"[{now}] [host {host}] [rank {rank}] {message}", flush=True)


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value, device):
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


def strip_state_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def unwrap_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def save_checkpoint(path, model, optimizer, scheduler, epoch, extra=None):
    os.makedirs(path, exist_ok=True)
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
    }
    if extra:
        state.update(extra)
    torch.save(state, os.path.join(path, "checkpoint.pt"))


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu", strict=True):
    ckpt_path = os.path.join(path, "checkpoint.pt") if os.path.isdir(path) else path
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(strip_state_prefixes(state["model"]), strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


def build_extended_noam_scheduler(optimizer, steps_per_epoch, warmup_epochs, peak_epochs, decay_rate):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    peak_steps = max(0, peak_epochs * steps_per_epoch)

    def lr_lambda(step):
        step = max(1, step)
        if step < warmup_steps:
            return step / warmup_steps
        if step < warmup_steps + peak_steps:
            return 1.0
        decay_step = step - peak_steps
        return (warmup_steps / max(decay_step, 1)) ** decay_rate

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def conv_out_length(lengths, kernel_size, stride, left_pad=0, right_pad=0, dilation=1):
    return ((lengths + left_pad + right_pad - dilation * (kernel_size - 1) - 1) // stride + 1).clamp(min=1)


def align_time(logits, targets, lengths=None):
    t = min(logits.size(1), targets.size(1))
    logits = logits[:, :t]
    targets = targets[:, :t]
    if lengths is not None:
        lengths = lengths.clamp(max=t)
    return logits, targets, lengths

