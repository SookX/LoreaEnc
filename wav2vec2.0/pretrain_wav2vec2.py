"""
Wav2Vec2 Pre-training Script
Run with:  python pretrain_wav2vec2.py
           (self-launches via accelerate launch; works for 1 or 8 GPUs)
"""

import os
import shutil
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import get_scheduler, set_seed

from dataset import LibriSpeechDataset, Wav2Vec2CollateFunctionForPreTraining
from model import Wav2Vec2ForPreTraining
from utils import Wav2Vec2Config

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these instead of passing command-line flags
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME     = "Pretraining_Wav2Vec2Base_100h"
WORKING_DIRECTORY   = "work_dir"
PATH_TO_DATA_ROOT   = "./dataset"

# Dataset
TRAIN_SPLITS            = ["train-clean-100"]
TEST_SPLITS             = ["dev-clean"]
MINIMUM_AUDIO_DURATION  = 2.0       # seconds
MAXIMUM_AUDIO_DURATION  = 5.0       # seconds — 15s → seq_len ~750; 5s → ~250, fits a 3080
SAMPLING_RATE           = 16000
NUM_WORKERS             = 4         # Windows worker startup is slow; 4 is enough

# Masking
MASKING_PROBABILITY = 0.065
MASKING_SPAN_LENGTH = 10
MINIMUM_SPANS       = 2
NUM_NEGATIVES       = 100

# Feature encoder convolutions
CONV_DIM    = (512, 512, 512, 512, 512, 512, 512)
CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)
CONV_BIAS   = True

# Feature projection
FEATURE_PROJ_DROPOUT_P = 0.0

# Convolutional positional embeddings
CONV_POS_EMB_GROUPS       = 16
CONV_POS_EMB_KERNEL_SIZE  = 128
CONV_POS_EMB_DROP_P       = 0.0

# Transformer
NUM_TRANSFORMER_LAYERS        = 8
NUM_ATTENTION_HEADS           = 8
EMBEDDING_DIMENSION           = 640
MLP_RATIO                     = 4
MLP_DROPOUT_P                 = 0.0
ATTENTION_DROPOUT_P           = 0.0
TRANSFORMER_ENCODER_DROPOUT_P = 0.0
LAYER_DROPOUT                 = 0.0    # disabled on multi-GPU automatically
INITIALIZER_RANGE             = 0.02

# Vector quantizer
NUM_CODEVECTOR_GROUPS       = 2
NUM_CODEVECTORS_PER_GROUP   = 320
CODEVECTOR_DIM              = 256
PRE_QUANTIZER_DROPOUT_P     = 0.0

# Gumbel softmax
MAX_GUMBEL_TEMPERATURE  = 2.0
MIN_GUMBEL_TEMPERATURE  = 0.5
GUMBEL_TEMPERATURE_DECAY = 0.999995

# Loss
CONTRASTIVE_LOGITS_TEMPERATURE = 0.1
DIVERSITY_LOSS_WEIGHT          = 0.1

# Optimiser
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 0.01
ADAM_BETA1       = 0.9
ADAM_BETA2       = 0.98
ADAM_EPSILON     = 1e-6
BIAS_WEIGHT_DECAY = False   # exclude bias params from weight decay
NORM_WEIGHT_DECAY = False   # exclude norm params from weight decay

# Training schedule
PER_GPU_BATCH_SIZE        = 16      # minibatch = 16/4 = 4 samples; fits a 3080 with 5s audio
GRADIENT_ACCUMULATION     = 4
NUM_TRAINING_STEPS        = 200_000
NUM_WARMUP_STEPS          = 32_000
LR_SCHEDULER_TYPE         = "polynomial"

# Logging / checkpointing
LOGGING_STEPS        = 2000
EVALUATION_INTERVAL  = 500
CHECKPOINT_INTERVAL  = 1000
NUM_KEEP_CHECKPOINTS = 5
LOG_WANDB            = True

# Reproducibility
SEED = 0

# Resume — set to e.g. "checkpoint_5000" to resume, or None to start fresh
RESUME_FROM_CHECKPOINT = None

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def multiply_gradients(params, constant):
    """
    wav2vec2 uses summed (not averaged) loss, so the number of masked tokens
    differs across GPUs. We gather the total and scale each GPU's gradients so
    the effective update is normalised by total masked tokens, not batch size.
    """
    for param in params:
        if param.grad is not None:
            param.grad.data.mul_(constant)


def compute_gradient_norms(params, scale=1.0):
    """
    Compute gradient norm for training health monitoring (target ~0.5–2).
    All operations stay on GPU; only one .item() sync at the very end.
    """
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2) / scale
    return total_norm.item()


def compute_batch_duration(attention_mask, sampling_rate):
    """Hours of audio in the batch (attention_mask is at the raw-audio level)."""
    return torch.sum(attention_mask.sum(dim=-1) / sampling_rate) / 3600


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():

    ### Seed ###
    if SEED is not None:
        set_seed(SEED)

    ### Accelerate ###
    # gradient_accumulation_steps tells accelerate to use no_sync() on DDP
    # intermediate steps — critical for 8-GPU efficiency.
    path_to_experiment = os.path.join(WORKING_DIRECTORY, EXPERIMENT_NAME)
    accelerator = Accelerator(
        project_dir=path_to_experiment,
        log_with="wandb" if LOG_WANDB else None,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    )
    if LOG_WANDB:
        accelerator.init_trackers(EXPERIMENT_NAME)

    ### Model Config ###
    config = Wav2Vec2Config(
        conv_dim=CONV_DIM,
        conv_stride=CONV_STRIDE,
        conv_kernel=CONV_KERNEL,
        conv_bias=CONV_BIAS,
        feature_projection_dropout_p=FEATURE_PROJ_DROPOUT_P,
        conv_positional_emb_drop_p=CONV_POS_EMB_DROP_P,
        conv_positional_emb_groups=CONV_POS_EMB_GROUPS,
        conv_positional_emb_kernel_size=CONV_POS_EMB_KERNEL_SIZE,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        embedding_dimension=EMBEDDING_DIMENSION,
        mlp_ratio=MLP_RATIO,
        mlp_dropout_p=MLP_DROPOUT_P,
        attention_dropout_p=ATTENTION_DROPOUT_P,
        transformer_encoder_dropout=TRANSFORMER_ENCODER_DROPOUT_P,
        # LayerDrop causes unused-parameter errors with DDP — disable on multi-GPU
        layer_dropout=LAYER_DROPOUT if accelerator.num_processes == 1 else 0.0,
        initializer_range=INITIALIZER_RANGE,
        num_codevector_groups=NUM_CODEVECTOR_GROUPS,
        num_codevectors_per_group=NUM_CODEVECTORS_PER_GROUP,
        codevector_dim=CODEVECTOR_DIM,
        pre_quantizer_dropout=PRE_QUANTIZER_DROPOUT_P,
        masking_probability=MASKING_PROBABILITY,
        masking_span_length=MASKING_SPAN_LENGTH,
        minimum_spans=MINIMUM_SPANS,
        contrastive_logits_temperature=CONTRASTIVE_LOGITS_TEMPERATURE,
        diversity_loss_weight=DIVERSITY_LOSS_WEIGHT,
        num_negatives=NUM_NEGATIVES,
    )

    ### Model ###
    model = Wav2Vec2ForPreTraining(config)
    n_params = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of Parameters: {n_params:,}")

    ### Datasets ###
    train_set = LibriSpeechDataset(
        path_to_data_root=PATH_TO_DATA_ROOT,
        include_splits=TRAIN_SPLITS,
        max_audio_duration=MAXIMUM_AUDIO_DURATION,
        min_audio_duration=MINIMUM_AUDIO_DURATION,
        sampling_rate=SAMPLING_RATE,
        return_transcripts=False,
    )
    test_set = LibriSpeechDataset(
        path_to_data_root=PATH_TO_DATA_ROOT,
        include_splits=TEST_SPLITS,
        max_audio_duration=MAXIMUM_AUDIO_DURATION,
        min_audio_duration=MINIMUM_AUDIO_DURATION,
        sampling_rate=SAMPLING_RATE,
        return_transcripts=False,
    )

    data_collator = Wav2Vec2CollateFunctionForPreTraining(config)
    minibatch_size = PER_GPU_BATCH_SIZE // GRADIENT_ACCUMULATION

    train_dataloader = DataLoader(
        train_set,
        batch_size=minibatch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        test_set,
        batch_size=minibatch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=data_collator,
    )

    ### Optimiser ###
    if not BIAS_WEIGHT_DECAY or not NORM_WEIGHT_DECAY:
        accelerator.print("Disabling Weight Decay on Some Parameters")
        wd_params, no_wd_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name and not BIAS_WEIGHT_DECAY:
                no_wd_params.append(param)
            elif "groupnorm" in name and not NORM_WEIGHT_DECAY:
                no_wd_params.append(param)
            else:
                wd_params.append(param)
        optimizer = torch.optim.AdamW(
            [{"params": wd_params, "weight_decay": WEIGHT_DECAY},
             {"params": no_wd_params, "weight_decay": 0.0}],
            lr=LEARNING_RATE, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON,
        )

    ### Scheduler ###
    scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=NUM_TRAINING_STEPS,
    )

    ### Hand Everything to Accelerate ###
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    accelerator.register_for_checkpointing(scheduler)

    ############################################################
    ### TRAINING
    ############################################################

    ### Resume ###
    if RESUME_FROM_CHECKPOINT is not None:
        path_to_checkpoint = os.path.join(path_to_experiment, RESUME_FROM_CHECKPOINT)
        with accelerator.main_process_first():
            accelerator.load_state(path_to_checkpoint)
        completed_steps = int(RESUME_FROM_CHECKPOINT.split("_")[-1])
        accelerator.print(f"Resuming from step {completed_steps}")
    else:
        completed_steps = 0

    train = True
    progress_bar = tqdm(
        range(completed_steps, NUM_TRAINING_STEPS),
        disable=not accelerator.is_local_main_process,
    )

    while train:

        accumulated_hours = 0
        accumulated_pct_masked = 0

        for batch in train_dataloader:

            # accumulate() handles no_sync() on intermediate steps for DDP —
            # gradient all-reduce only fires on the final accumulation step.
            with accelerator.accumulate(model):

                num_losses = batch["mask_time_indices"].sum()
                accumulated_hours = accumulated_hours + compute_batch_duration(
                    batch["attention_mask"], SAMPLING_RATE
                )
                accumulated_pct_masked = accumulated_pct_masked + (
                    num_losses / batch["sub_attention_mask"].sum()
                ) / GRADIENT_ACCUMULATION

                outputs = model(**batch)
                loss = outputs.loss / GRADIENT_ACCUMULATION
                accelerator.backward(loss)

                if accelerator.sync_gradients:

                    # Gather total masked tokens across all GPUs.
                    # On 1 GPU, gather() is a no-op that returns the input unchanged.
                    num_losses = accelerator.gather(num_losses.unsqueeze(0)).sum()

                    # Scale gradients so the update is normalised by masked tokens,
                    # not raw batch size.  num_processes cancels the DDP gradient average.
                    gradient_multiplier = accelerator.num_processes / num_losses
                    multiply_gradients(
                        accelerator.unwrap_model(model).parameters(),
                        gradient_multiplier,
                    )

                    # Gradient norm — get_scale() returns a Python float, no GPU sync.
                    scale = accelerator.scaler.get_scale() if accelerator.scaler is not None else 1.0
                    grad_norm = compute_gradient_norms(
                        accelerator.unwrap_model(model).parameters(), scale
                    )

                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    ### Gumbel temperature decay ###
                    gumbel_temp = max(
                        MAX_GUMBEL_TEMPERATURE * GUMBEL_TEMPERATURE_DECAY ** completed_steps,
                        MIN_GUMBEL_TEMPERATURE,
                    )
                    accelerator.unwrap_model(model).set_gumbel_temperature(gumbel_temp)

                    ### Logging ###
                    # accelerator.gather() is a no-op on 1 GPU and all-gathers on 8,
                    # so both paths share the same code.
                    if completed_steps % LOGGING_STEPS == 0:

                        train_loss   = accelerator.gather(outputs.loss.detach().unsqueeze(0)).sum() / num_losses
                        contr_loss   = accelerator.gather(outputs.contrastive_loss.detach().unsqueeze(0)).sum() / num_losses
                        div_loss     = accelerator.gather(outputs.diversity_loss.detach().unsqueeze(0)).sum() / num_losses
                        perplexity   = accelerator.gather(outputs.codevector_perplexity.detach().unsqueeze(0)).sum() / num_losses
                        hours_log    = accelerator.gather(accumulated_hours.unsqueeze(0)).sum()
                        pct_log      = accelerator.gather(accumulated_pct_masked.unsqueeze(0)).mean()

                        log = {
                            "train_loss":          train_loss,
                            "train_contrast_loss": contr_loss,
                            "train_div_loss":      div_loss,
                            "pct_masked":          pct_log,
                            "batch_hours":         hours_log,
                            "perplexity":          perplexity,
                            "lr":                  scheduler.get_last_lr()[0],
                            "temp":                gumbel_temp,
                            "grad_norm":           grad_norm,
                        }

                        log_str = "".join(
                            f"|{k[6:] if k.startswith('train_') else k}: "
                            f"{round(v.item() if torch.is_tensor(v) else v, 3)}"
                            for k, v in log.items()
                        )
                        if accelerator.is_main_process:
                            progress_bar.write(log_str)
                        if LOG_WANDB:
                            accelerator.log(log, step=completed_steps)

                    ### Evaluation ###
                    if completed_steps % EVALUATION_INTERVAL == 0:

                        if accelerator.is_main_process:
                            progress_bar.write("Evaluating...")

                        model.eval()
                        val_log = {"val_loss": 0.0, "val_contrast_loss": 0.0, "val_div_loss": 0.0}
                        all_val_losses = 0

                        for eval_batch in tqdm(eval_dataloader, disable=not accelerator.is_main_process):
                            eval_num_losses = eval_batch["mask_time_indices"].sum()
                            with torch.inference_mode():
                                eval_out = model(**eval_batch)

                            eval_num_losses = accelerator.gather(eval_num_losses.unsqueeze(0)).sum()
                            val_log["val_loss"]          += accelerator.gather(eval_out.loss.unsqueeze(0)).sum()
                            val_log["val_contrast_loss"] += accelerator.gather(eval_out.contrastive_loss.unsqueeze(0)).sum()
                            val_log["val_div_loss"]      += accelerator.gather(eval_out.diversity_loss.unsqueeze(0)).sum()
                            all_val_losses               += eval_num_losses

                        val_log = {k: v / all_val_losses for k, v in val_log.items()}
                        val_str = "".join(
                            f"|{k[4:]}: {round(v.item() if torch.is_tensor(v) else v, 3)}"
                            for k, v in val_log.items()
                        )
                        if accelerator.is_main_process:
                            progress_bar.write(val_str)
                        if LOG_WANDB:
                            accelerator.log(val_log, step=completed_steps)

                        model.train()

                    ### Checkpoint ###
                    if completed_steps % CHECKPOINT_INTERVAL == 0:

                        ckpt_path = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")
                        if accelerator.is_main_process:
                            progress_bar.write(f"Saving checkpoint → {ckpt_path}")

                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            accelerator.save_state(output_dir=ckpt_path)

                        if NUM_KEEP_CHECKPOINTS is not None and accelerator.is_main_process:
                            all_ckpts = sorted(
                                [c for c in os.listdir(path_to_experiment) if c.startswith("checkpoint_")],
                                key=lambda x: int(x.split("_")[-1]),
                            )
                            for old in all_ckpts[:-NUM_KEEP_CHECKPOINTS]:
                                old_path = os.path.join(path_to_experiment, old)
                                if os.path.isdir(old_path):
                                    shutil.rmtree(old_path)

                        accelerator.wait_for_everyone()

                    ### Done? ###
                    if completed_steps >= NUM_TRAINING_STEPS:
                        if accelerator.is_main_process:
                            progress_bar.write("Training complete!")
                        train = False
                        break

                    completed_steps += 1
                    progress_bar.update(1)

                    # Reset per-window accumulators after each optimizer step
                    accumulated_hours = 0
                    accumulated_pct_masked = 0

    accelerator.end_training()


if __name__ == "__main__":
    main()
