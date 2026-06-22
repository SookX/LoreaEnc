# SALMA 2026 Resubmission Plan

**Target venue:** SALMA 2026 (Speech and Audio Language Models Workshop @ EMNLP 2026, Budapest)
**Deadline:** July 27, 2026, 23:59 UTC-12 (≈ 5 weeks from 2026-06-22)
**Workshop page:** https://salma-workshop.github.io/salma-2026/

This document is a complete execution playbook. Every command in it is intended to be runnable
on the bg-eng-01 cluster as-is. Sections marked **"RUN NOW"** can be launched without further
code; sections marked **"NEEDS CODE"** require an item to be implemented first.

---

## 0. Why we're doing this

The ML4Audio submission was rejected with three substantive critiques:

1. **Limited algorithmic novelty.** Reviewer reading: "MelHuBERT objective + off-the-shelf
   SqueezeFormer; engineering assembly rather than algorithmic breakthrough."
2. **Restricted to clean English LibriSpeech.** Reviewer reading: "Need non-English
   low-resource or domain-divergent corpora to back the low-resource claim."
3. **Missing competitive baselines at matched compute budget.** Reviewer reading: "Compare
   against knowledge distillation from a larger pretrained teacher into a compact student."

The resubmission addresses each critique directly. The plan also adds a learned-quantizer
ablation (VQ / RVQ / parallel-VQ) as a strengthening item, contingent on a fast go/no-go test.

---

## 1. CRITICAL METHODOLOGY DISCLAIMER — read this once, then refer back

**Our recipe is NOT MelHuBERT.** It shares two surface traits (HuBERT-style masked prediction
objective; mel-spectrogram input domain), but the methodology differs in every other respect:

| | Our recipe | MelHuBERT (original paper) |
|---|---|---|
| Feature unit | PCA-64 of size-8 stride-4 log-mel chunks | per-frame log-mel features (no chunking) |
| Target codebooks | Two parallel k-means (K=100 + K=500) | One k-means at fixed K |
| Refinement | Iter-2: re-cluster encoder hidden states + continue SSL | None (iter-1 only in their primary recipe) |
| Backbone | SqueezeFormer-XS, 9M params, ASR-specialised | Vanilla Transformer, scale-as-published |
| Tokenizer / fine-tune | BPE-128 CTC, no LM, greedy decode | Their reported recipe varies |

**The Transformer baseline we built (variant `mh9m`) IS a faithful matched-scale MelHuBERT
reproduction.** That row is correctly labelled "MelHuBERT-style (Transformer)" in
[example_paper.tex](example_paper.tex).

**The 150k single-codebook SqueezeFormer row we're adding in Phase 1 is NOT a MelHuBERT
comparison.** It is an *ablation of our recipe* with the dual codebook collapsed to one. It
shares the chunking/PCA pipeline with our recipe and the backbone with our recipe. It uses
the codebook geometry MelHuBERT uses (single). It is a between-our-recipe-and-MelHuBERT row,
neither one nor the other. Paper label: **"Single-codebook ablation (SqueezeFormer-XS)"** or
**"Ours, single codebook"**, NOT "MelHuBERT-style SqueezeFormer".

If we accidentally write "MelHuBERT on SqueezeFormer" anywhere in the paper, a reviewer who
checked the MelHuBERT paper will catch us conflating two different recipes. **Don't.**

---

## 2. Status as of 2026-06-22

### Code on `main`

- [CausalSpecUnit/quantizer.py](CausalSpecUnit/quantizer.py) — VQEMA, RVQ, ParallelVQ classes. Unit tests pass.
- [CausalSpecUnit/quantizer_test.py](CausalSpecUnit/quantizer_test.py) — 10 sanity tests, all green.
- [CausalSpecUnit/train_quantizer.py](CausalSpecUnit/train_quantizer.py) — quantizer trainer (cache build + k-means init + EMA loop).
- [slurm/causal_specunit/20_train_quantizer.sh](slurm/causal_specunit/20_train_quantizer.sh) — parametric slurm wrapper for the trainer.
- [CausalSpecUnit/finetune_mh9m.py](CausalSpecUnit/finetune_mh9m.py) — standalone MelHuBERT-Transformer fine-tune.
- [CausalSpecUnit/eval_mh9m.py](CausalSpecUnit/eval_mh9m.py) — eval for mh9m fine-tuned checkpoints.
- [slurm/causal_specunit/12_melhubert_transformer_mh9m_baseline.sh](slurm/causal_specunit/12_melhubert_transformer_mh9m_baseline.sh) — pretrain + fine-tune mh9m.
- [slurm/causal_specunit/16_melhubert_finetune_standalone.sh](slurm/causal_specunit/16_melhubert_finetune_standalone.sh) — standalone fine-tune.
- [slurm/causal_specunit/17_eval_mh9m_all.sh](slurm/causal_specunit/17_eval_mh9m_all.sh) — eval sweep for all mh9m cells.
- [scripts/aggregate_mh9m.py](scripts/aggregate_mh9m.py) — mh9m results aggregator.

### Numbers in the paper (commit `c8785ac`)

| Row | 1h clean / other | 10h clean / other | 100h clean / other |
|---|---|---|---|
| Scratch (SqueezeFormer-XS) | 94.9±0.1 / 95.9±0.2 | 75.0±0.6 / 84.1±0.3 | 19.6±0.3 / 42.7±0.3 |
| MelHuBERT-style (Transformer, mh9m) | 78.1±0.6 / 86.1±0.2 | 49.5±0.3 / 67.0±0.3 | 20.7±0.3 / 43.5±0.1 |
| + SSL iter-1 (ours, dual k-means) | 63.5±0.2 / 75.1±0.1 | 38.9±0.4 / 57.5±0.2 | 17.2±0.2 / 38.5±0.1 |
| + SSL iter-2 (ours) | 56.8±0.3 / 68.7±0.2 | 34.1±0.2 / 51.5±0.3 | 16.1±0.2 / 36.2±0.1 |

### What's missing for SALMA

| Row / claim | Status |
|---|---|
| Single-codebook ablation @ matched 150k SSL steps | Need to run (Phase 1) |
| VQ ablation (smoke) | Need to run (Phase 0) |
| Distillation baseline @ matched compute | Need to build + run (Phase 3) |
| French MLS validation | Need to build + run (Phase 2) |
| German MLS validation | Need to build + run (Phase 2) |

---

## 3. Phase 0 — VQ smoke test  *(week 1, ~3-4 days)*

**Goal:** decide whether VQ helps over k-means at matched everything. 1 seed, 10h only,
go/no-go in <4 days.

### 3.1 Status

- Quantizer math + trainer: **DONE** (Stages 1-3 already on `main`).
- Per-utterance VQ target extraction: **NEEDS CODE** (`generate_vq_targets.py`).
- SSL pretrain on VQ targets: **NEEDS PATCH** to `pretrain_ssl.py` (k_fine plumbing for K=600).
- Fine-tune + eval: reuses existing scripts.

### 3.2 What to launch RIGHT NOW

Three quantizer training jobs (independent; the first one builds the shared PCA-64 chunks
cache so launch it first):

```bash
git pull
# First job builds the ~22 GB chunks cache (~30-60 min) then trains RVQ.
QUANTIZER_TYPE=rvq K1=100 K2=500 \
  sbatch slurm/causal_specunit/20_train_quantizer.sh

# After the first finishes (or just queue them — they wait on the cache file existing):
QUANTIZER_TYPE=vq K1=600 \
  sbatch slurm/causal_specunit/20_train_quantizer.sh

QUANTIZER_TYPE=parallel K1=100 K2=500 \
  sbatch slurm/causal_specunit/20_train_quantizer.sh
```

Outputs land in:
- `outputs/causal_specunit/vq/rvq_100_500/{state.pt, metrics.json, train.jsonl}`
- `outputs/causal_specunit/vq/vq_600/...`
- `outputs/causal_specunit/vq/parallel_100_500/...`

**Health check after each finishes:**
```bash
cat outputs/causal_specunit/vq/rvq_100_500/metrics.json
# Want ppl1 > 70 (out of 100), ppl2 > 350 (out of 500), active_q* close to K
```

If any cell has perplexity < 0.25 × K, that quantizer collapsed — retry with `BETA=0.1` or
revisit before continuing. **This is the smoke gate for VQ.**

### 3.3 What still needs code (I'll write this)

`CausalSpecUnit/generate_vq_targets.py` (~150 lines). Reads:
- `--source-targets-dir outputs/causal_specunit/targets_960h_c8/` (for PCA, CMVN, chunk geometry)
- `--quantizer-dir outputs/causal_specunit/vq/<type>/` (for trained quantizer)

Writes:
- `outputs/causal_specunit/targets_960h_c8_<type>/{targets.pt, metadata.json, cluster_artifacts.joblib symlink}`

Per-utterance contents:
- RVQ / ParallelVQ: `{uid: {"z100": z1, "z500": z2}}`
- Flat VQ-600: `{uid: {"z500": z}}` (single-stream)

`pretrain_ssl.py` patch (~20 lines): read `k_coarse` and `k_fine` from `metadata.json` of
the targets dir and pass them through to the model so `K_f=600` works for the flat VQ case.

### 3.4 SSL pretrain commands for the smoke (once Stage 4 + patch are in)

```bash
# Two-stream learned VQ — fits existing dual-codebook recipe (CODEBOOK_MODE=both).
TARGETS=outputs/causal_specunit/targets_960h_c8_rvq_100_500 \
SSL_OUTPUT_DIR=outputs/causal_specunit/ssl_rvq_iter1_150k \
sbatch slurm/causal_specunit/02_pretrain_ssl_200k_c8.sh   # 150k steps via STEPS env

# Single-stream flat VQ — needs --codebook-mode fine and K_f=600 pathway.
TARGETS=outputs/causal_specunit/targets_960h_c8_vq_600 \
SSL_OUTPUT_DIR=outputs/causal_specunit/ssl_vq600_iter1_150k \
CODEBOOK_MODE=fine \
sbatch slurm/causal_specunit/02_pretrain_ssl_200k_c8.sh
```

### 3.5 Fine-tune + eval (1 seed, 10h only)

```bash
SUBSETS="librilight_10h" SEED_LIST="42" \
SSL_CHECKPOINT=outputs/causal_specunit/ssl_rvq_iter1_150k/checkpoint_step150000/checkpoint.pt \
OUTPUT_ROOT=outputs/causal_specunit/vq_smoke/rvq \
sbatch slurm/causal_specunit/10_benchmark_1h_10h_100h_3seeds.sh
# (similar for vq_600 and parallel_100_500)
```

### 3.6 Decision criteria

**GO (expand to full study, 3 seeds × 3 budgets):** any VQ cell beats matched-budget
k-means iter-1 by ≥2 WER points test-other at 10h.

**NO-GO (drop from paper, mention as preliminary):** all VQ cells within ±1 WER point of
matched k-means.

---

## 4. Phase 1 — Single-codebook ablation at matched 150k SSL  *(week 1-2)*

**Goal:** add the single-codebook ablation row to the headline table at matched 150k
budget, so the dual-codebook contribution can be read directly off Table 1 instead of
buried in Table 2 at 50k steps.

This addresses reviewer critique #1 indirectly (better-controlled novelty argument) and is
the strongest way to honor the "single-codebook variant" complaint without redoing Table 1
from scratch.

### 4.1 Naming

Paper label: **"Single-codebook ablation (SqueezeFormer-XS, K=500)"** or **"Ours, single
codebook"**. **NOT "MelHuBERT-style SqueezeFormer"** — see §1.

### 4.2 Status

- Code: **DONE.** Use existing `02_pretrain_ssl_*.sh` and `10_benchmark_*.sh` with overrides.
- Targets: **DONE** — `targets_960h_c8` already has `k_coarse=100, k_fine=500`; we use
  `--codebook-mode fine` to train against only the K=500 stream.

### 4.3 Commands — **RUN NOW**

```bash
# SSL pretrain: 150k steps, single-codebook fine (K=500) on SqueezeFormer-XS.
sbatch slurm/causal_specunit/02_pretrain_ssl_200k_c8.sh \
  --export=ALL,CODEBOOK_MODE=fine,STEPS=150000,SSL_OUTPUT_DIR=outputs/causal_specunit/ssl_single_k500_iter1_150k
# (or equivalent env var override — adjust to your wrapper conventions)
```

If your `02_pretrain_ssl_200k_c8.sh` doesn't support env-var overrides for STEPS /
CODEBOOK_MODE / SSL_OUTPUT_DIR yet, copy it to `02b_pretrain_ssl_single_k500.sh` and hard-code
those three values inside the script. Trivial copy-paste; 5 minutes.

### 4.4 Fine-tune sweep

```bash
# 3 budgets × 2 seeds = 6 fine-tune cells. Seeds 42, 43.
SUBSETS="librilight_1h librilight_10h train-clean-100" \
SEED_LIST="42 43" \
SSL_CHECKPOINT=outputs/causal_specunit/ssl_single_k500_iter1_150k/checkpoint_step150000/checkpoint.pt \
CONDITIONS="ssl_single_k500" \
OUTPUT_ROOT=outputs/causal_specunit/benchmark_single_k500 \
sbatch slurm/causal_specunit/10_benchmark_1h_10h_100h_3seeds.sh
```

If `CONDITIONS` isn't recognized in script 10, just hard-code a minimal new wrapper
`10b_benchmark_single_k500.sh` that mirrors script 10 but with `ssl_single_k500` as the only
condition and the SSL checkpoint path locked in.

### 4.5 Eval

Already runs at the end of `10_benchmark_*.sh`. Output: `eval_results.json` per cell.

### 4.6 Aggregate into paper

Add to existing `scripts/aggregate_mh9m.py` (or a new `aggregate_single_codebook.py` clone)
a row that prints the mean ± std of the new cells.

---

## 5. Phase 2 — French + German MLS  *(weeks 2-4)*

**Goal:** validate the recipe on two non-English audiobook-domain languages. Closes
reviewer critique #2.

### 5.1 Corpus — locked in

**Multilingual LibriSpeech (MLS)**, https://www.openslr.org/94/. Reasons:
- Same audiobook domain as English LibriSpeech → cleanest cross-language comparison.
- Standard splits → no ad-hoc subset choices to defend.
- Built-in 10h "limited supervision" subset → matches our low-resource frame.

**Not Common Voice** — that's crowd-sourced read-aloud, which adds a *domain* confound on
top of the *language* axis. Avoid for this paper.

### 5.2 Scope cuts to fit calendar

- **No 1h budget for FR/DE.** MLS doesn't ship a 1h split; inventing one invites reviewer
  pushback. Use **10h + full** only.
- **No MelHuBERT-mh9m per language.** Costs ~30 H200-hours per language; the cross-language
  recipe-transfer claim doesn't need it.
- **2 seeds** (not 3). Acknowledged in paper text.

### 5.3 Per-language pipeline

| Step | What | Wall-clock |
|---|---|---|
| Download MLS-FR (~150 GB) and MLS-DE (~400 GB) | Use cluster's external bandwidth | 1-2 days each |
| Validate audio format (16 kHz, mono, flac) | Should already be MLS-normalised | 1 hr |
| Speaker-balanced 960h unlabeled subset per language | `scripts/prepare_mls.py` (NEW) | 1 hr each |
| Per-language BPE-128 tokenizer | `scripts/build_tokenizer_mls.py` (NEW) | 30 min each |
| Targets per language (PCA + dual k-means, our recipe) | Reuse `generate_targets.py` with new paths | 10 hrs each |
| SSL pretrain iter-1 (150k) + iter-2 (100k) per language | Reuse `02_pretrain_ssl_*.sh` | 25 hrs each (2 H200) |
| Fine-tune: scratch / iter-1 / iter-2 × {10h, full} × 2 seeds = 12 cells | Reuse `10_benchmark_*.sh` | 24 hrs each (4 H200) |
| Eval on standard MLS test sets | Reuse `evaluate_ctc.py` | 2 hrs each |

### 5.4 Code to write

- `scripts/prepare_mls.py` (~120 lines) — speaker-balanced 960h sampling; writes uid lists per language.
- `scripts/build_tokenizer_mls.py` (~80 lines) — per-language BPE-128 via SentencePiece.
- `CausalSpecUnit/data.py` — add `iter_mls_items(data_root, language, split)` (~30 lines).
- `slurm/causal_specunit/30_*.sh` through `34_*.sh` — copies of the English slurm scripts with paths overridden for FR/DE.

### 5.5 Commands

Cannot start yet — depends on `prepare_mls.py` and the data download. Once both are in:

```bash
# Stage 2a: prepare MLS-FR data
sbatch scripts/prepare_mls.sh --language fr --output dataset/mls/fr

# Stage 2b: targets
sbatch slurm/causal_specunit/30_generate_targets_mls.sh --language fr

# Stage 2c: pretrain
sbatch slurm/causal_specunit/31_pretrain_ssl_mls.sh --language fr --iter 1
sbatch slurm/causal_specunit/32_pretrain_ssl_mls.sh --language fr --iter 2

# Stage 2d: fine-tune sweep
sbatch slurm/causal_specunit/33_benchmark_finetune_mls.sh --language fr --conditions "scratch iter1 iter2" --subsets "mls_fr_10h mls_fr_full" --seeds "42 43"

# Stage 2e: eval
sbatch slurm/causal_specunit/34_eval_mls.sh --language fr
```

Mirror for `--language de`.

---

## 6. Phase 3 — Distillation baseline  *(weeks 2-4, parallel with Phase 2)*

**Goal:** add a distillation row to Table 1 at matched compute budget. Closes reviewer
critique #3 directly.

### 6.1 Scope — locked tight

- **Teacher:** public HuBERT-Base from HuggingFace (`facebook/hubert-base-ls960`), 95M params, frozen.
- **Student:** SqueezeFormer-XS, 9M params, randomly initialised (no SSL pretrain).
- **Distillation loss:** **logits-only** (KL divergence on CTC log-probabilities). NO hidden-state distillation in v1.
  - Hidden-state alignment between waveform-input HuBERT and mel-input SqueezeFormer requires a learned adapter and a layer-mapping experiment. Defer to follow-up if logits-only doesn't tell a clear story.
- **Compute budget:** **matched to our SSL** (~42 H200-hours of distillation training, not more) so the comparison is at matched compute, as the reviewer demanded.
- **Data:** same 960h unlabeled LibriSpeech, same fine-tune budgets (1h/10h/100h), same eval (test-clean/test-other).
- **No wav2vec2** in v1. HuBERT alone is enough to answer the reviewer; wav2vec2 is redundant for a single paper.
- **No "smaller pretrained HuBERT/wav2vec2 variants from scratch."** That would double the calendar. The reviewer's actual demand was "knowledge distillation from a larger pre-trained teacher into a compact student encoder" — public-teacher → our-student fits that exactly. Building new compact teachers from scratch is a separate paper.

### 6.2 Code to write

`CausalSpecUnit/distill_hubert_to_xs.py` (~300 lines):

1. Load pretrained HuBERT-Base from `transformers` (offline cache if cluster lacks internet).
2. Build SqueezeFormer-XS student from scratch (use existing `CausalSpecUnitCTC(variant='xs')`).
3. Distillation training loop:
   - For each audio batch: teacher forward (waveform input) → CTC log-probs.
   - Student forward (mel input via our existing pipeline) → CTC log-probs.
   - Resample teacher log-probs to student's time resolution (HuBERT outputs 50 Hz, SqueezeFormer-XS outputs 25 Hz given our chunk-stride 4).
   - Loss: KL(student || teacher) at common time steps.
   - AdamW, bf16 autocast, gradient clipping, same hyperparameters as iter-1 pretrain.
4. Save best student checkpoint by validation CTC loss on dev-other.

`slurm/causal_specunit/40_distill_hubert_to_xs.sh` — slurm wrapper. Single job, 4 H200s, ~10 hrs walltime.

`slurm/causal_specunit/41_distill_finetune.sh` — fine-tune sweep on the distilled student.

### 6.3 Commands (once code is in)

```bash
# Distillation training (one-shot, no per-budget split).
sbatch slurm/causal_specunit/40_distill_hubert_to_xs.sh

# Fine-tune the distilled student at 1h / 10h / 100h with 2 seeds.
SSL_CHECKPOINT=outputs/causal_specunit/distill_hubert_xs/checkpoint_best/checkpoint.pt \
SUBSETS="librilight_1h librilight_10h train-clean-100" \
SEED_LIST="42 43" \
OUTPUT_ROOT=outputs/causal_specunit/distill_benchmark \
sbatch slurm/causal_specunit/41_distill_finetune.sh
```

### 6.4 Risk

The HuBERT teacher consumes raw waveform; our student consumes log-mel. There's no shared
hidden-state space without an adapter, hence logits-only. If logits-only distillation
substantially underperforms our iter-2, that's actually the *strongest* result we could get
— it directly says "at matched compute, our recipe beats distillation." If it matches or
beats iter-2, the paper still benefits because we're transparent about a competitive
baseline.

---

## 7. Phase 4 — Paper integration  *(week 5)*

### 7.1 New Table 1 (final form, expanded)

```
Method                                           | 1h c/o          | 10h c/o         | 100h c/o
-------------------------------------------------|-----------------|-----------------|----------
SqueezeFormer-XS (scratch)                       | exist           | exist           | exist
SqueezeFormer-XS + distillation (HuBERT-Base)    | Phase 3 NEW     | Phase 3 NEW     | Phase 3 NEW
MelHuBERT-style Transformer (mh9m, 9M)           | exist           | exist           | exist
Ours, single-codebook ablation (SqueezeFormer)   | Phase 1 NEW     | Phase 1 NEW     | Phase 1 NEW
+ SSL iter-1 (ours, dual k-means)                | exist           | exist           | exist
+ SSL iter-2 (ours)                              | exist           | exist           | exist
+ SSL iter-2 (ours, dual VQ)                     | Phase 0 maybe   | Phase 0 maybe   | Phase 0 maybe
```

### 7.2 New cross-language Table

```
Lang | 10h c/o          | full c/o
-----|------------------|------------------
EN (LibriSpeech-100)  | exist            | exist
FR (MLS, ~1100h)      | Phase 2 NEW × 3  | Phase 2 NEW × 3   (× 3 conditions: scratch / iter-1 / iter-2)
DE (MLS, ~3200h)      | Phase 2 NEW × 3  | Phase 2 NEW × 3
```

### 7.3 Text changes

| Section | What to update |
|---|---|
| **Title** | Keep as-is. "Compute-Frugal Spectrogram-Only SSL for Low-Label Speech Recognition" — covers multilingual without rewording. |
| **Abstract** | Replace "low-label English ASR" with "low-label ASR across three audiobook-domain languages". Add one sentence: *"At matched compute, the recipe matches or beats logits-only distillation from a HuBERT-Base teacher into the same compact student."* |
| **§1 Intro** | Drop the "single-corpus" framing. Drop "we therefore do not claim multilingual performance" sentence. |
| **§3 Setup** | Add brief MLS-FR / MLS-DE setup description (~3 sentences). |
| **§3 Main results** | New paragraph on the distillation row. New paragraph on cross-language consistency. |
| **§3 Single-codebook ablation** | Move existing Table 2 content into a stronger paragraph using the new 150k matched-budget numbers. **Label this as "single-codebook ablation of our recipe", NOT "MelHuBERT comparison" — see §1 of this plan.** |
| **§4 Discussion** | New "What carries the gap (extended)" paragraph integrating distillation row. New "Cross-language consistency" paragraph. New "VQ extension" paragraph if Phase 0 went GO. |
| **§4 Limitations** | Remove "all experiments are English LibriSpeech". Add: *"Three languages, all audiobook-domain; we do not claim transfer to read-speech or spontaneous-speech corpora."* |

---

## 8. Calendar and compute summary

| Phase | Calendar | H200-hours | Critique closed |
|---|---|---|---|
| **0 (VQ smoke)** | week 1 (3-4 days) | ~78 | #1 (partially) |
| **1 (single-codebook ablation at 150k)** | week 1-2 (3-4 days) | ~30 | strengthens existing answer to #1 |
| **2 (FR + DE MLS)** | weeks 2-4 (~3 weeks) | ~100 | **#2 (the big one)** |
| **3 (distillation)** | weeks 2-4 (~2 weeks, parallel to #2) | ~60 | **#3 (the big one)** |
| **4 (paper integration)** | week 5 | 0 | — |
| **Total** | 5 weeks | **~270 H200-hours** | All three reviewer critiques |

Fits the July 27 deadline if we start Phase 0 + Phase 1 commands today.

---

## 9. What to do RIGHT NOW

In priority order, executable on the cluster after `git pull`:

```bash
# 1. Queue all three quantizer training jobs (Phase 0). First one builds the chunks cache.
QUANTIZER_TYPE=rvq K1=100 K2=500 sbatch slurm/causal_specunit/20_train_quantizer.sh
QUANTIZER_TYPE=vq K1=600 sbatch slurm/causal_specunit/20_train_quantizer.sh
QUANTIZER_TYPE=parallel K1=100 K2=500 sbatch slurm/causal_specunit/20_train_quantizer.sh
```

```bash
# 2. Queue the single-codebook SSL pretrain at 150k (Phase 1). Independent of #1.
#    May need to copy 02_pretrain_ssl_200k_c8.sh to a new file with these settings hard-coded:
#       --codebook-mode fine
#       --max-steps 150000
#       --output-dir outputs/causal_specunit/ssl_single_k500_iter1_150k
#    Then sbatch the new file.
```

```bash
# 3. Confirm MLS-FR and MLS-DE download paths. If using cluster's wget/aria2:
mkdir -p dataset/mls
cd dataset/mls
wget https://dl.fbaipublicfiles.com/mls/mls_french.tar.gz   # ~150 GB
wget https://dl.fbaipublicfiles.com/mls/mls_german.tar.gz   # ~400 GB
# These will run for ~1-2 days. Start now to lose no calendar time.
```

```bash
# 4. (Local / login node) Start designing the distillation script.
#    Pre-download the HuBERT-Base checkpoint into the cluster's HuggingFace cache:
HF_HOME=/valhalla/projects/bg-eng-01/.hf_cache \
python -c "from transformers import HubertModel; HubertModel.from_pretrained('facebook/hubert-base-ls960')"
```

While those four things run on the cluster, I (Claude) will write:
- `CausalSpecUnit/generate_vq_targets.py` (Phase 0 Stage 4)
- `pretrain_ssl.py` k_fine plumbing patch (Phase 0 Stage 6)
- `scripts/prepare_mls.py` (Phase 2)
- `CausalSpecUnit/distill_hubert_to_xs.py` (Phase 3)

---

## 10. What to cut if calendar slips

In rough priority order — drop in this order if the cluster queue gets congested or any
phase's wall-clock blows out:

1. **First cut: VQ scale-up.** Keep Phase 0 (1-seed smoke) only. Report as preliminary.
2. **Second cut: French.** Keep just German (3200h, more headroom for the low-resource
   claim). Drops Phase 2 to ~50 H200-hours.
3. **Third cut: distillation hidden-state loss.** Stay with logits-only. Already the plan.
4. **Never cut:** at least one of {French, German}, the single-codebook ablation row (Phase 1),
   and the distillation row (Phase 3). Those are the load-bearing critique-closers.

---

## 11. Decision criteria along the way

| Question | Threshold | If yes | If no |
|---|---|---|---|
| Does Phase 0 VQ beat matched k-means at 10h test-other? | ≥2 WER points | Scale to full study (Phase 0b: 3 seeds × 3 budgets) | Drop VQ from paper, mention as preliminary |
| Does Phase 1 single-codebook at 150k beat single-codebook at 50k by ≥3 points? | ≥3 WER points | Use as headline-table row | Use existing 50k numbers, save compute |
| Does distillation at matched compute beat iter-2? | iter-2 better | Strong paper claim | Disclose honestly; still useful baseline |
| Do FR and DE both show scratch < iter-1 < iter-2? | Both monotone | Strong cross-language claim | Discuss the asymmetry honestly |

---

## 12. Methodology vocabulary — use these consistently in the paper

| Concept | Correct term | Wrong term |
|---|---|---|
| MelHuBERT recipe with vanilla Transformer at compact scale | "MelHuBERT-style (Transformer)" | (fine) |
| Our recipe with single codebook instead of dual | "Single-codebook ablation" or "Ours, single codebook" | ❌ "MelHuBERT-style SqueezeFormer" |
| Our recipe at iter-2 | "Our recipe (iter-2)" or "Ours" | ❌ "spectrogram-domain HuBERT" |
| HuBERT-Base → SqueezeFormer-XS via KD | "Distilled from HuBERT-Base" | ❌ "HuBERT student" |
| French / German MLS | "MLS-FR" / "MLS-DE" (formal) or "French MLS" / "German MLS" (prose) | ❌ "low-resource MLS" — MLS-FR has 1100h, not low-resource in absolute terms; the *labeled* 10h split is what makes the experiment low-resource |
