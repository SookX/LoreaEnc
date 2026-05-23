#!/bin/bash
# Local CPU smoke test for the official Libri-Light 10min prepared split.
# This is intentionally tiny: it validates that the prepared directory is
# readable by CausalSpecUnit.train_ctc without running a full experiment.

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/mpl-cache}"

python3 -m CausalSpecUnit.train_ctc \
  --data-root data_local/datasets/librispeech/LibriSpeech \
  --cmvn-path none \
  --tokenizer-path dataset/bpe128.model \
  --train-splits librilight_10min \
  --output-dir outputs_local/librilight_10min_smoke \
  --variant xs \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum-steps 1 \
  --eval-batch-size 2 \
  --eval-split dev-clean \
  --eval-every 0 \
  --workers 0 \
  --dataloader-timeout 0 \
  --lr 1e-3 \
  --encoder-lr 3e-4 \
  --head-lr 1e-3 \
  --warmup-epochs 1 \
  --peak-epochs 1 \
  --noam-decay-rate 0.5 \
  --max-train-batches 2 \
  --seed 42 \
  --progress off \
  --log-every 1 \
  --save-every 1
