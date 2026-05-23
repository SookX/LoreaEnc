#!/bin/bash
# Submit divided official Libri-Light fine-tuning jobs.
#
# This submits 27 independent jobs:
#   3 subsets x 3 initializations x 3 seeds.
#
# Optional filters:
#   ONLY_SUBSET=librilight_10min
#   ONLY_CONDITION=iter2
#   ONLY_SEED=42

set -euo pipefail

SUBSETS=(librilight_10min librilight_1h librilight_10h)
CONDITIONS=(scratch iter1 iter2)
SEEDS=(42 43 44)

SCRIPT="slurm/causal_specunit/08_librilight_finetune_one.sh"

for subset in "${SUBSETS[@]}"; do
  if [ "${ONLY_SUBSET:-}" != "" ] && [ "${ONLY_SUBSET}" != "${subset}" ]; then
    continue
  fi
  for condition in "${CONDITIONS[@]}"; do
    if [ "${ONLY_CONDITION:-}" != "" ] && [ "${ONLY_CONDITION}" != "${condition}" ]; then
      continue
    fi
    for seed in "${SEEDS[@]}"; do
      if [ "${ONLY_SEED:-}" != "" ] && [ "${ONLY_SEED}" != "${seed}" ]; then
        continue
      fi
      echo "submit subset=${subset} condition=${condition} seed=${seed}"
      sbatch --export=ALL,SUBSET="${subset}",CONDITION="${condition}",SEED="${seed}" "${SCRIPT}"
    done
  done
done
