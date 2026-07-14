#!/bin/bash
# Rerun the bad benchmark cell only:
#   variant: librilight_1h
#   model:   iter2
#   seed:    44
#
# This wraps 10_benchmark_1h_10h_100h_3seeds.sh with filters, so it uses
# the exact same recipe and output layout as the benchmark table.
#
# Submit:
#   sbatch slurm/causal_specunit/11_rerun_librilight_1h_iter2_seed44.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_i2_s44
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_i2_s44.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_i2_s44.%j.err

set -euo pipefail

export SUBSETS="${SUBSETS:-librilight_1h}"
export CONDITIONS="${CONDITIONS:-iter2}"
export SEED_LIST="${SEED_LIST:-44}"
export CLEAN_FIRST="${CLEAN_FIRST:-1}"

exec bash slurm/causal_specunit/10_benchmark_1h_10h_100h_3seeds.sh
