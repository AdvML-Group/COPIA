#!/usr/bin/env bash
set -e

# Usage:
#   bash run_copia_five_models.sh 0
# The first argument is the GPU id. If omitted, GPU 0 is used.
GPU=${1:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
PYTHON=${PYTHON:-python}

# Always run from the directory where this script is located.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMMON_ARGS=(
  --gpu "$GPU"
  --batch_size "$BATCH_SIZE"
)

echo "Running COPIA five-model attacks on GPU ${GPU}"

$PYTHON COPIA_attack_resnet.py --attack COPIA_Resnet_50_five_cf --source_model resnet50 "${COMMON_ARGS[@]}"
$PYTHON COPIA_attack_resnet.py --attack COPIA_Resnet_152_five_cf --source_model resnet152 "${COMMON_ARGS[@]}"

echo "All COPIA five-model attacks finished."
