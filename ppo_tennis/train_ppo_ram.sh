#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -r "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -r "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  printf 'Cannot find Conda.\n' >&2
  exit 1
fi
conda activate yja
RUN_DIR="${RUN_DIR:-runs/tennis_ram_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-250000}"
DEVICE="${DEVICE:-cuda}"
if [[ -e "$RUN_DIR" || -e "$RUN_DIR.console.log" ]]; then
  printf 'Experiment already exists: %s\n' "$RUN_DIR" >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$RUN_DIR")"
trap 'code=$?; printf "%s\n" "$code" > "$RUN_DIR.exitcode"' EXIT
printf 'Conda: %s\nPython: %s\nRun: %s\n' "$CONDA_DEFAULT_ENV" "$CONDA_PREFIX/bin/python" "$RUN_DIR"
"$CONDA_PREFIX/bin/python" -u ram_ppo.py train \
  --run-dir "$RUN_DIR" --total-timesteps "$TOTAL_TIMESTEPS" \
  --device "$DEVICE" --threads 1 --n-envs 8 --n-steps 256 --batch-size 256 \
  --auto-serve --gamma 0.99 --target-mode "${TARGET_MODE:-recover_chase}" --beta "${BETA:-0.05}" --learning-rate 0.0001 \
  --eval-freq 50000 --eval-episodes 5 --seed "${SEED:-0}" \
  "$@" 2>&1 | tee "$RUN_DIR.console.log"

