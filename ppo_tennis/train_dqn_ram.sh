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
RUN_DIR="${RUN_DIR:-runs/dqn_ram_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-250000}"
DEVICE="${DEVICE:-cpu}"
if [[ -e "$RUN_DIR" || -e "$RUN_DIR.console.log" ]]; then
  printf 'Experiment already exists: %s\n' "$RUN_DIR" >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$RUN_DIR")"
trap 'code=$?; printf "%s\n" "$code" > "$RUN_DIR.exitcode"' EXIT
printf 'Conda: %s\nPython: %s\nRun: %s\n' "$CONDA_DEFAULT_ENV" "$CONDA_PREFIX/bin/python" "$RUN_DIR"
"$CONDA_PREFIX/bin/python" -u dqn_ram.py train \
  --run-dir "$RUN_DIR" --total-timesteps "$TOTAL_TIMESTEPS" --device "$DEVICE" --threads 1 \
  --n-envs 8 --buffer-size 100000 --learning-starts 10000 --batch-size 128 \
  --train-freq 4 --gradient-steps 8 --target-update-interval 10000 \
  --exploration-fraction 0.4 --exploration-final-eps 0.05 --learning-rate 0.0001 \
  --auto-serve --serve-mode align_x_fire --target-mode "${TARGET_MODE:-recover_chase}" \
  --gamma 0.99 --beta "${BETA:-0.05}" --seed "${SEED:-0}" \
  --eval-freq 50000 --eval-episodes 5 --eval-max-steps "${EVAL_MAX_STEPS:-27000}" --log-freq 1000 \
  "$@" 2>&1 | tee "$RUN_DIR.console.log"

