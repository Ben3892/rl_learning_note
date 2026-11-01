#!/usr/bin/env bash
# Follow-up to tennis_cuda_v5: earlier/stronger serve-idle penalty.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Non-interactive Bash needs Conda's shell function before activation.
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -r "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -r "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  printf 'Cannot find Conda. Initialize Conda before running this script.\n' >&2
  exit 1
fi
conda activate yja
PYTHON_BIN="$CONDA_PREFIX/bin/python"
printf 'Conda environment: %s\nPython: %s\n' "$CONDA_DEFAULT_ENV" "$PYTHON_BIN"
RUN_DIR="${RUN_DIR:-runs/tennis_cuda_v6_idle_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -e "$RUN_DIR" || -e "$RUN_DIR.console.log" ]]; then
  printf 'Experiment already exists: %s\n' "$RUN_DIR" >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$RUN_DIR")"
trap 'code=$?; printf "%s\n" "$code" > "$RUN_DIR.exitcode"' EXIT
printf 'Run: %s\nSteps: %s\nPython: %s\n' "$RUN_DIR" "$TOTAL_TIMESTEPS" "$PYTHON_BIN"

# Keep the v5 seed, rollout, learning settings and receiving rewards.
# Change only serve-idle shaping and the number of evaluation episodes.
"$PYTHON_BIN" -u tennis.py train \
  --run-dir "$RUN_DIR" \
  --device "$DEVICE" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --n-envs 4 --n-steps 128 --batch-size 256 --threads 4 \
  --seed 0 --reward-shaping \
  --return-bonus 0.1 --position-weight 0.1 --position-cap 0.05 \
  --position-mode intercept_2d --tracking-penalty 0.01 --tracking-radius 12 \
  --idle-grace 15 --idle-penalty 0.05 \
  --save-freq 100000 --eval-freq 100000 --eval-episodes 5 \
  --eval-max-steps 5000 --eval-idle-limit 150 \
  "$@" 2>&1 | tee "$RUN_DIR.console.log"
