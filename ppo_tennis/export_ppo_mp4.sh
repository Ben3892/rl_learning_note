#!/usr/bin/env bash
# Render the trained PPO policy into an MP4; no training is performed.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

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
conda activate ls 

RUN_DIR="${RUN_DIR:-runs/tennis_cuda_v6_idle_yja}"
MODEL="${MODEL:-$RUN_DIR/best/best_model.zip}"
VIDEO="${VIDEO:-$RUN_DIR/videos/ppo_best_$(date +%Y%m%d_%H%M%S).mp4}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"

if [[ ! -f "$MODEL" ]]; then
  printf 'Model not found: %s\n' "$MODEL" >&2
  exit 1
fi
if [[ -e "$VIDEO" ]]; then
  printf 'Video already exists; set VIDEO to a new path: %s\n' "$VIDEO" >&2
  exit 1
fi
printf 'Conda: %s\nModel: %s\nVideo: %s\nSteps: %s (15 fps)\n' \
  "$CONDA_DEFAULT_ENV" "$MODEL" "$VIDEO" "$STEPS"

# Deterministic policy, matching evaluation. A fixed-length clip may span games.
exec "$CONDA_PREFIX/bin/python" -u tennis.py play \
  --model "$MODEL" --video "$VIDEO" --steps "$STEPS" \
  --seed "$SEED" --device "$DEVICE" --threads 4
