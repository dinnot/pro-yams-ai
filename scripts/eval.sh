#!/bin/bash
set -e

CHECKPOINT=${1:?"Usage: eval.sh <checkpoint_path> [num_games]"}
NUM_GAMES=${2:-200}

docker run --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    --rm \
    pro_yams_ai \
    --mode eval \
    --checkpoint "/app/checkpoints/$(basename "$CHECKPOINT")" \
    --eval_games "$NUM_GAMES"
