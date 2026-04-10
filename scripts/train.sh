#!/bin/bash
set -e

CONFIG=${1:-config/default.yaml}
CHECKPOINT=${2:-""}

ARGS="--mode train --config /app/config/$(basename "$CONFIG")"
if [ -n "$CHECKPOINT" ]; then
    ARGS="$ARGS --checkpoint /app/checkpoints/$(basename "$CHECKPOINT")"
fi

docker run --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/config:/app/config" \
    -v "$(pwd)/logs:/app/logs" \
    --rm \
    pro_yams_ai $ARGS
