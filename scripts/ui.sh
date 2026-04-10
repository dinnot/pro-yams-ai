#!/bin/bash
set -e

CHECKPOINT=${1:-""}
PORT=${2:-8080}

ARGS="--log_dir /app/logs --static_dir /app/static --port $PORT"
if [ -n "$CHECKPOINT" ]; then
    ARGS="$ARGS --checkpoint /app/checkpoints/$(basename "$CHECKPOINT")"
fi

docker run --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/logs:/app/logs" \
    -p "$PORT:$PORT" \
    --rm \
    --entrypoint /usr/local/bin/pro_yams_ui \
    pro_yams_ai $ARGS
