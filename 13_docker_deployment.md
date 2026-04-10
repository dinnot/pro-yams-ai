# Task 13: Docker & Deployment

## Overview

Create the Docker configuration for reproducible training runs. The container packages the training binary with all CUDA/libtorch dependencies. Checkpoints, configs, and logs are persisted via volume mounts.

## Prerequisites

- All previous tasks completed (the full training system is buildable)

---

## 1. Base Image Strategy

Standard PyTorch containers do not yet support the RTX 5080 (Blackwell `sm_120` architecture). Therefore, we will build our environment from scratch using NVIDIA's CUDA 12.8 base image and pull the nightly `libtorch` binaries directly during the build process.

We use `ubuntu24.04` as the OS base because it natively uses GCC 13, avoiding bleeding-edge compiler conflicts with CUDA.

---

## 2. Dockerfile

```dockerfile
# docker/Dockerfile

# --- Build stage ---
FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download Nightly LibTorch (CUDA 12.8, CXX11 ABI)
WORKDIR /opt
RUN wget -q [https://download.pytorch.org/libtorch/nightly/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip](https://download.pytorch.org/libtorch/nightly/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip) && \
    unzip -q libtorch-cxx11-abi-shared-with-deps-latest.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-latest.zip

# Copy source
COPY . /app
WORKDIR /app/build

# Build in Release mode, pointing CMake to our downloaded libtorch
RUN cmake .. \
    -DCMAKE_PREFIX_PATH=/opt/libtorch \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    && make -j$(nproc)

# --- Runtime stage ---
FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime

# Copy the libtorch shared libraries so the binary can link at runtime
COPY --from=builder /opt/libtorch /opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Copy only the built binaries
COPY --from=builder /app/build/pro_yams_ai /usr/local/bin/pro_yams_ai
COPY --from=builder /app/build/pro_yams_ui /usr/local/bin/pro_yams_ui

# Copy default config and static UI assets
COPY config/default.yaml /app/config/default.yaml
COPY src/ui/static /app/static

# Create directories for mounted volumes
RUN mkdir -p /app/checkpoints /app/logs

WORKDIR /app

# Default: run training
ENTRYPOINT ["/usr/local/bin/pro_yams_ai"]
CMD ["--mode", "train", "--config", "/app/config/default.yaml"]
```

### 2.1 Multi-Stage Build

The two-stage build keeps the final image smaller:
- **Builder stage:** Has cmake, build-essential, git. Compiles the project.
- **Runtime stage:** Only has the NVIDIA runtime + our binaries. No compiler, no source code.

### 2.2 Build Context

The `.dockerignore` file prevents unnecessary files from being copied into the build context:

```
# docker/.dockerignore
build/
checkpoints/
logs/
.git/
*.o
*.a
```

---

## 3. Docker Compose

For convenience, a docker-compose file that sets up volumes and GPU access:

```yaml
# docker/docker-compose.yml

version: '3.8'

services:
  train:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ../checkpoints:/app/checkpoints
      - ../config:/app/config
      - ../logs:/app/logs
    command: ["--mode", "train", "--config", "/app/config/default.yaml"]

  ui:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ../checkpoints:/app/checkpoints
      - ../logs:/app/logs
    ports:
      - "8080:8080"
    entrypoint: ["/usr/local/bin/pro_yams_ui"]
    command: [
      "--log_dir", "/app/logs",
      "--static_dir", "/app/static",
      "--port", "8080",
      "--checkpoint", "/app/checkpoints/latest.model"
    ]
```

### 3.1 Usage

```bash
# Build
docker compose -f docker/docker-compose.yml build

# Start training
docker compose -f docker/docker-compose.yml up train

# Start UI (in another terminal, while training runs)
docker compose -f docker/docker-compose.yml up ui

# Stop training (clean shutdown via SIGTERM)
docker compose -f docker/docker-compose.yml stop train
```

---

## 4. Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./checkpoints` | `/app/checkpoints` | Model checkpoints (persist across runs) |
| `./config` | `/app/config` | Configuration files |
| `./logs` | `/app/logs` | Training and eval logs |

All training artifacts persist on the host filesystem. Container can be destroyed and recreated without data loss.

---

## 5. Runtime Scripts

### 5.1 Training Script

```bash
# scripts/train.sh
#!/bin/bash
set -e

CONFIG=${1:-config/default.yaml}
CHECKPOINT=${2:-""}

ARGS="--mode train --config /app/config/$(basename $CONFIG)"
if [ -n "$CHECKPOINT" ]; then
    ARGS="$ARGS --checkpoint /app/checkpoints/$(basename $CHECKPOINT)"
fi

docker run --gpus all \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/config:/app/config \
    -v $(pwd)/logs:/app/logs \
    --rm \
    pro_yams_ai $ARGS
```

### 5.2 Evaluation Script

```bash
# scripts/eval.sh
#!/bin/bash
set -e

CHECKPOINT=${1:?"Usage: eval.sh <checkpoint_path> [num_games]"}
NUM_GAMES=${2:-200}

docker run --gpus all \
    -v $(pwd)/checkpoints:/app/checkpoints \
    --rm \
    pro_yams_ai \
    --mode eval \
    --checkpoint /app/checkpoints/$(basename $CHECKPOINT) \
    --eval_games $NUM_GAMES
```

### 5.3 UI Script

```bash
# scripts/ui.sh
#!/bin/bash
set -e

CHECKPOINT=${1:-""}
PORT=${2:-8080}

ARGS="--log_dir /app/logs --static_dir /app/static --port $PORT"
if [ -n "$CHECKPOINT" ]; then
    ARGS="$ARGS --checkpoint /app/checkpoints/$(basename $CHECKPOINT)"
fi

docker run --gpus all \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/logs:/app/logs \
    -p $PORT:$PORT \
    --rm \
    --entrypoint /usr/local/bin/pro_yams_ui \
    pro_yams_ai $ARGS
```

---

## 6. NVIDIA Container Toolkit

### 6.1 Prerequisites

The host machine needs the NVIDIA Container Toolkit installed for `--gpus all` to work:

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 6.2 Verification

```bash
docker run --gpus all nvcr.io/nvidia/pytorch:24.12-py3 nvidia-smi
```

Should display the RTX 5080 with driver info.

---

## 7. Quick Reference

```bash
# Build the Docker image
docker build -t pro_yams_ai -f docker/Dockerfile .

# Run training with default config
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    -v ./config:/app/config \
    -v ./logs:/app/logs \
    pro_yams_ai --mode train --config /app/config/default.yaml

# Resume training from checkpoint
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    -v ./config:/app/config \
    -v ./logs:/app/logs \
    pro_yams_ai --mode train --config /app/config/default.yaml \
    --checkpoint /app/checkpoints/checkpoint_step_5000

# Run evaluation
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    pro_yams_ai --mode eval \
    --checkpoint /app/checkpoints/checkpoint_step_5000 \
    --eval_games 500

# Run with custom config overrides (no YAML edit needed)
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    -v ./config:/app/config \
    -v ./logs:/app/logs \
    pro_yams_ai --mode train --config /app/config/default.yaml \
    --hidden_layers 2 --hidden_width 512 --learning_rate 0.0003

# Start UI
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    -v ./logs:/app/logs \
    -p 8080:8080 \
    --entrypoint /usr/local/bin/pro_yams_ui \
    pro_yams_ai --log_dir /app/logs --port 8080

# System info
docker run --gpus all pro_yams_ai --mode info
```

---

## 8. Definition of Done

This task is complete when:

1. `docker build` succeeds and produces a working image.
2. `docker run --gpus all ... --mode info` detects the GPU and prints system info.
3. `docker run --gpus all ... --mode train` starts training and produces checkpoints and logs in mounted volumes.
4. Training can be stopped with `docker stop` (SIGTERM) and resumes from checkpoint.
5. The UI binary runs in the container and is accessible at `http://localhost:8080`.
6. Volume mounts correctly persist checkpoints, logs, and configs across container restarts.
7. Docker compose file works for both train and UI services.
8. Helper scripts (`train.sh`, `eval.sh`, `ui.sh`) work correctly.
9. The runtime image contains only the necessary binaries and assets (no compiler, no source).
