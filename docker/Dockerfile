# Pro Yams AI — Docker build
#
# Multi-stage build:
#   1. Builder: CUDA 12.8 devel + nightly libtorch + compile
#   2. Runtime: CUDA 12.8 runtime + binaries only
#
# Standard PyTorch containers do not yet support RTX 5080 (sm_120).
# We use the CUDA 12.8 base and pull nightly libtorch directly.

# --- Build stage ---
FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    wget \
    unzip \
    ca-certificates \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Download nightly libtorch (CUDA 12.8).
WORKDIR /opt
RUN wget -q https://download.pytorch.org/libtorch/nightly/cu128/libtorch-shared-with-deps-latest.zip && \
    unzip -q libtorch-shared-with-deps-latest.zip && \
    rm libtorch-shared-with-deps-latest.zip && \
    echo "void dummy() {}" > dummy.c && \
    gcc -shared -o /opt/libtorch/lib/libtorch_nvshmem.so dummy.c && \
    rm dummy.c

# Copy source.
COPY . /app
WORKDIR /app/build

# Build both binaries in Release mode (tests disabled for smaller image).
RUN cmake .. \
    -DCMAKE_PREFIX_PATH=/opt/libtorch \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--unresolved-symbols=ignore-in-shared-libs" \
    -DBUILD_TESTS=OFF \
    && make -j$(nproc)

# --- Runtime stage ---
FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime

# OpenMP runtime and cuDNN required by libtorch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# libtorch shared libraries needed at runtime.
COPY --from=builder /opt/libtorch /opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Binaries.
COPY --from=builder /app/build/pro_yams_ai /usr/local/bin/pro_yams_ai
COPY --from=builder /app/build/src/ui/pro_yams_ui /usr/local/bin/pro_yams_ui

# Default config and UI static assets.
COPY config/default.yaml /app/config/default.yaml
COPY src/ui/static /app/static

# Volume mount points.
RUN mkdir -p /app/checkpoints /app/logs

WORKDIR /app

ENTRYPOINT ["/usr/local/bin/pro_yams_ai"]
CMD ["--mode", "train", "--config", "/app/config/default.yaml"]