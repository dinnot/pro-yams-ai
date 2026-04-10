# Task 01: Project Scaffolding & CMake Build System

## Overview

Set up the complete project directory structure, CMake build system, test framework, and a minimal "hello world" proof that everything compiles, links, and runs — including libtorch GPU detection, Google Test, and Google Benchmark.

This task produces no game logic. Its sole purpose is to establish the build infrastructure that all subsequent tasks build on.

## Prerequisites

- Ubuntu 25.10 with GCC 13+ or Clang 17+
- NVIDIA GPU drivers installed
- CUDA toolkit installed (12.x)
- Pre-built libtorch (cxx11 ABI, CUDA-enabled) downloaded and extracted
- CMake 3.24+

---

## 1. Directory Structure

Create the following directory tree:

```
pro_yams_ai/
├── CMakeLists.txt                  # Root CMake file
├── src/
│   ├── engine/
│   │   ├── CMakeLists.txt          # Builds libengine (no ML deps)
│   │   └── placeholder.cc          # Minimal source so library compiles
│   ├── solver/
│   │   ├── CMakeLists.txt          # Builds libsolver, depends on libengine
│   │   └── placeholder.cc
│   ├── model/
│   │   ├── CMakeLists.txt          # Builds libmodel, depends on libtorch
│   │   └── placeholder.cc
│   ├── heuristic/
│   │   ├── CMakeLists.txt          # Builds libheuristic, depends on libengine + libsolver
│   │   └── placeholder.cc
│   ├── self_play/
│   │   ├── CMakeLists.txt          # Builds libselfplay, depends on libengine + libsolver + libmodel
│   │   └── placeholder.cc
│   ├── training/
│   │   ├── CMakeLists.txt          # Builds libtraining, depends on libmodel + libselfplay
│   │   └── placeholder.cc
│   ├── eval/
│   │   ├── CMakeLists.txt          # Builds libeval, depends on libengine + libsolver + libmodel + libheuristic
│   │   └── placeholder.cc
│   ├── ui/
│   │   ├── CMakeLists.txt          # Builds libui, depends on libengine + libsolver + libmodel + libheuristic
│   │   └── placeholder.cc
│   ├── config/
│   │   ├── CMakeLists.txt          # Builds libconfig (YAML/JSON parsing)
│   │   └── placeholder.cc
│   └── main.cpp                    # Entry point
├── tests/
│   ├── CMakeLists.txt              # Test root, pulls in gtest/gbench
│   ├── engine/
│   │   ├── CMakeLists.txt
│   │   └── engine_smoke_test.cc    # Minimal test proving gtest works with libengine
│   ├── solver/
│   │   └── CMakeLists.txt
│   ├── model/
│   │   └── CMakeLists.txt
│   └── benchmarks/
│       ├── CMakeLists.txt
│       └── engine_bench.cc         # Minimal benchmark proving gbench works
├── config/
│   └── default.yaml                # Skeleton config file
├── docker/
│   └── Dockerfile                  # Placeholder, completed in Task 16
├── scripts/
│   └── README.md                   # Placeholder for helper scripts
├── checkpoints/
│   └── .gitkeep
└── logs/
    └── .gitkeep
```

Each `placeholder.cc` file contains a minimal compiled symbol (e.g., a namespace with a single dummy function) so the library target is valid. These files are replaced with real implementations in subsequent tasks.

---

## 2. Root CMakeLists.txt

The root CMake file handles:

### 2.1 Project Declaration

```cmake
cmake_minimum_required(VERSION 3.24)
project(pro_yams_ai LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

Use C++20 for `std::format`, structured bindings, concepts (if useful later), and other modern features.

### 2.2 Build Type Configuration

```cmake
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native -DNDEBUG")
```

Three build types:
- **Release:** Maximum optimization, targets the exact Ryzen 9900X architecture. `NDEBUG` disables assertions.
- **Debug:** No optimization, address/UB sanitizers enabled, assertions active.
- **RelWithDebInfo:** Optimized with debug symbols for profiling (perf, vtune, etc.).

### 2.3 libtorch Integration

```cmake
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
```

The user sets `CMAKE_PREFIX_PATH` to the libtorch installation directory when invoking CMake:
```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

### 2.4 Options

```cmake
option(BUILD_TESTS "Build unit tests and benchmarks" ON)
```

Tests are built by default during development but can be disabled for production/Docker builds.

### 2.5 Subdirectories

```cmake
add_subdirectory(src/engine)
add_subdirectory(src/solver)
add_subdirectory(src/model)
add_subdirectory(src/heuristic)
add_subdirectory(src/self_play)
add_subdirectory(src/training)
add_subdirectory(src/eval)
add_subdirectory(src/ui)
add_subdirectory(src/config)

# Main executable
add_executable(pro_yams_ai src/main.cpp)
target_link_libraries(pro_yams_ai
    PRIVATE engine solver model heuristic self_play training eval ui config
    PRIVATE ${TORCH_LIBRARIES}
)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

---

## 3. Library CMakeLists.txt Files

Each library follows the same pattern. The key is the dependency graph — this enforces our architectural boundaries at compile time.

### 3.1 Dependency Graph

```
config          (depends on: yaml-cpp)
engine          (depends on: nothing — no ML, no external deps)
solver          (depends on: engine)
heuristic       (depends on: engine, solver)
model           (depends on: libtorch, config)
self_play       (depends on: engine, solver, model)
training        (depends on: model, self_play, config)
eval            (depends on: engine, solver, model, heuristic)
ui              (depends on: engine, solver, model, heuristic)
```

**Critical boundary:** `engine`, `solver`, and `heuristic` never link against libtorch. If someone accidentally includes a torch header in the engine, the build fails.

### 3.2 Example: Engine Library

```cmake
# src/engine/CMakeLists.txt
add_library(engine STATIC
    placeholder.cc
    # Future files added here as they are implemented
)
target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/src)
```

Using `PUBLIC` include directories so that any library depending on `engine` can `#include "engine/some_header.h"`.

### 3.3 Example: Model Library (with libtorch)

```cmake
# src/model/CMakeLists.txt
add_library(model STATIC
    placeholder.cc
)
target_include_directories(model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(model PUBLIC ${TORCH_LIBRARIES})
```

Only libraries in the ML path (`model`, `self_play`, `training`) link against Torch.

### 3.4 Example: Solver Library

```cmake
# src/solver/CMakeLists.txt
add_library(solver STATIC
    placeholder.cc
)
target_include_directories(solver PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(solver PUBLIC engine)
```

---

## 4. Test Framework Setup

### 4.1 Google Test & Google Benchmark via FetchContent

In `tests/CMakeLists.txt`:

```cmake
include(FetchContent)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Google Benchmark
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_subdirectory(engine)
add_subdirectory(solver)
add_subdirectory(model)
add_subdirectory(benchmarks)
```

`FetchContent` downloads and builds gtest/gbench automatically at configure time. No manual installation required.

### 4.2 Example Test Target

```cmake
# tests/engine/CMakeLists.txt
add_executable(engine_tests engine_smoke_test.cc)
target_link_libraries(engine_tests PRIVATE engine gtest_main)
add_test(NAME engine_tests COMMAND engine_tests)
```

Using `gtest_main` provides the `main()` function automatically — no need to write one per test file.

### 4.3 Example Benchmark Target

```cmake
# tests/benchmarks/CMakeLists.txt
add_executable(engine_bench engine_bench.cc)
target_link_libraries(engine_bench PRIVATE engine benchmark::benchmark_main)
```

### 4.4 Smoke Test File

`tests/engine/engine_smoke_test.cc`:
```cpp
#include <gtest/gtest.h>

// Proves gtest is working and can link against libengine
TEST(EngineSmoke, Compiles) {
    EXPECT_EQ(1 + 1, 2);
}
```

### 4.5 Smoke Benchmark File

`tests/benchmarks/engine_bench.cc`:
```cpp
#include <benchmark/benchmark.h>

// Proves google benchmark is working
static void BM_Placeholder(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(1 + 1);
    }
}
BENCHMARK(BM_Placeholder);
```

---

## 5. Main Entry Point

`src/main.cpp` — a minimal entry point that parses a `--mode` argument and proves libtorch + CUDA are working:

```cpp
#include <iostream>
#include <string>
#include <torch/torch.h>

int main(int argc, char* argv[]) {
    std::string mode = "info";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
            mode = argv[i + 1];
        }
    }

    if (mode == "info") {
        std::cout << "Pro Yams AI\n";
        std::cout << "CUDA available: " << (torch::cuda::is_available() ? "yes" : "no") << "\n";
        std::cout << "CUDA devices: " << torch::cuda::device_count() << "\n";
        if (torch::cuda::is_available()) {
            auto t = torch::randn({3, 3}, torch::kCUDA);
            std::cout << "GPU tensor test: OK (" << t.sizes() << ")\n";
        }
    } else if (mode == "train") {
        std::cout << "Training mode (not yet implemented)\n";
    } else if (mode == "eval") {
        std::cout << "Evaluation mode (not yet implemented)\n";
    } else if (mode == "play") {
        std::cout << "Play mode (not yet implemented)\n";
    } else if (mode == "benchmark") {
        std::cout << "Benchmark mode (not yet implemented)\n";
    } else if (mode == "generate") {
        std::cout << "Generate heuristic data mode (not yet implemented)\n";
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        std::cerr << "Available: info, train, eval, play, benchmark, generate\n";
        return 1;
    }

    return 0;
}
```

---

## 6. Skeleton Config File

`config/default.yaml`:

```yaml
# Pro Yams AI — Default Configuration
# All values are overridable via config file passed with --config

# Network architecture
network:
  hidden_layers: 3
  hidden_width: 256
  learning_rate: 0.001
  optimizer: adam

# Training
training:
  td_mode: td0
  td_lambda: 0.7
  batch_size: 256
  replay_buffer_size: 50000
  model_swap_interval: 100

# Self-play
self_play:
  num_concurrent_games: 512
  inference_batch_size: 512
  inference_timeout_ms: 5

# Exploration
exploration:
  placement_temperature: 1.0
  hold_temperature: 1.0
  temperature_decay: 0.999
  min_temperature: 0.05

# Bootstrapping
bootstrapping:
  bootstrap_games: 10000
  epsilon_start: 0.5
  epsilon_decay: 0.999
  epsilon_min: 0.0

# Evaluation
evaluation:
  eval_interval: 1000
  eval_games: 200

# Checkpointing
checkpointing:
  checkpoint_interval: 5000
  max_checkpoints: 5
  checkpoint_dir: "./checkpoints"

# Logging
logging:
  log_dir: "./logs"
  log_interval: 100
```

This file is not parsed yet (config loading is Task 14), but its presence establishes the format and documents all hyperparameters from the start.

---

## 7. Config Library Placeholder

The config library will eventually parse YAML using yaml-cpp. For now, it's a placeholder, but we set up the yaml-cpp dependency so it's ready:

```cmake
# src/config/CMakeLists.txt
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

add_library(config STATIC placeholder.cc)
target_include_directories(config PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(config PUBLIC yaml-cpp::yaml-cpp)
```

---

## 8. Build & Verify Instructions

### 8.1 First Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 8.2 Verify GPU

```bash
./pro_yams_ai --mode info
# Expected output:
# Pro Yams AI
# CUDA available: yes
# CUDA devices: 1
# GPU tensor test: OK ([3, 3])
```

### 8.3 Run Tests

```bash
ctest --output-on-failure
# Expected: engine_tests passes (1 test)
```

### 8.4 Run Benchmarks

```bash
./tests/benchmarks/engine_bench
# Expected: BM_Placeholder runs and reports timing
```

### 8.5 Debug Build

```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
# Address sanitizer + UB sanitizer active
```

---

## 9. Definition of Done

This task is complete when:

1. The directory structure exists as specified above.
2. `cmake` configures without errors (finds libtorch, gtest, gbench, yaml-cpp).
3. `make` compiles the main executable and all placeholder libraries without errors or warnings.
4. `./pro_yams_ai --mode info` runs and confirms CUDA availability and GPU tensor operations.
5. `ctest` runs the smoke test and it passes.
6. The benchmark executable runs and produces output.
7. Debug build compiles and runs with sanitizers active.
8. All placeholder libraries respect the dependency graph (engine has no libtorch linkage, etc.).

## 10. User Environment Specifics
* **LibTorch Path:** The user has downloaded the CXX11 ABI version of libtorch to `/home/sorin/dev/libs/libtorch`. 
* **CMake Command:** When configuring CMake in Step 8.1, you MUST use this exact command:
  `cmake .. -DCMAKE_PREFIX_PATH=/home/sorin/dev/libs/libtorch -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13`