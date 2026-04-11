# Pro Yams AI — Usage Guide

## Prerequisites

- CMake 3.24+
- C++20 compiler (GCC 12+ or Clang 15+)
- CUDA toolkit (12.x recommended)
- libtorch (matching your CUDA version)

## Building

### Local build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Build types: `Release` (default, `-O3 -march=native`), `Debug` (`-O0 -g`, ASan+UBSan), `RelWithDebInfo`.

To skip tests and benchmarks:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch -DBUILD_TESTS=OFF
```

This produces two binaries:
- `build/pro_yams_ai` — main executable (training, eval, info)
- `build/src/ui/pro_yams_ui` — web UI server


## Running

### Check system info

```bash
./build/pro_yams_ai --mode info
```

Prints CUDA availability, device count, and runs a quick GPU tensor test.

### Print resolved configuration

```bash
./build/pro_yams_ai --mode config --config config/default.yaml
```

### Training

```bash
./build/pro_yams_ai --mode train --config config/default.yaml
```

Resume from a checkpoint:

```bash
./build/pro_yams_ai --mode train --config config/default.yaml --checkpoint checkpoints/step_5000.pt
```

Training supports graceful shutdown with `Ctrl+C` (SIGINT/SIGTERM) — it finishes the current step and saves a checkpoint.

### Training Debug Mode

Run training with instrumentation to diagnose convergence issues (e.g., "Softmax Scratch Avalanche").

```bash
./build/pro_yams_ai --mode train --config config/default.yaml --debug_mode 1
```

Debug mode enables:
- **Solver Logging:** Decisions for the first active game are logged to `logs/debug_game_0.log`.
- **Heuristic Bootstrapping:** Adds a decaying heuristic weight (initial: 1.0) to the win probability to guide the network during early training.
- **Tensor Health:** Monitors for NaN/Inf values in input features and gradients.

### Evaluation

Evaluate a checkpoint against the heuristic bot:

```bash
./build/pro_yams_ai --mode eval --checkpoint checkpoints/step_5000.pt
```

Set the number of evaluation games (default 200):

```bash
./build/pro_yams_ai --mode eval --checkpoint checkpoints/step_5000.pt --eval_games 500
```

### Web UI (play against the AI)

```bash
./build/src/ui/pro_yams_ui --checkpoint checkpoints/step_5000.pt
```

Options:
- `--port 8080` — HTTP port (default 8080)
- `--static_dir ./src/ui/static` — path to frontend assets
- `--log_dir ./logs` — game log directory
- `--checkpoint <path>` — model checkpoint (omit for heuristic-only mode)

Open `http://localhost:8080` in your browser.

## CLI Flags

All flags can override values from the YAML config file.

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | `info`, `config`, `train`, `eval`, `play` | `info` |
| `--config` | Path to YAML config file | (built-in defaults) |
| `--checkpoint` | Path to model checkpoint | (none) |
| `--seed` | RNG seed | 42 |
| `--num_steps` | Training gradient steps | 100000 |
| `--learning_rate` | Model learning rate | 0.001 |
| `--hidden_layers` | Number of hidden layers | 3 |
| `--hidden_width` | Hidden layer width | 256 |
| `--td_mode` | TD target: `mc`, `td0`, `tdlambda` | `mc` |
| `--batch_size` | Training batch size | 512 |
| `--replay_buffer_size` | Replay buffer capacity | 2000000 |
| `--placement_temperature` | Initial softmax temperature | 1.0 |
| `--num_workers` | Self-play worker threads | 16 |
| `--num_games` | Self-play games per batch | 512 |
| `--eval_games` | Evaluation games per round | 200 |
| `--eval_interval` | Steps between evaluations (0=off) | 1000 |
| `--debug_mode` | Enable debug logging & bootstrapping (0/1) | 0 |
| `--initial_heuristic_weight` | Heuristic bootstrapping strength | 1.0 |
| `--heuristic_decay_steps` | Steps to decay heuristic to zero | 50000 |

## Tests

Run all unit tests:

```bash
cd build && ctest --output-on-failure
```

Run a specific test suite:

```bash
./build/tests/engine/engine_test
./build/tests/solver/solver_test
./build/tests/model/model_test
./build/tests/heuristic/heuristic_test
./build/tests/self_play/self_play_test
./build/tests/training/training_test
./build/tests/eval/eval_test
./build/tests/config/config_test
./build/tests/ui/ui_test
```

## Benchmarks

Run all benchmarks:

```bash
for bench in build/tests/benchmarks/*_bench; do $bench; done
```

Available benchmarks: `engine_bench`, `solver_bench`, `tensor_bench`, `model_bench`, `self_play_bench`, `training_bench`, `eval_bench`.

Run a single benchmark:

```bash
./build/tests/benchmarks/model_bench
```

Filter specific benchmark cases with `--benchmark_filter`:

```bash
./build/tests/benchmarks/model_bench --benchmark_filter=BM_ForwardPass
```


## Project Layout

```
src/
  engine/       Game rules, board state, scoring, duels
  solver/       Precomputed probability tables, heuristic solver
  heuristic/    Heuristic evaluation bot
  model/        Neural network (ProYamsNet), trainer, checkpointing
  self_play/    Self-play workers, inference batching, data extraction
  training/     Replay buffer, training loop, resume logic
  eval/         NN vs heuristic evaluation
  config/       YAML + CLI config loading, validation, printing
  ui/           HTTP server, game sessions, JSON serialization, static frontend
config/         YAML configuration files
checkpoints/    Saved model checkpoints
logs/           Training logs and game logs
tests/          Unit tests and benchmarks (Google Test / Google Benchmark)
```
