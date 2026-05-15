# Pro Yams AI — Usage Guide

## Game variants

Pro Yams AI supports two game variants, selected per training run:

- **1v1** (two players) — the original variant. Self-play, training, checkpoints, and UI all fully supported.
- **2v2** (four players, team play) — Team 0 `{P0, P2}` vs Team 1 `{P1, P3}`, clockwise seating `A→B→C→D`. Rules in `pro_yams_2v2.md`; design and tensor layout in `2v2_variant.md`.

The variant is selected via the `game_variant: "1v1" | "2v2"` field in the YAML config (and matched on the model side via `ModelConfig::game_variant`). Checkpoints carry a variant tag — loading a 1v1 checkpoint into a 2v2 process (or vice versa) fails fast with a `std::runtime_error` rather than crashing on shape mismatch.

> **2v2 status:** the engine, scoring, duel math, heuristic, tensor, and TD-target machinery are all 2v2-ready. The self-play orchestration cascade (Worker, BatchManager, Coordinator, SelfPlayOrchestrator, TrainingLoop) is still being templated — see `2v2migration_remaining.md` for what's left and the execution order. Until that lands, 2v2 training cannot be launched end-to-end from the CLI.

## Prerequisites

- CMake 3.24+
- C++20 compiler (GCC 12+ or Clang 15+)
- CUDA toolkit (12.x recommended)
- libtorch (matching your CUDA version)

## Building

### Local build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/home/sorin/dev/libs/libtorch -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Build types: `Release` (default, `-O3 -march=native`), `Debug` (`-O0 -g`, ASan+UBSan), `RelWithDebInfo`.

To skip tests and benchmarks:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/home/sorin/dev/libs/libtorch -DBUILD_TESTS=OFF
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

#### 1v1 (default)

```bash
./build/pro_yams_ai --mode train --config config/mlp-256/seed.yaml
```

The 1v1 config sets `game_variant: "1v1"` (the default if omitted), `model.input_size: 986`, and the rest of the standard mlp-256 hyperparameters. Three phases — `seed.yaml` (heuristic bootstrap), `selfplay.yaml`, `selfplay2.yaml` — are chained by setting `--resume` on the previous phase's checkpoint.

#### 2v2

```bash
./build/pro_yams_ai --mode train --config config/mlp-512/seed.yaml
```

2v2 configs live in `config/mlp-512/` and must set:

```yaml
game_variant: "2v2"
training:
  model:
    input_size: 2126        # Yams2v2::kTensorSize
    hidden_width: 512       # wider model — see 2v2_variant.md §6
    learning_rate: 0.0003   # halved vs 1v1 — see 2v2_variant.md §6
```

The binary dispatches on `game_variant` at startup and instantiates either the `Yams1v1` or `Yams2v2` training pipeline. Both variants share the same binary; they never run concurrently in the same process.

#### Initialize from a pre-trained model

Start fresh training (step 0, new optimizer) using an existing checkpoint's weights as the starting point:

```bash
./build/pro_yams_ai --mode train --config config/my_config.yaml \
    --checkpoint checkpoints/test_td0_100k/checkpoint_step_50000
```

`--checkpoint` accepts a file stem or a directory (uses the latest checkpoint in the directory):

```bash
./build/pro_yams_ai --mode train --config config/my_config.yaml \
    --checkpoint checkpoints/test_td0_100k
```

#### Resume a cancelled training run

Resume from where training left off — restores the saved config, model weights, optimizer state, training step, temperature, and replay buffer:

```bash
./build/pro_yams_ai --mode train --resume checkpoints/test_td0_100k
```

The config is loaded automatically from `<checkpoint_dir>/config.yaml` (saved on every training start). You can still override individual settings:

```bash
./build/pro_yams_ai --mode train --resume checkpoints/test_td0_100k --num_steps 200000
```

`--resume` and `--checkpoint` cannot be used together.

`--resume` carries the variant forward — a checkpoint trained in 2v2 always resumes in 2v2. If the saved `config.yaml`'s `game_variant` differs from the checkpoint's stored variant tag (or from an explicit `--game_variant` override), training aborts with the variant-mismatch error.

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

Evaluate a 1v1 checkpoint against the heuristic bot:

```bash
./build/pro_yams_ai --mode eval --checkpoint checkpoints/step_5000.pt
```

Set the number of evaluation games (default 200):

```bash
./build/pro_yams_ai --mode eval --checkpoint checkpoints/step_5000.pt --eval_games 500
```

> **2v2 eval** is not yet implemented — the `evaluator` and `tournament` modules are still 1v1-only. Once templated (see `2v2migration_remaining.md` step 7.X2), the same flags work with `--game_variant 2v2`.

### Web UI (play against the AI)

#### 1v1

```bash
./build/src/ui/pro_yams_ui --checkpoint checkpoints/step_5000.pt
```

#### 2v2

```bash
./build/src/ui/pro_yams_ui --variant 2v2 --checkpoint checkpoints/mlp-512/step_5000.pt \
    --bot_versions V17,V17,V17
```

In 2v2 the human plays seat **P0** (top-left in the 2×2 grid); P1, P2, P3 are bots. `--bot_versions` is a comma-separated list of three heuristic versions, one per non-human seat in clockwise order `(P1, P2, P3)`. Default is `V2,V2,V2`. Pass a model checkpoint to swap in the NN for any seat (TODO once implemented).

The UI renders 2v2 boards as a 2×2 grid: Team 0 (P0, P2) shaded one colour on the left, Team 1 (P1, P3) shaded another on the right. Final scoring is shown as a Team 0 vs Team 1 margin.

Options (common to both variants):
- `--port 8080` — HTTP port (default 8080)
- `--static_dir ./src/ui/static` — path to frontend assets
- `--log_dir ./logs` — game log directory
- `--checkpoint <path>` — model checkpoint (omit for heuristic-only mode)
- `--variant 1v1|2v2` — selects the variant (default `1v1`)

Open `http://localhost:8080` in your browser.

## CLI Flags

All flags can override values from the YAML config file.

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | `info`, `config`, `train`, `eval`, `play` | `info` |
| `--config` | Path to YAML config file | (built-in defaults) |
| `--game_variant` | `1v1` or `2v2` (overrides YAML) | `1v1` |
| `--checkpoint` | Init model weights from checkpoint (stem or dir) | (none) |
| `--resume` | Resume training from checkpoint dir (loads saved config) | (none) |
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
