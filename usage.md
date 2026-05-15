# Pro Yams AI — Usage Guide

## Game variants

Pro Yams AI supports two game variants, selected per training run:

- **1v1** (two players) — the original variant. Self-play, training, checkpoints, evaluation, tournament, and UI fully supported.
- **2v2** (four players, team play) — Team 0 `{P0, P2}` vs Team 1 `{P1, P3}`, clockwise seating `A→B→C→D`. Same self-play / training / eval / tournament / UI surface area as 1v1. Rules in `pro_yams_2v2.md`; design and tensor layout in [`2v2_variant.md`](2v2_variant.md).

The variant is selected via the `game_variant: "1v1" | "2v2"` field in the YAML config (and matched on the model side via `ModelConfig::game_variant`). Checkpoints carry a variant tag — loading a 1v1 checkpoint into a 2v2 process (or vice versa) fails fast with a `std::runtime_error` rather than crashing on shape mismatch.

The whole binary compiles for both variants and dispatches at startup, so a single build supports both. `eval` mode auto-detects the variant from the checkpoint's tag — no `--game_variant` flag needed when evaluating.

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

This produces three binaries:
- `build/pro_yams_ai` — main executable (training, eval, info)
- `build/src/ui/pro_yams_ui` — full web UI server (game, dashboard, tournament)
- `build/src/ui/pro_yams_play` — slim "play vs the AI" web app

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

The output reflects all defaults, YAML overrides, and CLI overrides — useful for verifying that a `game_variant: "2v2"` config resolves to the right `input_size` (`2126`) and friends.

### Training

#### 1v1 (default)

```bash
./build/pro_yams_ai --mode train --config config/mlp-256/seed.yaml
```

The 1v1 config defaults `game_variant: "1v1"` and `model.input_size: 986`. Three phases — `seed.yaml` (heuristic bootstrap), `selfplay.yaml`, `selfplay2.yaml` — are chained by setting `--resume` on the previous phase's checkpoint.

#### 2v2

```bash
./build/pro_yams_ai --mode train --config config/mlp-512/seed.yaml
```

2v2 configs live in `config/mlp-512/` and must set at least:

```yaml
training:
  game_variant: "2v2"
  model:
    input_size: 2126        # Yams2v2::kTensorSize
    hidden_width: 512       # wider model — see 2v2_variant.md §6
    learning_rate: 0.0003   # slightly lower — see 2v2_variant.md §6
```

`input_size` will be auto-corrected to 2126 if `game_variant: "2v2"` is set and `input_size` is left at the 1v1 default. The validator will reject an explicit mismatched pair (e.g. `game_variant: "2v2"` with `input_size: 986`).

The binary dispatches on `game_variant` at startup and instantiates either the `Yams1v1` or `Yams2v2` training pipeline. Both variants share the same binary; they never run concurrently in the same process.

You can also override the variant from the CLI without touching YAML:

```bash
./build/pro_yams_ai --mode train --config config/some_config.yaml --game_variant 2v2
```

#### Initialize from a pre-trained model

Start fresh training (step 0, new optimizer) using an existing checkpoint's weights as the starting point:

```bash
./build/pro_yams_ai --mode train --config config/my_config.yaml \
    --checkpoint checkpoints/test_td0_100k/checkpoint_step_50000
```

`--checkpoint` accepts a file stem or a directory (uses the latest checkpoint in the directory). The checkpoint's variant tag must match the active variant; otherwise `load_checkpoint` throws with a clear error.

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

Evaluate a checkpoint against the heuristic bot:

```bash
./build/pro_yams_ai --mode eval --checkpoint checkpoints/step_5000
```

`mode eval` reads the variant tag straight from the checkpoint and dispatches to the matching pipeline — `Eval variant: 1v1 (Yams1v1)` or `Eval variant: 2v2 (Yams2v2)` is printed on startup. No `--game_variant` is required.

Set the number of evaluation games (default 200):

```bash
./build/pro_yams_ai --mode eval --checkpoint checkpoints/step_5000 --eval_games 500
```

In 2v2 mode the output rows are labelled:

```
  NN as Team 0 win rate: …    (NN drives both seats of Team 0)
  NN as Team 1 win rate: …    (NN drives both seats of Team 1)
```

Internally, `EvalResult::nn_wins_as_p0` counts games where the NN played Team 0, and `nn_wins_as_p1` counts the other team.

### Web UI

The web UI binary (`pro_yams_ui`) supports both variants. Pick one with `--variant`:

#### 1v1

```bash
./build/src/ui/pro_yams_ui --checkpoint checkpoints/step_5000
```

#### 2v2

```bash
./build/src/ui/pro_yams_ui --variant 2v2 --checkpoint checkpoints/mlp-512/step_5000
```

In 2v2 the frontend automatically switches into a 2×2 grid layout when the server reports `game_variant: "2v2"` in the JSON state:

```
+----------+----------+
| P0 (T0)  | P1 (T1)  |
+----------+----------+
| P2 (T0)  | P3 (T1)  |
+----------+----------+
```

Team 0 (P0, P2) is shaded one colour on the left; Team 1 (P1, P3) the other on the right. Final scoring is shown as a Team 0 vs Team 1 margin. The `New Game` HTML form currently exposes selectors for P0 and P1; P2 and P3 default to Heuristic V2. The JSON API accepts `player0..player3` fields for full programmatic control.

The tournament tab works in both variants — in 2v2 each matchup pits one bot driving both seats of one team against another bot driving the other team's seats.

Common options:
- `--port 8080` — HTTP port (default 8080)
- `--static_dir ./src/ui/static` — path to frontend assets
- `--log_dir ./logs` — game log directory
- `--checkpoints_dir <dir>` — root directory the tournament tab scans for checkpoints (default: parent of `--checkpoint`)
- `--checkpoint <path>` — model checkpoint (omit for heuristic-only mode)
- `--variant 1v1|2v2` — selects the variant (default `1v1`)

Open `http://localhost:8080` in your browser. NN checkpoints loaded into a tournament are rejected at start if their `input_size` doesn't match the server's active variant.

#### Standalone play app

```bash
./build/src/ui/pro_yams_play --checkpoint checkpoints/step_5000 [--variant 2v2] \
    --static_dir ./src/ui/play_static --port 8090
```

Same UI but with the slim mobile-first frontend and tournament endpoints disabled.

## CLI Flags

All flags can override values from the YAML config file.

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | `info`, `config`, `train`, `eval`, `play` | `info` |
| `--config` | Path to YAML config file | (built-in defaults) |
| `--game_variant` | `1v1` or `2v2` (overrides YAML; ignored by `--mode eval`, which auto-detects) | `1v1` |
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
| `--past_opponent_probability` | Probability a recycled game uses an older checkpoint | 0.0 |

## Tests

Run all unit tests:

```bash
cd build && ctest --output-on-failure
```

The 2v2 path is covered by dedicated tests alongside the existing 1v1 ones:
- `tests/engine/duel_2v2_test.cc`, `tests/engine/golden_rule_teammate_test.cc`
- `tests/engine/tensor_rotation_test.cc`, `tests/engine/tensor_team_swap_test.cc`
- `tests/heuristic/heuristic_2v2_test.cc`
- `tests/self_play/trajectory_2v2_test.cc`
- `tests/training/training_loop_2v2_test.cc` — end-to-end smoke for `TrainingLoopT<Yams2v2>`

Run a specific test suite:

```bash
./build/tests/engine/engine_tests
./build/tests/solver/solver_tests
./build/tests/model/model_tests
./build/tests/heuristic/heuristic_tests
./build/tests/self_play/self_play_tests
./build/tests/training/training_tests
./build/tests/eval/eval_tests
./build/tests/config/config_tests
./build/tests/ui/ui_tests
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
  engine/       Game rules, board state, scoring, duels (templated on Yams1v1/Yams2v2)
  solver/       Precomputed probability tables, expectimax solver
  heuristic/    Heuristic evaluation bot (templated)
  model/        Neural network (ProYamsNet), trainer, checkpointing
  self_play/    Templated self-play workers, inference batching, data extraction
  training/     Templated replay buffer, training loop, resume logic
  eval/         NN-vs-heuristic evaluation + multi-agent tournaments (templated)
  config/       YAML + CLI config loading, validation, printing
  ui/           HTTP server, game sessions, JSON serialization, static frontend (templated)
config/         YAML configuration files (`mlp-256/` = 1v1, `mlp-512/` = 2v2)
checkpoints/    Saved model checkpoints (each carries a game_variant tag)
logs/           Training logs and game logs
tests/          Unit tests and benchmarks (Google Test / Google Benchmark)
```
