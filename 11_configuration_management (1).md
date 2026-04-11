# Task 11: Configuration Management

## Overview

Implement YAML configuration loading, command-line argument parsing, and the fully wired `main.cpp` entry point that launches the appropriate mode (train, eval, play, info, benchmark). All hyperparameters are configurable without recompilation.

## Prerequisites

- Task 01 completed (yaml-cpp dependency set up)
- Tasks 02–10 completed (all component config structs defined)

---

## 1. Master Configuration Struct

```cpp
// src/config/app_config.h

struct AppConfig {
    ModelConfig model;
    TrainingConfig training;
    SelfPlayConfig self_play;
    SolverConfig solver;

    // Runtime options (from command line, not YAML)
    std::string mode = "info";           // train, eval, play, info, benchmark
    std::string config_path = "";        // Path to YAML config file
    std::string checkpoint_path = "";    // Path to checkpoint to resume from
    uint64_t seed = 42;                  // Master random seed
};
```

---

## 2. YAML Configuration File

### 2.1 Default Configuration

`config/default.yaml`:

```yaml
# Pro Yams AI — Default Configuration

# Network architecture
network:
  hidden_layers: 3
  hidden_width: 256
  learning_rate: 0.001

# Training
training:
  td_mode: mc                # mc recommended (targets never go stale in replay buffer)
  td_lambda: 0.7
  batch_size: 256
  replay_buffer_size: 2000000 # ~6.5GB. Large buffer prevents policy oscillation.
  min_samples_before_training: 1000
  model_swap_interval: 100

# Self-play
self_play:
  num_workers: 16
  num_games: 512
  max_inference_batch: 1024
  min_games_per_batch: 2
  batch_timeout_ms: 5

# Exploration
exploration:
  placement_temperature: 1.0
  hold_temperature: 1.0
  temperature_decay: 0.999
  min_temperature: 0.05

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

# Random seed
seed: 42
```

### 2.2 Loading

```cpp
// src/config/config_loader.h

/// Load configuration from a YAML file.
/// Missing fields use default values.
/// @param path Path to YAML file
/// @return Populated AppConfig
AppConfig load_config_from_yaml(const std::string& path);

/// Apply command-line overrides to an existing config.
/// Supports flat key=value overrides like --learning_rate 0.0003
/// @param config Config to modify in-place
/// @param argc Argument count
/// @param argv Argument values
void apply_cli_overrides(AppConfig& config, int argc, char* argv[]);

/// Load config with full pipeline: defaults → YAML → CLI overrides.
/// @param argc Argument count
/// @param argv Argument values
/// @return Final configuration
AppConfig load_config(int argc, char* argv[]);
```

### 2.3 YAML Parsing Implementation

```cpp
AppConfig load_config_from_yaml(const std::string& path) {
    AppConfig config;  // Starts with all defaults

    YAML::Node root = YAML::LoadFile(path);

    // Network
    if (auto n = root["network"]) {
        if (n["hidden_layers"]) config.model.hidden_layers = n["hidden_layers"].as<int>();
        if (n["hidden_width"]) config.model.hidden_width = n["hidden_width"].as<int>();
        if (n["learning_rate"]) config.model.learning_rate = n["learning_rate"].as<double>();
    }

    // Training
    if (auto t = root["training"]) {
        if (t["td_mode"]) {
            std::string mode = t["td_mode"].as<std::string>();
            if (mode == "td0") config.training.td_mode = TDMode::kTD0;
            else if (mode == "td_lambda") config.training.td_mode = TDMode::kTDLambda;
            else if (mode == "mc") config.training.td_mode = TDMode::kMC;
        }
        if (t["td_lambda"]) config.training.td_lambda = t["td_lambda"].as<double>();
        if (t["batch_size"]) config.training.training_batch_size = t["batch_size"].as<int>();
        if (t["replay_buffer_size"]) config.training.replay_buffer_size = t["replay_buffer_size"].as<int>();
        if (t["min_samples_before_training"]) config.training.min_samples_before_training = t["min_samples_before_training"].as<int>();
        if (t["model_swap_interval"]) config.training.model_swap_interval = t["model_swap_interval"].as<int>();
    }

    // Self-play
    if (auto s = root["self_play"]) {
        if (s["num_workers"]) config.self_play.num_workers = s["num_workers"].as<int>();
        if (s["num_games"]) config.self_play.num_games = s["num_games"].as<int>();
        if (s["max_inference_batch"]) config.self_play.max_inference_batch = s["max_inference_batch"].as<int>();
        if (s["min_games_per_batch"]) config.self_play.min_games_per_batch = s["min_games_per_batch"].as<int>();
        if (s["batch_timeout_ms"]) config.self_play.batch_timeout_ms = s["batch_timeout_ms"].as<int>();
    }

    // Exploration
    if (auto e = root["exploration"]) {
        if (e["placement_temperature"]) config.training.placement_temperature_start = e["placement_temperature"].as<double>();
        if (e["hold_temperature"]) config.training.hold_temperature_start = e["hold_temperature"].as<double>();
        if (e["temperature_decay"]) config.training.temperature_decay = e["temperature_decay"].as<double>();
        if (e["min_temperature"]) config.training.min_temperature = e["min_temperature"].as<double>();
    }

    // Evaluation
    if (auto ev = root["evaluation"]) {
        if (ev["eval_interval"]) config.training.eval_interval = ev["eval_interval"].as<int>();
        if (ev["eval_games"]) config.training.eval_games = ev["eval_games"].as<int>();
    }

    // Checkpointing
    if (auto c = root["checkpointing"]) {
        if (c["checkpoint_interval"]) config.training.checkpoint_interval = c["checkpoint_interval"].as<int>();
        if (c["max_checkpoints"]) config.training.max_checkpoints = c["max_checkpoints"].as<int>();
        if (c["checkpoint_dir"]) config.training.checkpoint_dir = c["checkpoint_dir"].as<std::string>();
    }

    // Logging
    if (auto l = root["logging"]) {
        if (l["log_dir"]) config.training.log_dir = l["log_dir"].as<std::string>();
        if (l["log_interval"]) config.training.log_interval = l["log_interval"].as<int>();
    }

    // Seed
    if (root["seed"]) config.seed = root["seed"].as<uint64_t>();

    return config;
}
```

### 2.4 CLI Override Parsing

Supports `--key value` pairs that override YAML values. Also handles `--mode`, `--config`, and `--checkpoint` as special arguments.

```cpp
void apply_cli_overrides(AppConfig& config, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            config.mode = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config.config_path = argv[++i];
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            config.checkpoint_path = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoull(argv[++i]);
        }
        // Network overrides
        else if (arg == "--hidden_layers" && i + 1 < argc) {
            config.model.hidden_layers = std::stoi(argv[++i]);
        } else if (arg == "--hidden_width" && i + 1 < argc) {
            config.model.hidden_width = std::stoi(argv[++i]);
        } else if (arg == "--learning_rate" && i + 1 < argc) {
            config.model.learning_rate = std::stod(argv[++i]);
        }
        // Training overrides
        else if (arg == "--td_mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "td0") config.training.td_mode = TDMode::kTD0;
            else if (mode == "td_lambda") config.training.td_mode = TDMode::kTDLambda;
            else if (mode == "mc") config.training.td_mode = TDMode::kMC;
        } else if (arg == "--batch_size" && i + 1 < argc) {
            config.training.training_batch_size = std::stoi(argv[++i]);
        } else if (arg == "--replay_buffer_size" && i + 1 < argc) {
            config.training.replay_buffer_size = std::stoi(argv[++i]);
        }
        // Exploration overrides
        else if (arg == "--placement_temperature" && i + 1 < argc) {
            config.training.placement_temperature_start = std::stod(argv[++i]);
        } else if (arg == "--hold_temperature" && i + 1 < argc) {
            config.training.hold_temperature_start = std::stod(argv[++i]);
        } else if (arg == "--temperature_decay" && i + 1 < argc) {
            config.training.temperature_decay = std::stod(argv[++i]);
        }
        // Self-play overrides
        else if (arg == "--num_workers" && i + 1 < argc) {
            config.self_play.num_workers = std::stoi(argv[++i]);
        } else if (arg == "--num_games" && i + 1 < argc) {
            config.self_play.num_games = std::stoi(argv[++i]);
        }
        // Evaluation overrides
        else if (arg == "--eval_games" && i + 1 < argc) {
            config.training.eval_games = std::stoi(argv[++i]);
        } else if (arg == "--eval_interval" && i + 1 < argc) {
            config.training.eval_interval = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
        }
    }
}
```

### 2.5 Full Config Loading Pipeline

```cpp
AppConfig load_config(int argc, char* argv[]) {
    AppConfig config;  // Defaults

    // First pass: find --config path
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config.config_path = argv[i + 1];
            break;
        }
    }

    // Load YAML if specified
    if (!config.config_path.empty()) {
        config = load_config_from_yaml(config.config_path);
    }

    // Apply CLI overrides (takes precedence over YAML)
    apply_cli_overrides(config, argc, argv);

    return config;
}
```

---

## 3. Configuration Validation

```cpp
/// Validate configuration values are within acceptable ranges.
/// Prints warnings for questionable values, returns false for invalid ones.
bool validate_config(const AppConfig& config);
```

**Checks:**
- `hidden_layers >= 1`
- `hidden_width >= 16`
- `learning_rate > 0 && learning_rate < 1`
- `training_batch_size >= 1`
- `replay_buffer_size >= training_batch_size`
- `num_workers >= 1`
- `num_games >= num_workers` (otherwise workers have nothing to do)
- `placement_temperature >= 0`
- `hold_temperature >= 0`
- `temperature_decay > 0 && temperature_decay <= 1`
- `eval_games >= 2` (need at least 1 game per side)
- `checkpoint_dir` and `log_dir` are writable (create if missing)
- `td_lambda >= 0 && td_lambda <= 1`

---

## 4. Configuration Printing

```cpp
/// Print the full configuration to stdout in a readable format.
/// Useful for verifying config before starting a long training run.
void print_config(const AppConfig& config);
```

Outputs all values grouped by section, matching the YAML structure. Called at startup before any work begins.

---

## 5. Wired main.cpp

Replace the skeleton `main.cpp` from Task 01 with the fully wired entry point:

```cpp
// src/main.cpp

#include <iostream>
#include <atomic>
#include <signal.h>
#include <torch/torch.h>

// All component headers...

std::atomic<bool> g_shutdown{false};

void signal_handler(int signal) {
    std::cout << "\nShutdown requested (signal " << signal << ")...\n";
    g_shutdown = true;
}

int main(int argc, char* argv[]) {
    // Register signal handler for clean shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Load configuration
    AppConfig config = load_config(argc, argv);

    if (!validate_config(config)) {
        std::cerr << "Invalid configuration. Exiting.\n";
        return 1;
    }

    print_config(config);

    // Initialize precomputed tables (one-time cost)
    PrecomputedTables tables;
    init_precomputed_tables(tables);

    // Get GPU device
    torch::Device device = get_device();
    std::cout << "Device: " << device << "\n";

    // === Mode dispatch ===

    if (config.mode == "info") {
        std::cout << "CUDA available: "
                  << (torch::cuda::is_available() ? "yes" : "no") << "\n";
        std::cout << "CUDA devices: " << torch::cuda::device_count() << "\n";
        std::cout << "Tensor size: " << kTensorSize << "\n";
        std::cout << "Precomputed tables initialized.\n";
        return 0;
    }

    if (config.mode == "train") {
        // Create model and trainer
        ModelTrainer trainer(config.model, device);

        // Resume from checkpoint if specified
        if (!config.checkpoint_path.empty()) {
            // ... load checkpoint, restore state
        }

        // Create inference engine with initial model clone
        auto inference_model = trainer.clone_for_inference(device);
        InferenceEngine inference(inference_model, device);

        // Create solver config (mutable, updated by training loop)
        SolverConfig solver_config;
        solver_config.placement_temperature = config.training.placement_temperature_start;
        solver_config.hold_temperature = config.training.hold_temperature_start;
        solver_config.exploration_enabled = true;

        // Create self-play orchestrator
        SelfPlayOrchestrator orchestrator(
            config.self_play, tables, inference, solver_config);
        orchestrator.start();

        // Run training loop (blocks until shutdown)
        TrainingLoop training_loop(
            config.training, trainer, inference,
            orchestrator, solver_config, tables);
        training_loop.run(g_shutdown);

        // Clean shutdown
        orchestrator.stop();
        std::cout << "Training complete.\n";
        return 0;
    }

    if (config.mode == "eval") {
        // Load model from checkpoint
        if (config.checkpoint_path.empty()) {
            std::cerr << "Eval mode requires --checkpoint\n";
            return 1;
        }
        ModelTrainer trainer(config.model, device);
        int step; double temp;
        trainer.load_checkpoint(config.checkpoint_path, step, temp);

        // Run evaluation
        EvalResult result = run_evaluation(
            trainer.model(), device, tables,
            config.training.eval_games, config.seed);

        std::cout << "Evaluation results (" << result.total_games << " games):\n";
        std::cout << "  NN win rate: " << result.nn_win_rate() << "\n";
        std::cout << "  NN wins: " << result.nn_wins
                  << ", Heuristic wins: " << result.heuristic_wins
                  << ", Draws: " << result.draws << "\n";
        std::cout << "  As P0: " << result.nn_win_rate_as_p0()
                  << ", As P1: " << result.nn_win_rate_as_p1() << "\n";
        std::cout << "  Avg margin (all games): " << result.avg_duel_margin << "\n";
        return 0;
    }

    if (config.mode == "play") {
        // Human vs bot — handled by UI (Task 12)
        std::cout << "Play mode — requires UI (Task 12).\n";
        return 0;
    }

    if (config.mode == "benchmark") {
        // Run engine and solver benchmarks
        std::cout << "Benchmark mode (run via ctest or benchmark executables).\n";
        return 0;
    }

    std::cerr << "Unknown mode: " << config.mode << "\n";
    std::cerr << "Available: info, train, eval, play, benchmark\n";
    return 1;
}
```

### 5.1 Signal Handling

`SIGINT` (Ctrl+C) and `SIGTERM` set the `g_shutdown` atomic flag. All loops check this flag and perform clean shutdown:
- Training loop completes current step, saves checkpoint
- Self-play orchestrator signals workers and coordinator, joins threads
- No data is lost — in-progress games are abandoned (acceptable)

---

## 6. File Organization

```
src/config/
├── app_config.h             # AppConfig master struct
├── config_loader.h          # load_config, load_config_from_yaml, apply_cli_overrides
├── config_loader.cc         # YAML parsing and CLI override implementation
├── config_validator.h       # validate_config
├── config_validator.cc      # Validation implementation
├── config_printer.h         # print_config
├── config_printer.cc        # Pretty-print implementation
└── CMakeLists.txt

src/main.cpp                 # Fully wired entry point (replaces Task 01 skeleton)
```

```cmake
# src/config/CMakeLists.txt
add_library(config STATIC
    config_loader.cc
    config_validator.cc
    config_printer.cc
)
target_include_directories(config PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(config PUBLIC yaml-cpp::yaml-cpp)
```

---

## 7. Unit Tests

### 7.1 Config Loading Tests (`tests/config/config_loader_test.cc`)

**Default values:**
- Load config with no YAML file. Verify all fields have expected defaults.

**YAML loading:**
- Create a test YAML file with a few overridden values. Load it. Verify overridden values are applied and non-specified values retain defaults.

**Partial YAML:**
- YAML file with only `network` section. Verify network values are loaded, all other sections use defaults.

**CLI overrides:**
- Load from YAML, then apply `--learning_rate 0.0003 --hidden_layers 4`. Verify CLI values take precedence over YAML.

**Mode and config path:**
- Pass `--mode train --config /path/to/config.yaml`. Verify both are captured correctly.

**Unknown arguments:**
- Pass an unknown `--foobar` argument. Verify it prints a warning but doesn't crash.

### 7.2 Config Validation Tests (`tests/config/config_validator_test.cc`)

**Valid config:**
- Default config passes validation.

**Invalid values:**
- hidden_layers = 0 → fails.
- learning_rate = -1 → fails.
- num_games < num_workers → fails.
- replay_buffer_size < training_batch_size → fails.

**Edge cases:**
- hidden_layers = 1, hidden_width = 16 → passes (minimum valid).
- temperature = 0.0 → passes (greedy play).

### 7.3 Config Printer Tests (`tests/config/config_printer_test.cc`)

**Output format:**
- Print config to a string stream. Verify all sections are present and values match.

---

## 8. Definition of Done

This task is complete when:

1. `load_config_from_yaml` correctly parses all YAML fields with default fallbacks for missing values.
2. `apply_cli_overrides` correctly overrides any config value from command-line arguments.
3. CLI overrides take precedence over YAML values, which take precedence over defaults.
4. `validate_config` catches invalid configurations and prints helpful error messages.
5. `print_config` displays the full configuration in a readable format at startup.
6. `main.cpp` correctly dispatches to train, eval, play, info, and benchmark modes.
7. Signal handling enables clean shutdown via Ctrl+C during training.
8. Checkpoint resumption works via `--checkpoint` argument.
9. All unit tests pass.
10. A full training run can be started with `./pro_yams_ai --mode train --config config/default.yaml`.
