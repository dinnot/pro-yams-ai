# Task 09: Replay Buffer & Training Loop

## Overview

Implement the replay buffer for storing training samples, the training loop that continuously trains the model, and the sample collection pipeline that extracts data from completed games. This task also handles temperature/epsilon decay, model swapping to the inference engine, checkpointing, and metric logging.

## Prerequisites

- Task 07 completed (ModelTrainer, InferenceEngine, checkpoint save/load)
- Task 08 completed (SelfPlayOrchestrator, GameInstance, training data extraction)

---

## 1. Replay Buffer

### 1.1 Ring Buffer Design

A fixed-capacity circular buffer of training samples. Pre-allocated, constant-time insert, automatic eviction of oldest samples.

```cpp
// src/training/replay_buffer.h

class ReplayBuffer {
public:
    /// Construct with fixed capacity.
    explicit ReplayBuffer(int capacity);

    /// Add a single sample. If buffer is full, overwrites the oldest sample.
    void add(const TrainingSample& sample);

    /// Add multiple samples at once. More efficient than individual adds.
    void add_batch(const TrainingSample* samples, int count);

    /// Sample a random mini-batch for training.
    /// Writes sampled states and targets into separate contiguous arrays
    /// suitable for direct use with ModelTrainer::train_step.
    ///
    /// @param batch_size Number of samples to draw
    /// @param rng Random engine for sampling
    /// @param out_states Output: batch_size * kTensorSize floats, contiguous
    /// @param out_targets Output: batch_size doubles
    void sample_batch(int batch_size, RNG& rng,
                      float* out_states, double* out_targets);

    /// Current number of samples in the buffer.
    int size() const { return size_; }

    /// Maximum capacity.
    int capacity() const { return capacity_; }

    /// Clear all samples.
    void clear();

    /// Save buffer contents to disk (for checkpointing).
    void save(const std::string& path) const;

    /// Load buffer contents from disk.
    void load(const std::string& path);

private:
    std::vector<TrainingSample> samples_;  // Pre-allocated ring buffer
    int capacity_;
    int size_;
    int write_idx_;
    mutable std::mutex mutex_;  // Protects concurrent add/sample
};
```

### 1.2 Thread Safety

The replay buffer is accessed by the training thread (add + sample) simultaneously. Since both operations happen on the training thread, a mutex is only needed if we allow external threads to add samples. In our design, the training thread handles both collection and training, so the mutex is a safety measure for future flexibility.

If we later separate collection into its own thread, the mutex ensures correctness without redesign.

### 1.3 Implementation Details

**add:**
```cpp
void ReplayBuffer::add(const TrainingSample& sample) {
    std::lock_guard<std::mutex> lock(mutex_);
    samples_[write_idx_] = sample;
    write_idx_ = (write_idx_ + 1) % capacity_;
    if (size_ < capacity_) size_++;
}
```

**add_batch:**
```cpp
void ReplayBuffer::add_batch(const TrainingSample* samples, int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < count; ++i) {
        samples_[write_idx_] = samples[i];
        write_idx_ = (write_idx_ + 1) % capacity_;
        if (size_ < capacity_) size_++;
    }
}
```

**sample_batch:**
```cpp
void ReplayBuffer::sample_batch(int batch_size, RNG& rng,
                                 float* out_states, double* out_targets) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(size_ >= batch_size);

    for (int i = 0; i < batch_size; ++i) {
        int idx = rng.uniform_int(0, size_ - 1);
        memcpy(out_states + i * kTensorSize,
               samples_[idx].state,
               kTensorSize * sizeof(float));
        out_targets[i] = samples_[idx].target;
    }
}
```

### 1.4 Serialization

For checkpointing, save the buffer contents to disk so training can resume without re-generating all samples.

**Save format:** Binary file with header (capacity, size, write_idx) followed by raw sample data. Simple, fast, no parsing overhead.

```cpp
void ReplayBuffer::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&capacity_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&size_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&write_idx_), sizeof(int));
    file.write(reinterpret_cast<const char*>(samples_.data()),
               size_ * sizeof(TrainingSample));
}
```

**Load:** Read header, verify capacity matches, read samples.

---

## 2. Training Configuration

```cpp
// src/training/training_config.h

enum class TDMode {
    kTD0,
    kTDLambda,
    kMC
};

struct TrainingConfig {
    // Training
    TDMode td_mode = TDMode::kMC;      // MC recommended as default (targets never go stale)
    double td_lambda = 0.7;
    int training_batch_size = 256;
    int replay_buffer_size = 2000000;    // ~6.5GB at 3.2KB/sample. Large buffer prevents
                                         // catastrophic forgetting and policy oscillation.
    int min_samples_before_training = 1000;  // Don't train until buffer has this many

    // Model swap
    int model_swap_interval = 100;  // Training steps between model swaps

    // Exploration decay
    double placement_temperature_start = 1.0;
    double hold_temperature_start = 1.0;
    double temperature_decay = 0.999;
    double min_temperature = 0.05;

    // Checkpointing
    int checkpoint_interval = 5000;   // Training steps between checkpoints
    int max_checkpoints = 5;
    std::string checkpoint_dir = "./checkpoints";

    // Logging
    int log_interval = 100;           // Training steps between log entries
    std::string log_dir = "./logs";

    // Evaluation
    int eval_interval = 1000;         // Training steps between eval runs
    int eval_games = 200;
};
```

---

## 3. Training Loop

### 3.1 Training Thread

The training thread runs the complete training pipeline: collecting completed games, extracting samples, training, model swapping, decay, checkpointing, and logging.

```cpp
// src/training/training_loop.h

class TrainingLoop {
public:
    TrainingLoop(const TrainingConfig& config,
                 ModelTrainer& trainer,
                 InferenceEngine& inference,
                 SelfPlayOrchestrator& orchestrator,
                 SolverConfig& solver_config,
                 const PrecomputedTables& tables);

    /// Run the training loop. Blocks until shutdown is signaled.
    void run(std::atomic<bool>& shutdown);

    /// Get current training metrics (thread-safe read).
    TrainingMetrics get_metrics() const;

private:
    void collect_completed_games();
    void do_training_step();
    void maybe_swap_model();
    void maybe_checkpoint();
    void maybe_evaluate();
    void maybe_log();
    void decay_exploration();

    // Configuration
    TrainingConfig config_;

    // Components
    ModelTrainer& trainer_;
    InferenceEngine& inference_;
    SelfPlayOrchestrator& orchestrator_;
    SolverConfig& solver_config_;
    const PrecomputedTables& tables_;

    // Replay buffer
    ReplayBuffer buffer_;

    // Pre-allocated training batch buffers
    std::vector<float> batch_states_;     // training_batch_size * kTensorSize
    std::vector<double> batch_targets_;   // training_batch_size

    // Pre-allocated sample extraction buffer
    static constexpr int kMaxSamplesPerGame = 156;
    TrainingSample sample_buffer_[kMaxSamplesPerGame];

    // Current exploration values
    double placement_temperature_;
    double hold_temperature_;

    // Metrics
    mutable std::mutex metrics_mutex_;
    TrainingMetrics metrics_;

    // Seed for sampling RNG (separate from game RNGs)
    RNG sampling_rng_;

    // Game counter for seeding recycled games
    uint64_t game_counter_;
};
```

### 3.2 Main Loop

```cpp
void TrainingLoop::run(std::atomic<bool>& shutdown) {
    // Initialize exploration values
    placement_temperature_ = config_.placement_temperature_start;
    hold_temperature_ = config_.hold_temperature_start;

    // Pre-allocate batch buffers
    batch_states_.resize(config_.training_batch_size * kTensorSize);
    batch_targets_.resize(config_.training_batch_size);

    while (!shutdown) {
        // === Phase 1: Collect completed games ===
        collect_completed_games();

        // === Phase 2: Train (if enough samples) ===
        if (buffer_.size() >= config_.min_samples_before_training) {
            do_training_step();
            maybe_swap_model();
            maybe_checkpoint();
            maybe_evaluate();
            maybe_log();
            decay_exploration();
        } else {
            // Not enough samples yet — brief sleep to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}
```

### 3.3 Collecting Completed Games

```cpp
void TrainingLoop::collect_completed_games() {
    GameInstance* games[64];  // Collect up to 64 at a time
    int count = orchestrator_.collect_completed(games, 64);

    for (int i = 0; i < count; ++i) {
        GameInstance* game = games[i];

        // Extract training samples from trajectory
        int num_samples = extract_training_samples(
            *game, config_.td_mode, config_.td_lambda,
            sample_buffer_, kMaxSamplesPerGame);

        // Add to replay buffer
        buffer_.add_batch(sample_buffer_, num_samples);

        // Update metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_.games_completed++;
            if (game->result == 1.0) metrics_.p0_wins++;
            else if (game->result == 0.0) metrics_.p1_wins++;
            else metrics_.draws++;
        }

        // Recycle game for a new round
        orchestrator_.recycle_game(game, game_counter_++);
    }
}
```

### 3.4 Training Step

```cpp
void TrainingLoop::do_training_step() {
    // Sample mini-batch from replay buffer
    buffer_.sample_batch(config_.training_batch_size, sampling_rng_,
                         batch_states_.data(), batch_targets_.data());

    // Run training step
    double loss = trainer_.train_step(
        batch_states_.data(), batch_targets_.data(),
        config_.training_batch_size);

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.training_steps++;
        metrics_.latest_loss = loss;
        metrics_.loss_sum += loss;
    }
}
```

### 3.5 Model Swap

```cpp
void TrainingLoop::maybe_swap_model() {
    if (trainer_.training_step() % config_.model_swap_interval != 0) return;

    auto new_model = trainer_.clone_for_inference(inference_.device());
    inference_.swap_model(new_model);

    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.model_swaps++;
    }
}
```

### 3.6 Exploration Decay

```cpp
void TrainingLoop::decay_exploration() {
    // Decay temperatures
    placement_temperature_ = std::max(
        config_.min_temperature,
        placement_temperature_ * config_.temperature_decay);
    hold_temperature_ = std::max(
        config_.min_temperature,
        hold_temperature_ * config_.temperature_decay);

    // Update solver config (workers read this)
    solver_config_.placement_temperature = placement_temperature_;
    solver_config_.hold_temperature = hold_temperature_;
    solver_config_.exploration_enabled = (placement_temperature_ > 0.001);
}
```

### 3.7 Checkpointing

```cpp
void TrainingLoop::maybe_checkpoint() {
    if (trainer_.training_step() % config_.checkpoint_interval != 0) return;

    int step = trainer_.training_step();
    std::string prefix = config_.checkpoint_dir + "/checkpoint_step_"
                          + std::to_string(step);

    // Save model + optimizer
    trainer_.save_checkpoint(prefix, step, placement_temperature_);

    // Save replay buffer
    buffer_.save(prefix + ".buffer");

    // Prune old checkpoints
    prune_old_checkpoints(config_.checkpoint_dir, config_.max_checkpoints);
}
```

### 3.8 Logging

```cpp
void TrainingLoop::maybe_log() {
    if (trainer_.training_step() % config_.log_interval != 0) return;

    TrainingMetrics m;
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        m = metrics_;
    }

    // Log to structured file (CSV or JSON)
    log_metrics(config_.log_dir, m, placement_temperature_,
                hold_temperature_, epsilon_, buffer_.size());
}
```

---

## 4. Training Metrics

```cpp
// src/training/metrics.h

struct TrainingMetrics {
    int training_steps = 0;
    int games_completed = 0;
    int p0_wins = 0;
    int p1_wins = 0;
    int draws = 0;
    int model_swaps = 0;
    double latest_loss = 0.0;
    double loss_sum = 0.0;
    int completed_queue_size = 0;  // Monitor for backpressure issues

    double avg_loss() const {
        return training_steps > 0 ? loss_sum / training_steps : 0.0;
    }
    double p0_win_rate() const {
        return games_completed > 0
            ? static_cast<double>(p0_wins) / games_completed : 0.0;
    }
};
```

---

## 5. Metric Logging

### 5.1 Log Format

CSV file with one row per log entry, written to `log_dir/training_log.csv`:

```csv
timestamp,step,loss,avg_loss,games,p0_wins,p1_wins,draws,p0_wr,buffer_size,completed_queue,placement_temp,hold_temp,model_swaps
2026-04-02T10:30:00,100,0.0342,0.0451,847,421,418,8,0.497,12340,3,0.95,0.95,1
```

### 5.2 Log Function

```cpp
/// Append a metrics entry to the training log CSV.
void log_metrics(const std::string& log_dir,
                 const TrainingMetrics& metrics,
                 double placement_temp, double hold_temp, double epsilon,
                 int buffer_size);
```

Creates the CSV file with headers on first call, appends rows on subsequent calls.

---

## 6. Checkpoint Pruning

```cpp
/// Remove old checkpoints, keeping only the most recent max_keep.
/// Checkpoints are identified by step number in the filename.
void prune_old_checkpoints(const std::string& dir, int max_keep);
```

Implementation: list all files matching `checkpoint_step_*.model` in the directory, sort by step number, delete the oldest until only `max_keep` remain. Also delete corresponding `.optimizer` and `.buffer` files.

---

## 7. Resuming from Checkpoint

```cpp
/// Resume training from a checkpoint.
/// Loads model weights, optimizer state, replay buffer, and exploration values.
/// Returns the training step to resume from.
int resume_from_checkpoint(const std::string& checkpoint_dir,
                            ModelTrainer& trainer,
                            ReplayBuffer& buffer,
                            double& placement_temperature,
                            double& hold_temperature,
                            double& epsilon);
```

Implementation: find the latest checkpoint in the directory (highest step number), load model + optimizer via `ModelTrainer::load_checkpoint`, load replay buffer via `ReplayBuffer::load`, return the step count.

---

## 8. File Organization

```
src/training/
├── training_config.h         # TrainingConfig, TDMode
├── replay_buffer.h           # ReplayBuffer class
├── replay_buffer.cc          # Ring buffer implementation + serialization
├── metrics.h                 # TrainingMetrics struct
├── logging.h                 # log_metrics, prune_old_checkpoints
├── logging.cc                # Logging implementation
├── training_loop.h           # TrainingLoop class
├── training_loop.cc          # Main training loop implementation
├── resume.h                  # resume_from_checkpoint
├── resume.cc                 # Resume implementation
└── CMakeLists.txt
```

```cmake
# src/training/CMakeLists.txt
add_library(training STATIC
    replay_buffer.cc
    logging.cc
    training_loop.cc
    resume.cc
)
target_include_directories(training PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(training PUBLIC model self_play engine)
```

---

## 9. Unit Tests

### 9.1 Replay Buffer Tests (`tests/training/replay_buffer_test.cc`)

**Basic operations:**
- Add 10 samples, verify size = 10.
- Add samples up to capacity, verify size = capacity.
- Add beyond capacity, verify size stays at capacity (oldest evicted).

**Ring buffer behavior:**
- Fill buffer to capacity with samples numbered 0 to N-1. Add sample N. Verify sample 0 is no longer retrievable (overwritten) and sample N is present.

**Sampling:**
- Fill buffer with 100 samples. Sample batch of 10. Verify all sampled indices are in valid range. Verify output arrays have correct values.
- Verify sampling is uniform-ish: sample 100K times from a buffer of 100, check each index is hit roughly 1000 times (within statistical tolerance).

**Empty buffer:**
- Verify `sample_batch` asserts when buffer has fewer samples than requested batch size.

**Serialization round-trip:**
- Fill buffer with known samples. Save to disk. Create new buffer, load from disk. Verify contents match exactly.
- Resume scenario: fill buffer, save, add more samples, save again. Load the second save, verify it contains the latest state.

### 9.2 Training Loop Tests (`tests/training/training_loop_test.cc`)

**Sample collection:**
- Create a GameInstance with a known trajectory. Mark as completed. Run collect_completed_games. Verify samples are in the replay buffer with correct count and values.

**Training step:**
- Pre-fill replay buffer with synthetic samples. Run do_training_step. Verify loss is returned and metrics are updated.

**Model swap:**
- Run enough training steps to trigger a model swap. Verify inference engine received updated weights (run inference before and after swap, verify outputs change).

**Exploration decay:**
- Set initial temperature = 1.0, decay = 0.5. Run decay_exploration twice. Verify temperature = 0.25. Verify it doesn't go below min_temperature.

**Checkpoint round-trip:**
- Train for 50 steps. Checkpoint. Create a new TrainingLoop, resume from checkpoint. Verify:
  - Training step count restored
  - Replay buffer contents restored
  - Temperature values restored
  - Next training step doesn't spike in loss (optimizer state preserved)

### 9.3 Logging Tests (`tests/training/logging_test.cc`)

**CSV output:**
- Run log_metrics several times. Read the CSV file, verify headers are correct and rows have the right number of fields.
- Verify timestamps are valid and increasing.

**Checkpoint pruning:**
- Create 10 dummy checkpoint files. Call prune with max_keep=3. Verify only the 3 most recent remain.
- Verify associated .optimizer and .buffer files are also deleted.

---

## 10. Benchmarks

Add to `tests/benchmarks/training_bench.cc`:

- **BM_ReplayBufferAdd:** Benchmark adding samples to the buffer (individual and batch).
- **BM_ReplayBufferSample:** Benchmark sampling mini-batches of 64, 256, 1024.
- **BM_TrainingStep:** Benchmark one complete training step (sample + forward + backward + optimizer) for various batch sizes.
- **BM_CollectAndExtract:** Benchmark collecting a completed game and extracting training samples.
- **BM_CheckpointSave:** Benchmark saving model + optimizer + buffer.
- **BM_CheckpointLoad:** Benchmark loading from checkpoint.

---

## 11. Definition of Done

This task is complete when:

1. ReplayBuffer correctly implements ring buffer semantics with add, sample, and eviction.
2. ReplayBuffer serialization saves and loads correctly for checkpoint resumability.
3. TrainingLoop collects completed games, extracts samples, and adds to buffer.
4. TrainingLoop performs training steps using sampled mini-batches from the buffer.
5. Model swap to inference engine occurs at configured intervals.
6. Temperature and epsilon decay correctly with configurable rates and minimums.
7. Checkpoints save and load all state (model, optimizer, buffer, exploration values).
8. Old checkpoints are pruned correctly.
9. Metrics are logged to CSV at configured intervals.
10. Training resumes correctly from checkpoint without loss spikes or state corruption.
11. All unit tests pass.
12. Benchmarks establish baseline training throughput.
