# Task 08: Solver NN Integration & Async Self-Play Architecture

## Overview

Connect the solver to the neural network for afterstate evaluation and implement the async self-play architecture: worker threads for CPU game logic, a coordinator thread for GPU batching, and the queue system that connects them.

This is the most architecturally complex task. It produces the core infrastructure for running self-play games with NN inference at scale.

## Prerequisites

- Task 05 completed (solver two-step API, SolverBuffers, heuristic bot)
- Task 06 completed (tensor generation, generate_tensor_batch)
- Task 07 completed (InferenceEngine, batch_inference)

---

## 1. Game Instance

Each concurrent game has its own self-contained state, buffers, and trajectory storage. A GameInstance is only ever accessed by one thread at a time — no concurrent access, no locks needed on per-game data.

```cpp
// src/self_play/game_instance.h

enum class GamePhase {
    kNeedRequests,     // Worker: generate requests + tensors
    kWaitingInference, // In pending queue, waiting for GPU
    kNeedResolve,      // Worker: EVs ready, run solver_resolve
    kCompleted         // Game finished, ready for data collection
};

struct TrajectoryStep {
    float tensor[kTensorSize];   // Afterstate tensor at time of placement
    double value;                 // V(s) from solver for chosen placement
    int8_t player;               // Which player made this placement
};

struct GameInstance {
    // === Game state ===
    GameState state;
    GameContext ctx;
    RNG rng;
    int game_id;
    GamePhase phase;

    // === Solver buffers (reused across turns) ===
    SolverBuffers solver_buffers;

    // === Tensor buffer for afterstate evaluation ===
    // Holds tensors for all afterstates in current solver call.
    // Written by worker (tensor generation), read by coordinator (GPU batching).
    static constexpr int kMaxAfterstates = 512;
    float tensor_buffer[kMaxAfterstates * kTensorSize];

    // === Trajectory for training data ===
    // One entry per placement over the entire game.
    TrajectoryStep trajectory[kNumPlayers * kNumColumns * kNumRows]; // 156 max
    int trajectory_length;

    // === Game result ===
    double result;  // 1.0, 0.0, or 0.5 (set when game completes)
};
```

**Memory per GameInstance:** ~1.6MB (tensor buffer) + ~500KB (trajectory) + ~13KB (solver buffers) + ~200 bytes (game state) ≈ **2.1MB**.

With 512 concurrent games: ~1.1GB total. Well within 64GB RAM.

---

## 2. Queue System

Three thread-safe queues manage game flow:

```cpp
// src/self_play/game_queues.h

/// Thread-safe queue for passing GameInstance pointers between threads.
/// Uses mutex + condition variable for blocking operations.
class GameQueue {
public:
    /// Push a game onto the queue.
    void push(GameInstance* game);

    /// Push multiple games at once (reduces lock acquisitions).
    void push_batch(GameInstance** games, int count);

    /// Pop a game from the queue. Blocks until one is available.
    GameInstance* pop();

    /// Try to pop a game. Returns nullptr if queue is empty (non-blocking).
    GameInstance* try_pop();

    /// Collect up to max_count games, waiting up to timeout_ms.
    /// Returns the number of games collected.
    /// Used by coordinator to build inference batches.
    int collect(GameInstance** out, int max_count, int timeout_ms);

    /// Number of games currently in the queue.
    int size() const;

private:
    std::deque<GameInstance*> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};
```

Three queue instances:

```cpp
GameQueue available_queue;    // Games ready for CPU work
GameQueue pending_queue;      // Games with tensors, waiting for GPU
GameQueue completed_queue;    // Finished games, ready for data collection
```

---

## 3. Worker Threads

### 3.1 Worker Logic

Each worker thread runs a simple loop: pull a game from the available queue, do CPU work, push it to the appropriate next queue.

```cpp
// src/self_play/worker.h

/// Worker thread function. Runs until shutdown is signaled.
///
/// @param available Available games queue (pull from here)
/// @param pending Pending inference queue (push here after tensor generation)
/// @param available_again Available queue again (push here after resolve)
/// @param completed Completed games queue (push here when game ends)
/// @param tables Precomputed tables (read-only, shared)
/// @param config Solver config (exploration settings)
/// @param shutdown Atomic flag to signal shutdown
void worker_thread(GameQueue& available, GameQueue& pending,
                   GameQueue& completed,
                   const PrecomputedTables& tables,
                   const SolverConfig& config,
                   std::atomic<bool>& shutdown);
```

**Implementation:**

```
while (!shutdown):
    game = available.pop()  // blocks until a game is available
    if shutdown: break
    
    switch game->phase:
    
    case kNeedRequests:
        // Step 1: Generate afterstate requests
        solver_get_requests(game->state, game->ctx, tables,
                           game->solver_buffers)
        
        // Step 2: Generate tensors for all requests
        generate_tensor_batch(
            game->state.board, game->ctx,
            game->state.board.current_player,
            game->solver_buffers.requests,
            game->solver_buffers.request_count,
            tables,
            game->tensor_buffer)
        
        // Step 3: Submit to pending queue for GPU inference
        game->phase = kWaitingInference
        pending.push(game)
    
    case kNeedResolve:
        // EVs are populated by coordinator, run solver resolution
        result = solver_resolve(game->state, game->ctx, tables,
                               game->solver_buffers)
        
        if result.should_place:
            // Record trajectory step BEFORE applying placement
            step = &game->trajectory[game->trajectory_length]
            
            // Find the EV for the chosen placement
            // (this is the value of the afterstate the solver picked)
            step->value = result.expected_value
            step->player = game->state.board.current_player
            
            // Generate the specific afterstate tensor for the chosen placement
            // (we need to store this for training, not just the EV)
            BoardState clone = game->state.board
            clone.cells[step->player][result.placement.column][result.placement.row] 
                = result.score
            clone.cells_filled++
            clone.current_player = 1 - step->player
            generate_tensor(clone, game->ctx, step->player, tables, step->tensor)
            game->trajectory_length++
            
            // Apply placement (updates board, context, rolls dice for next player)
            perform_placement(result.placement, result.score,
                            game->state, game->ctx, game->rng)
            
            if is_terminal(game->state.board):
                game->result = get_game_result(game->state, game->ctx)
                game->phase = kCompleted
                completed.push(game)
            else:
                // Next player's turn, need new solver call
                game->phase = kNeedRequests
                available.push(game)
        
        else:  // should hold and reroll
            if !can_reroll(game->state, game->ctx):
                assert(false)  // Solver bug
            perform_hold(result.hold_mask, game->state, game->rng)
            // Same turn, new dice, need new solver call
            game->phase = kNeedRequests
            available.push(game)
```

### 3.2 Trajectory Tensor Generation

Note: in the worker's kNeedResolve phase, after the solver picks a placement, we generate the afterstate tensor for the *chosen* placement specifically. This might seem redundant since we already generated tensors for all afterstates in the kNeedRequests phase. However, those tensors are in `tensor_buffer` which gets overwritten on the next solver call. We need to copy the chosen afterstate's tensor into the trajectory for training.

**Optimization:** Instead of regenerating the tensor, we could look up which request index the solver chose and copy from `tensor_buffer`. But `solver_resolve` returns the placement, not the request index. We could modify `SolverResult` to include the request index:

```cpp
// Add to SolverResult:
int16_t chosen_request_idx;  // Index into requests[] for the chosen placement
```

Then the worker can do:
```cpp
memcpy(step->tensor,
       game->tensor_buffer + result.chosen_request_idx * kTensorSize,
       kTensorSize * sizeof(float));
```

This avoids regenerating the tensor entirely. A single `memcpy` of 3.2KB.

---

## 4. Coordinator Thread

### 4.1 Coordinator Logic

The coordinator thread manages GPU inference batching. It collects pending games, assembles a contiguous batch tensor, runs inference, and distributes results.

```cpp
// src/self_play/coordinator.h

/// Coordinator thread function. Manages GPU inference batching.
///
/// @param pending Pending games queue (pull from here)
/// @param available Available games queue (push here after inference)
/// @param inference InferenceEngine for GPU batch inference
/// @param config Batch size and timeout configuration
/// @param shutdown Atomic flag to signal shutdown
void coordinator_thread(GameQueue& pending, GameQueue& available,
                        InferenceEngine& inference,
                        const SelfPlayConfig& config,
                        std::atomic<bool>& shutdown);
```

### 4.2 Batch Assembly

```cpp
struct SelfPlayConfig {
    int max_inference_batch = 1024;   // Max tensors per GPU batch
    int min_games_per_batch = 2;       // Min games before sending (or timeout)
    int batch_timeout_ms = 5;          // Max wait time for batch assembly
    int num_workers = 16;              // Number of worker threads
    int num_games = 512;               // Total concurrent games
};
```

**Coordinator loop:**

```
// Pre-allocated buffers
float batch_tensor_buffer[max_inference_batch * kTensorSize]
double batch_result_buffer[max_inference_batch]

// Track which games are in the current batch and their tensor ranges
struct BatchEntry {
    GameInstance* game;
    int start_idx;      // Start index in batch_tensor_buffer
    int count;          // Number of tensors for this game
};
BatchEntry batch_entries[max_games_per_batch]  // generous upper bound

while (!shutdown):
    // === Collect games from pending queue ===
    int num_entries = 0
    int total_tensors = 0
    
    // Wait for at least min_games_per_batch or timeout
    collected = pending.collect(temp_buffer, min_games_per_batch, batch_timeout_ms)
    
    // Add collected games to batch
    for each game in collected:
        if total_tensors + game->solver_buffers.request_count > max_inference_batch:
            // This game would overflow the batch — put it back
            pending.push(game)
            break
        batch_entries[num_entries] = {
            game,
            total_tensors,
            game->solver_buffers.request_count
        }
        // Copy this game's tensors into the contiguous batch buffer
        memcpy(batch_tensor_buffer + total_tensors * kTensorSize,
               game->tensor_buffer,
               game->solver_buffers.request_count * kTensorSize * sizeof(float))
        total_tensors += game->solver_buffers.request_count
        num_entries++
    
    // Try to fill remaining batch space with more games (non-blocking)
    while total_tensors < max_inference_batch:
        game = pending.try_pop()
        if game == nullptr: break
        if total_tensors + game->solver_buffers.request_count > max_inference_batch:
            pending.push(game)
            break
        // Add to batch (same as above)
        batch_entries[num_entries] = {game, total_tensors, game->solver_buffers.request_count}
        memcpy(...)
        total_tensors += game->solver_buffers.request_count
        num_entries++
    
    if total_tensors == 0: continue  // Nothing to process
    
    // === Run GPU inference ===
    inference.batch_inference(batch_tensor_buffer, total_tensors, batch_result_buffer)
    
    // === Distribute results back to games ===
    for i in 0..num_entries:
        entry = batch_entries[i]
        // Copy EVs from batch result buffer to game's solver buffer
        memcpy(entry.game->solver_buffers.evs,
               batch_result_buffer + entry.start_idx,
               entry.count * sizeof(double))
        entry.game->phase = kNeedResolve
        available.push(entry.game)
```

### 4.3 Batch Size Dynamics

The batch naturally adapts to game progress:
- **Early game:** Each game has many afterstates (200-336). A batch of 1024 tensors holds 3-5 games.
- **Late game:** Each game has few afterstates (10-50). A batch of 1024 tensors holds 20-100 games.
- **Mixed:** Games at different stages naturally fill the batch with varying game counts.

The min_games_per_batch = 2 ensures we don't waste GPU cycles on tiny batches. The timeout ensures we don't wait forever when games are slow. The max_inference_batch caps the batch to what the GPU handles efficiently.

---

## 5. Self-Play Orchestrator

Top-level management that creates all games, threads, and queues:

```cpp
// src/self_play/orchestrator.h

class SelfPlayOrchestrator {
public:
    /// Initialize the orchestrator with configuration.
    SelfPlayOrchestrator(const SelfPlayConfig& config,
                          const PrecomputedTables& tables,
                          InferenceEngine& inference,
                          const SolverConfig& solver_config);

    /// Start self-play: create games, launch workers and coordinator.
    void start();

    /// Stop self-play: signal shutdown, join all threads.
    void stop();

    /// Collect completed games for training data extraction.
    /// Returns number of games collected.
    int collect_completed(GameInstance** out, int max_count);

    /// Recycle a game instance: reset state, put back in available queue.
    void recycle_game(GameInstance* game, uint64_t new_seed);

private:
    // Game instances (pre-allocated pool)
    std::vector<std::unique_ptr<GameInstance>> games_;

    // Queues
    GameQueue available_queue_;
    GameQueue pending_queue_;
    GameQueue completed_queue_;

    // Threads
    std::vector<std::thread> workers_;
    std::thread coordinator_;
    std::atomic<bool> shutdown_{false};

    // Shared resources
    const PrecomputedTables& tables_;
    InferenceEngine& inference_;
    SelfPlayConfig config_;
    SolverConfig solver_config_;
};
```

### 5.1 Initialization

```cpp
void SelfPlayOrchestrator::start() {
    // Pre-allocate all game instances
    for (int i = 0; i < config_.num_games; ++i) {
        auto game = std::make_unique<GameInstance>();
        game->game_id = i;
        game->rng = RNG(base_seed + i);
        init_game(game->state, game->ctx, game->rng);
        game->phase = GamePhase::kNeedRequests;
        game->trajectory_length = 0;
        available_queue_.push(game.get());
        games_.push_back(std::move(game));
    }

    // Launch worker threads
    for (int i = 0; i < config_.num_workers; ++i) {
        workers_.emplace_back(worker_thread,
            std::ref(available_queue_), std::ref(pending_queue_),
            std::ref(completed_queue_),
            std::ref(tables_), std::ref(solver_config_),
            std::ref(shutdown_));
    }

    // Launch coordinator thread
    coordinator_ = std::thread(coordinator_thread,
        std::ref(pending_queue_), std::ref(available_queue_),
        std::ref(inference_), std::ref(config_),
        std::ref(shutdown_));
}
```

### 5.2 Game Recycling

When a completed game's training data has been extracted, reset it for a new game:

```cpp
void SelfPlayOrchestrator::recycle_game(GameInstance* game, uint64_t new_seed) {
    game->rng = RNG(new_seed);
    init_game(game->state, game->ctx, game->rng);
    game->phase = GamePhase::kNeedRequests;
    game->trajectory_length = 0;
    available_queue_.push(game);
}
```

---

## 6. Training Data Extraction

When a game completes, extract TD samples from its trajectory:

```cpp
// src/self_play/training_data.h

struct TrainingSample {
    float state[kTensorSize];    // Afterstate tensor
    double target;                // TD target (win probability)
};

/// Extract training samples from a completed game's trajectory.
///
/// For TD(0): target = 1 - V(s') where s' is the next afterstate
///            from the opponent's perspective.
/// For MC: target = game outcome (1.0 / 0.0 / 0.5).
///
/// @param game The completed game instance
/// @param td_mode Training mode (td0, td_lambda, mc)
/// @param td_lambda Lambda value (only used if td_mode == td_lambda)
/// @param samples Output: pre-allocated array for samples
/// @param max_samples Maximum number of samples to extract
/// @return Number of samples extracted
int extract_training_samples(const GameInstance& game,
                              TDMode td_mode, double td_lambda,
                              TrainingSample* samples, int max_samples);
```

### 6.1 TD(0) Target Computation

For each placement in the trajectory:

```
For trajectory step i (player A places at time i):
    tensor_i = trajectory[i].tensor    // A's afterstate
    value_i = trajectory[i].value      // V(s) = P(A wins) from A's perspective

    // Find the next step where the opponent (B) places
    // That's trajectory[i+1] (B's afterstate after B's placement)
    If i+1 < trajectory_length:
        value_next = trajectory[i+1].value  // V(s') = P(B wins) from B's perspective
        target = 1.0 - value_next           // Convert to P(A wins)
    Else:
        // Last placement of the game — use actual game outcome
        target = game outcome from A's perspective
        If trajectory[i].player == 0:
            target = game.result
        Else:
            target = 1.0 - game.result

    Sample: (tensor_i, target)
```

### 6.2 MC Target Computation

Simpler — every trajectory step gets the actual game outcome:

```
For trajectory step i (player P places):
    tensor_i = trajectory[i].tensor
    If P == 0:
        target = game.result          // P(player 0 wins)
    Else:
        target = 1.0 - game.result    // P(player 1 wins)
    
    Sample: (tensor_i, target)
```

### 6.3 TD(λ) Target Computation

Blended target using exponential decay:

```
For trajectory step i (player A places):
    // Collect future returns from A's perspective
    // Step through trajectory, alternating perspectives
    target = 0.0
    weight_sum = 0.0
    lambda_power = 1.0
    
    for j in (i+1) to (trajectory_length - 1):
        // Each subsequent step alternates perspective
        if trajectory[j].player != trajectory[i].player:
            bootstrap_value = 1.0 - trajectory[j].value
        else:
            bootstrap_value = trajectory[j].value
        
        weight = (1 - td_lambda) * lambda_power
        target += weight * bootstrap_value
        weight_sum += weight
        lambda_power *= td_lambda
    
    // Terminal value gets remaining weight
    terminal_target = (trajectory[i].player == 0) ? game.result : 1.0 - game.result
    target += lambda_power * terminal_target
    weight_sum += lambda_power
    
    // Normalize (weight_sum should equal 1.0 but normalize for safety)
    target /= weight_sum
    
    Sample: (tensor_i, target)
```

---

## 7. Exploration Integration

The solver's exploration (softmax temperature on placements and hold masks) is configured via `SolverConfig` passed to the workers:

```cpp
struct SolverConfig {
    double placement_temperature;
    double hold_temperature;
    bool exploration_enabled;
};
```

During self-play, exploration is enabled. During evaluation games (Task 13), exploration is disabled (greedy play).

The temperature values are managed externally (by the training loop in Task 11) and can be updated between games. Workers read the config at the start of each solver call — no synchronization needed since the config is updated atomically between game batches, not mid-game.

---

## 8. File Organization

```
src/self_play/
├── game_instance.h          # GameInstance, GamePhase, TrajectoryStep
├── game_queues.h             # GameQueue thread-safe queue
├── game_queues.cc            # Queue implementation
├── worker.h                  # Worker thread function
├── worker.cc                 # Worker implementation
├── coordinator.h             # Coordinator thread function
├── coordinator.cc            # Coordinator implementation
├── orchestrator.h            # SelfPlayOrchestrator class
├── orchestrator.cc           # Orchestrator implementation
├── training_data.h           # TrainingSample, extract_training_samples
├── training_data.cc          # Training data extraction
└── CMakeLists.txt
```

```cmake
# src/self_play/CMakeLists.txt
add_library(self_play STATIC
    game_queues.cc
    worker.cc
    coordinator.cc
    orchestrator.cc
    training_data.cc
)
target_include_directories(self_play PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(self_play PUBLIC engine solver model)
```

---

## 9. Unit Tests

### 9.1 Queue Tests (`tests/self_play/queue_test.cc`)

**Basic operations:**
- Push and pop a single game. Verify same pointer returned.
- Push 10 games, pop 10. Verify FIFO order.

**Thread safety:**
- Launch 4 threads, each pushing 100 games. One thread popping. Verify all 400 games collected, no crashes.

**Collect with timeout:**
- Push 3 games, collect with max=5 and timeout=100ms. Verify returns 3 immediately (doesn't wait for 5).
- Empty queue, collect with timeout=50ms. Verify returns 0 after ~50ms.

**Blocking pop:**
- Launch a thread that calls pop() on empty queue. After 100ms, push a game from main thread. Verify the blocking thread receives it.

### 9.2 Worker Tests (`tests/self_play/worker_test.cc`)

**Single turn processing:**
- Create one GameInstance, put in available queue. Run worker logic for one iteration. Verify game ends up in pending queue with tensors generated and correct request count.

**Resolve and place:**
- Set up a GameInstance in kNeedResolve phase with pre-populated EVs. Run worker logic. Verify placement is applied, trajectory step recorded, game moves to available (or completed if terminal).

**Hold and reroll:**
- Set up scenario where solver recommends holding. Verify perform_hold is called, game goes back to available with kNeedRequests phase.

**Trajectory recording:**
- Play a complete game (using heuristic EVs for simplicity). Verify trajectory has correct length (156 steps), each step has valid tensor and value.

### 9.3 Coordinator Tests (`tests/self_play/coordinator_test.cc`)

**Batch assembly:**
- Push 3 games to pending queue with known request counts (100, 200, 150). Run one coordinator iteration. Verify:
  - All 450 tensors sent in one batch
  - EVs distributed back correctly to each game
  - Games moved to available queue with kNeedResolve phase

**Batch overflow:**
- Push games whose total tensors exceed max_inference_batch. Verify excess games are pushed back to pending, not lost.

**Timeout behavior:**
- Push 1 game (below min_games_per_batch). Verify coordinator waits up to timeout then processes it anyway.

### 9.4 Integration Tests (`tests/self_play/integration_test.cc`)

**Complete game with NN:**
- Initialize InferenceEngine with a random model. Start orchestrator with 4 workers, 8 games. Let games run to completion. Verify:
  - All 8 games complete
  - Each game's trajectory has 156 steps
  - Game results are valid (0.0, 0.5, or 1.0)

**Training data extraction:**
- Complete a game, extract TD(0) samples. Verify:
  - Sample count = trajectory length
  - All targets are in [0, 1]
  - Targets use perspective-flipped V(s') correctly

**MC vs TD(0):**
- Extract samples with MC mode. Verify all targets equal the game outcome (from each player's perspective).
- Extract with TD(0). Verify targets differ from MC targets (they use bootstrapped values).

### 9.5 Trajectory Tests (`tests/self_play/trajectory_test.cc`)

**Perspective correctness:**
- Play a game, examine trajectory. For consecutive steps i and i+1 (different players), verify that step i's tensor encodes player i's perspective and step i+1 encodes the other player's perspective.

**TD(0) target math:**
- Manually compute expected targets for a short trajectory (e.g., 4 steps). Verify extract_training_samples produces matching values.

**TD(λ) blending:**
- With λ=0, verify targets match TD(0).
- With λ=1, verify targets match MC.
- With λ=0.5, verify targets are between TD(0) and MC.

---

## 10. Benchmarks

Add to `tests/benchmarks/self_play_bench.cc`:

- **BM_WorkerIteration:** Benchmark one worker iteration (request generation + tensor batch generation) for early-game and late-game board states.
- **BM_CoordinatorBatch:** Benchmark coordinator batch assembly + GPU inference + result distribution for various batch sizes.
- **BM_FullGameWithNN:** Benchmark a complete game using NN inference (measures end-to-end games/sec).
- **BM_TrainingDataExtraction:** Benchmark extracting training samples from a completed game trajectory.
- **BM_QueueThroughput:** Benchmark push/pop throughput of GameQueue under contention.

---

## 11. Definition of Done

This task is complete when:

1. GameInstance struct correctly holds all per-game state, buffers, and trajectory.
2. GameQueue provides thread-safe push/pop/collect with blocking and timeout support.
3. Worker threads correctly process kNeedRequests (generate requests + tensors) and kNeedResolve (solver resolve + placement/hold).
4. Coordinator thread correctly assembles batches from pending games, runs GPU inference, and distributes results.
5. Batch sizing respects min_games_per_batch and max_inference_batch limits.
6. Trajectory recording captures correct tensors and values at each placement.
7. Training data extraction correctly computes TD(0), TD(λ), and MC targets with proper perspective flipping.
8. A complete game runs end-to-end with NN inference without deadlocks or data corruption.
9. Multiple games run concurrently without race conditions.
10. All unit tests pass.
11. Benchmarks establish baseline games/sec with NN inference.
12. Shutdown is clean — no threads left hanging, no games lost.
