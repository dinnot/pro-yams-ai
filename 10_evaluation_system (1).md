# Task 10: Evaluation System

## Overview

Implement the evaluation system that periodically measures the NN bot's strength by playing games against the heuristic bot. Evaluation runs synchronously on the training thread, uses greedy play (no exploration), and logs win rates and other metrics.

## Prerequisites

- Task 05 completed (heuristic bot, solver)
- Task 07 completed (ModelTrainer with direct model access)
- Task 09 completed (training loop with `maybe_evaluate()` hook)

---

## 1. Evaluation Configuration

Evaluation parameters are part of `TrainingConfig` (from Task 09):

```cpp
// Already in TrainingConfig:
int eval_interval = 1000;     // Training steps between eval runs
int eval_games = 200;         // Games per eval run
```

---

## 2. Evaluation Runner

### 2.1 Core Function

```cpp
// src/eval/evaluator.h

struct EvalResult {
    int total_games;
    int nn_wins;
    int heuristic_wins;
    int draws;
    int nn_wins_as_p0;      // NN wins when playing as player 0
    int nn_wins_as_p1;      // NN wins when playing as player 1
    int games_as_p0;         // Total games where NN was player 0
    int games_as_p1;         // Total games where NN was player 1
    double avg_duel_margin;  // Average duel point margin across all games

    double nn_win_rate() const {
        return total_games > 0
            ? static_cast<double>(nn_wins) / total_games : 0.0;
    }
    double nn_win_rate_as_p0() const {
        return games_as_p0 > 0
            ? static_cast<double>(nn_wins_as_p0) / games_as_p0 : 0.0;
    }
    double nn_win_rate_as_p1() const {
        return games_as_p1 > 0
            ? static_cast<double>(nn_wins_as_p1) / games_as_p1 : 0.0;
    }
};

/// Run evaluation: play games between the NN bot and heuristic bot.
/// Runs synchronously on the calling thread (intended for the training thread).
///
/// Games alternate which player the NN controls:
/// even-numbered games: NN = player 0, heuristic = player 1
/// odd-numbered games: NN = player 1, heuristic = player 0
///
/// @param model The current model (used directly for inference, no async queue)
/// @param tables Precomputed tables
/// @param num_games Number of games to play
/// @param base_seed RNG seed for reproducibility
/// @return Aggregated evaluation results
EvalResult run_evaluation(ProYamsNet& model, torch::Device device,
                           const PrecomputedTables& tables,
                           int num_games, uint64_t base_seed);
```

### 2.2 Single Eval Game

```cpp
/// Play one evaluation game between NN and heuristic bot.
///
/// @param model The model for NN inference
/// @param device GPU device for inference
/// @param tables Precomputed tables
/// @param nn_player Which player the NN controls (0 or 1)
/// @param rng Random engine for this game
/// @param out_duel_margin Output: raw duel point margin from NN's perspective
/// @return 1.0 if NN wins, 0.0 if heuristic wins, 0.5 if draw
double play_eval_game(ProYamsNet& model, torch::Device device,
                       const PrecomputedTables& tables,
                       int nn_player, RNG& rng,
                       int& out_duel_margin);
```

**Implementation:**

```
GameState state
GameContext ctx
SolverBuffers buffers
SolverConfig greedy_config = { temperature=0, exploration=false }

init_game(state, ctx, rng)

while !is_terminal(state.board):
    player = state.board.current_player

    if player == nn_player:
        // NN bot turn: use NN inference for afterstate evaluation
        nn_play_turn(model, device, state, ctx, tables, buffers,
                     greedy_config, rng)
    else:
        // Heuristic bot turn
        heuristic_play_turn(state, ctx, tables, buffers, rng)

duel_points = compute_duel(state.board, ctx)
// Convert to NN's perspective
if nn_player == 1:
    duel_points = -duel_points

out_duel_margin = duel_points
return get_game_result_for_player(state, ctx, nn_player)
```

### 2.3 NN Turn (Synchronous Inference)

During evaluation, NN inference runs synchronously — no async queue. The training thread calls the model directly.

```cpp
/// Play one complete turn using NN inference (synchronous, for evaluation).
///
/// @param model The model for inference
/// @param device GPU device
/// @param state Game state (modified)
/// @param ctx Game context (modified)
/// @param tables Precomputed tables
/// @param buffers Solver buffers (reused)
/// @param tensor_buffer Heap-allocated buffer for afterstate tensors (kMaxAfterstates * kTensorSize floats)
/// @param config Solver config (greedy, no exploration)
/// @param rng Random engine
void nn_play_turn(ProYamsNet& model, torch::Device device,
                   GameState& state, GameContext& ctx,
                   const PrecomputedTables& tables,
                   SolverBuffers& buffers,
                   std::vector<float>& tensor_buffer,
                   const SolverConfig& config, RNG& rng);
```

**Implementation:**

Note: The tensor buffer is large (~1.6MB) and must NOT be allocated on the stack (risk of stack overflow). It is passed as a parameter, allocated once per evaluation run on the heap.

```
// tensor_buffer is heap-allocated, passed in by the caller.
// Allocated once per eval run: std::vector<float> tensor_buffer(kMaxAfterstates * kTensorSize)

while true:
    // Step 1: Get afterstate requests
    solver_get_requests(state, ctx, tables, buffers)

    // Step 2: Generate tensors for all requests
    generate_tensor_batch(state.board, ctx, state.board.current_player,
                         buffers.requests, buffers.request_count,
                         tables, tensor_buffer.data())

    // Step 3: Run NN inference synchronously
    // Wrap as libtorch tensor, forward pass, extract results
    {
        torch::NoGradGuard no_grad;
        auto input = torch::from_blob(tensor_buffer.data(),
                        {buffers.request_count, kTensorSize},
                        torch::kFloat32).to(device);
        auto output = model.forward(input).to(torch::kCPU);
        auto accessor = output.accessor<float, 2>();
        for (int i = 0; i < buffers.request_count; ++i) {
            buffers.evs[i] = static_cast<double>(accessor[i][0]);
        }
    }

    // Step 4: Resolve
    result = solver_resolve(state, ctx, tables, buffers)

    if result.should_place:
        perform_placement(result.placement, result.score, state, ctx, rng)
        return
    else:
        if !can_reroll(state, ctx): assert(false)
        perform_hold(result.hold_mask, state, rng)
        // Loop: new dice, re-evaluate
```

### 2.4 Running Full Evaluation

```cpp
EvalResult run_evaluation(ProYamsNet& model, torch::Device device,
                           const PrecomputedTables& tables,
                           int num_games, uint64_t base_seed) {
    EvalResult result = {};
    result.total_games = num_games;

    model.eval();  // Set to evaluation mode (disables dropout if any)

    double margin_sum = 0.0;

    for (int i = 0; i < num_games; ++i) {
        RNG rng(base_seed + i);
        int nn_player = i % 2;  // Alternate sides

        if (nn_player == 0) result.games_as_p0++;
        else result.games_as_p1++;

        int duel_margin = 0;
        double outcome = play_eval_game(model, device, tables,
                                         nn_player, rng, duel_margin);

        margin_sum += duel_margin;

        if (outcome == 1.0) {
            result.nn_wins++;
            if (nn_player == 0) result.nn_wins_as_p0++;
            else result.nn_wins_as_p1++;
        } else if (outcome == 0.0) {
            result.heuristic_wins++;
        } else {
            result.draws++;
        }
    }

    result.avg_duel_margin = num_games > 0
        ? margin_sum / num_games : 0.0;

    return result;
}
```

---

## 3. Integration with Training Loop

### 3.1 maybe_evaluate() Implementation

In `TrainingLoop` (from Task 09):

```cpp
void TrainingLoop::maybe_evaluate() {
    if (trainer_.training_step() % config_.eval_interval != 0) return;

    // Run evaluation using the current training model
    EvalResult eval = run_evaluation(
        *trainer_.model_ptr(), trainer_.device(),
        tables_, config_.eval_games,
        trainer_.training_step());  // Use step as seed for reproducibility

    // Log evaluation results
    log_evaluation(config_.log_dir, trainer_.training_step(), eval);

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.latest_eval_win_rate = eval.nn_win_rate();
    }
}
```

### 3.2 Updated Training Metrics

Add evaluation tracking to `TrainingMetrics`:

```cpp
// Add to TrainingMetrics:
double latest_eval_win_rate = 0.0;
```

---

## 4. Evaluation Logging

### 4.1 Separate Eval Log File

Evaluation results are logged to a separate CSV for easy analysis:

`log_dir/eval_log.csv`:

```csv
timestamp,step,games,nn_wins,heur_wins,draws,win_rate,wr_as_p0,wr_as_p1,avg_margin
2026-04-02T11:00:00,1000,200,87,110,3,0.435,0.44,0.43,12450
2026-04-02T11:15:00,2000,200,142,56,2,0.710,0.72,0.70,28930
```

### 4.2 Log Function

```cpp
/// Append evaluation results to the eval log CSV.
void log_evaluation(const std::string& log_dir, int training_step,
                     const EvalResult& result);
```

Creates the file with headers on first call, appends rows on subsequent calls.

---

## 5. Helper: Game Result for Specific Player

```cpp
/// Get game result from a specific player's perspective.
/// @param nn_player Which player to evaluate for (0 or 1)
/// @return 1.0 if nn_player wins, 0.0 if they lose, 0.5 if draw
double get_game_result_for_player(const GameState& state,
                                   const GameContext& ctx,
                                   int player);
```

**Implementation:**

```cpp
double get_game_result_for_player(const GameState& state,
                                   const GameContext& ctx,
                                   int player) {
    int duel_points = compute_duel(state.board, ctx);
    // duel_points is from player 0's perspective
    if (player == 1) duel_points = -duel_points;
    if (duel_points > 0) return 1.0;
    if (duel_points < 0) return 0.0;
    return 0.5;
}
```

---

## 6. File Organization

```
src/eval/
├── evaluator.h              # EvalResult, run_evaluation, play_eval_game,
│                            # nn_play_turn, get_game_result_for_player
├── evaluator.cc             # Evaluation implementation
├── eval_logging.h           # log_evaluation
├── eval_logging.cc          # Eval log CSV writing
└── CMakeLists.txt
```

```cmake
# src/eval/CMakeLists.txt
add_library(eval STATIC
    evaluator.cc
    eval_logging.cc
)
target_include_directories(eval PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(eval PUBLIC engine solver heuristic model)
```

---

## 7. Unit Tests

### 7.1 Eval Game Tests (`tests/eval/eval_game_test.cc`)

**NN vs heuristic completes:**
- Create a random NN model. Play one eval game (NN as player 0). Verify game completes without assertions, result is valid (0.0, 0.5, or 1.0).
- Same with NN as player 1.

**Greedy play (no exploration):**
- Play the same eval game twice with the same seed and same model. Verify results are identical (deterministic greedy play).

**Duel margin tracking:**
- Play an eval game. Verify duel_margin is positive when NN wins, negative when heuristic wins.

### 7.2 Evaluation Runner Tests (`tests/eval/evaluator_test.cc`)

**Alternating sides:**
- Run evaluation with 10 games. Verify games_as_p0 == 5 and games_as_p1 == 5.

**Win rate computation:**
- Set up known results. Verify nn_win_rate(), nn_win_rate_as_p0(), nn_win_rate_as_p1() compute correctly.

**Heuristic baseline:**
- Run evaluation with a randomly initialized NN (untrained). Win rate should be low (heuristic should win most games against a random value function). This establishes the baseline that training should improve from.

**Full evaluation run:**
- Run 50 eval games. Verify:
  - Total games = 50
  - nn_wins + heuristic_wins + draws = 50
  - No assertion failures
  - Results logged to eval CSV

### 7.3 Eval Logging Tests (`tests/eval/eval_logging_test.cc`)

**CSV output:**
- Run log_evaluation multiple times. Read CSV, verify headers and row format.
- Verify step numbers are increasing.

---

## 8. Benchmarks

Add to `tests/benchmarks/eval_bench.cc`:

- **BM_SingleEvalGame:** Benchmark one NN vs heuristic game. Measures the cost of synchronous NN inference during evaluation.
- **BM_FullEvaluation:** Benchmark a full 200-game evaluation run. This determines how much time each eval interval costs during training.

Target: 200 eval games should complete in under 30 seconds. If it takes longer, we can reduce eval_games or increase eval_interval.

---

## 9. Definition of Done

This task is complete when:

1. `play_eval_game` correctly plays a complete game between NN and heuristic bot with greedy play.
2. `nn_play_turn` performs synchronous NN inference without using the async queue.
3. `run_evaluation` plays the configured number of games, alternating which side the NN plays.
4. Win rates are computed correctly overall and per-side (player 0 vs player 1).
5. Average duel margin is tracked across all games.
6. Evaluation results are logged to a separate CSV file.
7. `maybe_evaluate` integrates correctly with the training loop at configured intervals.
8. A randomly initialized NN loses most eval games against the heuristic (sanity check).
9. Evaluation is deterministic given the same model and seed.
10. All unit tests pass.
11. Benchmarks confirm evaluation completes in reasonable time.
