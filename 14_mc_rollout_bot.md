# Task 14: MC Rollout Bot (Nice-to-Have)

## Overview

Build an enhanced bot that uses Monte Carlo rollouts to improve placement decisions beyond what the basic NN+solver provides. Given a configurable time budget per turn (e.g., 2 seconds), the bot simulates many possible continuations of the game to refine its value estimates.

This is the strongest bot variant — it uses the NN as a fast evaluator within a deeper search. It's intended for evaluation against humans and as the ultimate strength benchmark, not for self-play training (too slow).

## Prerequisites

- Task 05 completed (solver, heuristic bot)
- Task 07 completed (NN model, inference)
- Task 10 completed (nn_play_turn for synchronous NN inference)

---

## 1. Concept

The basic NN+solver bot evaluates afterstates with a single NN forward pass — it asks "how good is this board position?" and trusts the answer. The MC rollout bot goes deeper: after evaluating an afterstate, it simulates the rest of the game multiple times (using the NN+solver for both players) and uses the actual win/loss outcomes to refine the value estimate.

This is similar to how AlphaZero uses MCTS to improve on the raw policy/value network — the search adds strength on top of the learned evaluation.

### 1.1 Why This Helps

The NN's value estimate V(s) is an approximation. It might have blind spots — positions where it over- or under-estimates win probability. By actually playing out the game from that position many times, we get an empirical win rate that's more accurate (at the cost of computation time).

The more rollouts per position, the more accurate the estimate, but with diminishing returns. The time budget controls this tradeoff.

### 1.2 When to Use

- **Human vs MC bot:** The strongest opponent available for testing
- **Evaluation:** Measuring how much search adds over raw NN evaluation
- **NOT for training:** Far too slow for self-play (one turn takes seconds instead of milliseconds)

---

## 2. Algorithm

### 2.1 Turn-Level Flow

```
MC bot's turn begins (dice are rolled):

1. Run the normal solver to get afterstate requests and NN evaluations
   (same as the basic NN+solver bot)

2. Get the solver's top-K candidate actions (placements or hold+reroll)
   ranked by NN-based EV

3. For each candidate action:
   a. Apply the action to get an afterstate
   b. Simulate N complete games from that afterstate
      (both players use the basic NN+solver bot, greedy play)
   c. Record win rate across the N simulations

4. Pick the action with the highest empirical win rate

5. If the best action is "hold and reroll":
   Apply the hold, get new dice, go back to step 1
   (with reduced time budget for remaining decisions this turn)

6. If the best action is "place":
   Apply the placement, turn is over
```

### 2.2 Time Budget Management

The bot has a total time budget per turn (e.g., 2 seconds). It needs to allocate this across:
- Potentially multiple hold/reroll decisions (up to 3 roll phases per turn)
- Multiple candidate actions per decision
- Multiple rollouts per candidate

Strategy: allocate time proportionally. If the first roll has 2 rerolls remaining, use ~40% of the budget on the first decision, ~35% on the second, ~25% on the third (if reached). This accounts for the fact that early decisions are more impactful.

```cpp
struct MCTimeAllocation {
    double total_budget_ms;
    double elapsed_ms;

    double budget_for_current_decision(int rolls_left) const {
        double remaining = total_budget_ms - elapsed_ms;
        if (rolls_left == 0) return remaining;  // Must decide now
        // Allocate proportionally: more for earlier decisions
        double weights[] = {1.0, 0.6, 0.4};  // rolls_left = 0, 1, 2
        double total_weight = 0;
        for (int r = 0; r <= rolls_left; ++r) total_weight += weights[r];
        return remaining * weights[rolls_left] / total_weight;
    }
};
```

### 2.3 Candidate Selection

Rather than simulating all possible actions (too many), select the top-K candidates from the solver's ranking:

```cpp
struct MCConfig {
    double time_budget_ms = 2000.0;  // Total time per turn
    int max_candidates = 5;           // Top-K actions to evaluate with rollouts
    int min_rollouts_per_candidate = 10;  // Minimum rollouts before comparing
    int rollout_batch_size = 4;       // Rollouts to run before checking time
};
```

**Top-K selection for placements:**
After `solver_resolve`, we have the EV for each possible placement. Take the top `max_candidates` by EV. These are the most promising actions — rollouts determine which is truly best.

**Top-K selection for hold masks:**
The solver evaluates all 32 hold masks. Take the top `max_candidates` masks by expected value.

**Including "stop and place" as a candidate:**
When the bot has rerolls remaining, "stop and place now" competes with hold masks. The best placement (from V0) is always included as one of the candidates.

---

## 3. Rollout Simulation

### 3.1 Single Rollout

A rollout plays the game from an afterstate to completion using the basic NN+solver bot for both players:

```cpp
/// Simulate a complete game from the given afterstate.
/// Both players use the NN+solver bot with greedy play (no exploration).
///
/// @param board The afterstate board (after the candidate action was applied)
/// @param ctx Game context for the afterstate (can be approximate)
/// @param mc_player Which player the MC bot is (for result interpretation)
/// @param model NN model for inference
/// @param device GPU device
/// @param tables Precomputed tables
/// @param rng Random engine (each rollout uses a unique seed)
/// @return 1.0 if mc_player wins, 0.0 if they lose, 0.5 if draw
double simulate_rollout(const BoardState& board, const GameContext& ctx,
                         int mc_player,
                         ProYamsNet& model, torch::Device device,
                         const PrecomputedTables& tables,
                         RNG& rng);
```

**Implementation:**

```
// Create a fresh game state from the afterstate
GameState sim_state
sim_state.board = board  // Copy the afterstate
// Rebuild context from board (since we don't have incrementally maintained ctx)
GameContext sim_ctx
rebuild_context_from_board(sim_state.board, sim_ctx)

// Roll dice for the current player
roll_dice(sim_state, rng)

SolverBuffers buffers

// Play to completion
while !is_terminal(sim_state.board):
    nn_play_turn(model, device, sim_state, sim_ctx, tables,
                 buffers, greedy_config, rng)

return get_game_result_for_player(sim_state, sim_ctx, mc_player)
```

### 3.2 Context Rebuilding

When starting a rollout from a cloned afterstate, we need a valid `GameContext`. Since the afterstate is a clone with one cell modified, we can't use the incrementally maintained context from the real game. We need to rebuild it from scratch:

```cpp
/// Rebuild a complete GameContext from a BoardState.
/// Scans all cells to reconstruct golden_max, upper_sum, scratch status,
/// clean column eligibility, legal placements, etc.
/// Slower than incremental updates but only called once per rollout.
void rebuild_context_from_board(const BoardState& board, GameContext& ctx);
```

This function iterates all cells and reconstructs all cached state. It's O(156) — trivial cost compared to a full game simulation.

### 3.3 Batched Rollouts

To improve GPU utilization, we can run multiple rollouts concurrently. However, since each rollout involves many sequential NN inference calls (one per turn for 78+ turns), true parallelism requires multiple CPU threads per MC decision.

Simpler approach: run rollouts sequentially but batch the NN inference within each rollout's solver calls (already handled by `nn_play_turn`). The GPU still processes batched tensors per solver call.

For the time-budgeted approach, sequential rollouts work well — we run rollouts in groups of `rollout_batch_size`, check the elapsed time after each group, and stop when the budget is exhausted.

---

## 4. MC Turn Implementation

```cpp
// src/solver/mc_bot.h

/// Play one turn using MC rollout-enhanced decisions.
///
/// @param state Game state (modified: hold/placement applied)
/// @param ctx Game context (modified)
/// @param model NN model
/// @param device GPU device
/// @param tables Precomputed tables
/// @param mc_config MC configuration (time budget, candidates, etc.)
/// @param rng Random engine
void mc_play_turn(GameState& state, GameContext& ctx,
                   ProYamsNet& model, torch::Device device,
                   const PrecomputedTables& tables,
                   const MCConfig& mc_config, RNG& rng);
```

**Implementation:**

```
MCTimeAllocation time_alloc = { mc_config.time_budget_ms, 0.0 }
SolverBuffers buffers
SolverConfig greedy_config = { temperature=0, exploration=false }

while true:
    auto start = high_resolution_clock::now()
    double decision_budget = time_alloc.budget_for_current_decision(state.rolls_left)

    // Step 1: Get NN-based evaluations (same as basic NN bot)
    solver_get_requests(state, ctx, tables, buffers)
    
    // Generate tensors and run NN inference
    float tensor_buf[kMaxAfterstates * kTensorSize]
    generate_tensor_batch(state.board, ctx, state.board.current_player,
                         buffers.requests, buffers.request_count,
                         tables, tensor_buf)
    // ... run batch_inference synchronously ...
    
    // Step 2: Get solver's ranking of all actions
    // Collect candidate actions: top-K placements + top-K hold masks + "stop" option
    
    CandidateAction candidates[mc_config.max_candidates]
    int num_candidates = select_top_candidates(
        state, ctx, tables, buffers, mc_config.max_candidates, candidates)
    
    // Step 3: Rollout evaluation for each candidate
    for each candidate:
        candidate.win_count = 0
        candidate.rollout_count = 0
    
    double elapsed_for_decision = 0.0
    while elapsed_for_decision < decision_budget:
        for each candidate:
            // Run a batch of rollouts
            for r in 0..rollout_batch_size:
                BoardState rollout_board = apply_candidate(state.board, candidate)
                RNG rollout_rng(rng.next())  // Unique seed per rollout
                double outcome = simulate_rollout(
                    rollout_board, ctx, state.board.current_player,
                    model, device, tables, rollout_rng)
                candidate.win_count += outcome
                candidate.rollout_count++
        
        elapsed_for_decision = elapsed_since(start)
        
        // Early termination: if one candidate is clearly best
        // (statistical significance test)
        if can_terminate_early(candidates, num_candidates):
            break
    
    // Step 4: Pick best candidate by empirical win rate
    best = candidate with highest (win_count / rollout_count)
    
    // Update time tracking
    time_alloc.elapsed_ms += elapsed_for_decision
    
    // Step 5: Apply action
    if best.is_placement:
        perform_placement(best.placement, best.score, state, ctx, rng)
        return
    else:
        perform_hold(best.hold_mask, state, rng)
        // Continue loop for next roll decision
```

### 4.1 Candidate Action Structure

```cpp
struct CandidateAction {
    bool is_placement;       // true = place, false = hold+reroll
    Placement placement;     // Only valid if is_placement
    int8_t score;            // Only valid if is_placement
    uint8_t hold_mask;       // Only valid if !is_placement
    double nn_ev;            // NN-based expected value (for initial ranking)
    double win_count;        // Accumulated wins from rollouts
    int rollout_count;       // Number of rollouts completed
    
    double empirical_wr() const {
        return rollout_count > 0 ? win_count / rollout_count : 0.0;
    }
};
```

### 4.2 Early Termination

If one candidate has a statistically significant lead over all others, we can stop rolling out early and save time. A simple approach: if the best candidate's win rate lower bound (using a confidence interval) exceeds the second-best's upper bound, terminate.

```cpp
/// Check if the leading candidate is statistically significantly better
/// than all others (using Wilson score confidence interval at 95%).
bool can_terminate_early(const CandidateAction* candidates, int count);
```

This avoids wasting rollouts on decisions where the best action is already clear.

---

## 5. Integration with UI

The MC bot is available as a player type in the UI (Task 12):

```cpp
// Add to PlayerType enum:
kMCRollout   // MC rollout-enhanced NN bot
```

The session manager creates an MC bot when `"mc"` is specified as a player type:

```json
{
    "player0": "human",
    "player1": "mc"
}
```

The MC bot's turn takes noticeably longer (up to 2 seconds), so the UI should show a "thinking..." indicator during the bot's turn.

### 5.1 UI Updates

- Add `"mc"` as a player type option in the new game dialog
- Show a progress indicator during MC bot turns
- Optionally show the MC bot's candidate evaluation (win rates per candidate) for educational purposes

---

## 6. File Organization

```
src/solver/
├── ... (existing files)
├── mc_bot.h                 # MCConfig, CandidateAction, mc_play_turn
├── mc_bot.cc                # MC rollout implementation
└── mc_candidates.h          # select_top_candidates, can_terminate_early
    mc_candidates.cc
```

Update `src/solver/CMakeLists.txt`:

```cmake
add_library(solver STATIC
    precomputed_tables.cc
    solver.cc
    mc_bot.cc
    mc_candidates.cc
)
target_include_directories(solver PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(solver PUBLIC engine model)
```

Note: the solver library now depends on `model` because the MC bot calls NN inference directly. This dependency only exists for the MC bot — the basic solver and heuristic bot remain model-free.

---

## 7. Context Rebuilding

Add to the engine library:

```cpp
// src/engine/board_init.h (or context_rebuild.h)

/// Rebuild a complete GameContext from a BoardState by scanning all cells.
/// Used by MC rollouts which start from a cloned board position.
void rebuild_context_from_board(const BoardState& board, GameContext& ctx);
```

**Implementation:**

```
Zero all context fields

For each player p:
    For each column c:
        For each row r:
            cell = board.cells[p][c][r]
            if cell == kCellEmpty: continue
            
            // Golden max
            if cell > ctx.golden_max[c][r]:
                ctx.golden_max[c][r] = cell
            
            // Upper sum
            if r <= 5:
                ctx.upper_sum[p][c] += cell
            
            // SS/LS scratch
            if r == 6 and cell == 0: ctx.ss_scratched[p][c] = true
            if r == 7 and cell == 0: ctx.ls_scratched[p][c] = true
            
            // Lower scratch
            if r >= 6 and cell == 0: ctx.lower_has_scratch[p][c] = true

// Rebuild legal placements
For each player p:
    rebuild_legal_placements(p, board, ctx)

// Non-turbo cells remaining
For each player p:
    count = 0
    For each non-turbo column c:
        For each row r:
            if board.cells[p][c][r] == kCellEmpty: count++
    ctx.non_turbo_cells_remaining[p] = count
```

---

## 8. Unit Tests

### 8.1 MC Bot Tests (`tests/solver/mc_bot_test.cc`)

**Single rollout:**
- Run one rollout from a mid-game position. Verify it completes and returns a valid result (0.0, 0.5, or 1.0).

**Multiple rollouts consistency:**
- Run 100 rollouts from the same position with different seeds. Verify win rate is between 0 and 1. Verify deterministic seeds produce deterministic results.

**Time budget enforcement:**
- Set time budget to 500ms. Run mc_play_turn. Verify it completes within 600ms (budget + overhead tolerance).
- Set time budget to 100ms. Verify it completes quickly with fewer rollouts.

**Candidate selection:**
- Set up a position with a clearly dominant action (e.g., filling Yams with 100). Verify the MC bot selects it, even with few rollouts.

**Early termination:**
- Set up a position with one clearly superior candidate. Verify early termination fires and saves time.

### 8.2 Context Rebuild Tests (`tests/engine/context_rebuild_test.cc`)

**Round-trip consistency:**
- Play a game to mid-point using normal incremental context updates. Clone the board. Rebuild context from the clone. Verify all fields match the incrementally maintained context:
  - golden_max
  - upper_sum
  - ss_scratched, ls_scratched
  - lower_has_scratch
  - legal_placements (both all and no_turbo)
  - non_turbo_cells_remaining

### 8.3 MC vs NN Comparison Tests (`tests/solver/mc_vs_nn_test.cc`)

**MC at least as good as raw NN:**
- Play 50 games: MC bot vs heuristic bot. Play 50 games: basic NN bot vs heuristic bot (same checkpoints, same seeds). Verify MC win rate >= NN win rate (MC should be at least as strong since it uses NN as a base and adds search).

---

## 9. Benchmarks

Add to `tests/benchmarks/mc_bench.cc`:

- **BM_SingleRollout:** Benchmark one complete rollout simulation from a mid-game position.
- **BM_ContextRebuild:** Benchmark rebuilding GameContext from a BoardState.
- **BM_MCTurn:** Benchmark one MC bot turn with various time budgets (100ms, 500ms, 2000ms). Report number of rollouts completed within each budget.

---

## 10. Definition of Done

This task is complete when:

1. `simulate_rollout` plays a complete game from an afterstate and returns correct win/loss/draw.
2. `rebuild_context_from_board` produces identical results to incrementally maintained context.
3. `mc_play_turn` selects actions using rollout-refined win rates within the time budget.
4. Time budget is respected (bot doesn't exceed budget by more than ~100ms).
5. Early termination works when one candidate is statistically dominant.
6. The MC bot is selectable in the UI as player type `"mc"`.
7. MC bot win rate vs heuristic is at least as high as basic NN bot win rate vs heuristic.
8. All unit tests pass.
9. Benchmarks establish rollout throughput and rollouts-per-budget metrics.
