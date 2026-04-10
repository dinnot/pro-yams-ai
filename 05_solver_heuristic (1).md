# Task 05: Expectimax Solver with Heuristic Evaluation

## Overview

Build the expectimax solver that determines optimal actions (hold/reroll or place) given a game state. The solver uses a two-step API: first it declares which afterstates need evaluation, then it consumes the evaluations to compute the best action via dynamic programming.

This task implements the solver with the heuristic evaluation function (score × coefficient) as the initial value function, producing the first playable bot. Task 08 will integrate NN evaluation using the same solver API.

## Prerequisites

- Task 02 completed (engine data structures, scoring)
- Task 03 completed (game flow, legal placements)
- Task 04 completed (precomputed tables, transitions)

---

## 1. Solver Data Types

### 1.1 Afterstate Request

A request for the caller to evaluate a hypothetical placement:

```cpp
// src/solver/solver.h

struct AfterstateRequest {
    Placement placement;    // (column, row) to fill
    int8_t score;           // Score to place (0 = scratch)
};
```

### 1.2 Solver Result

The solver's recommended action:

```cpp
struct SolverResult {
    bool should_place;       // true = place now, false = hold and reroll
    uint8_t hold_mask;       // Which dice to hold (only valid if should_place == false)
    Placement placement;     // Where to place (only valid if should_place == true)
    int8_t score;            // Score to place (only valid if should_place == true)
    double expected_value;   // EV of the best action
};
```

### 1.3 Static Buffers

Pre-allocated buffers for afterstate requests and EVs. These avoid heap allocation on every solver call.

```cpp
// Maximum number of afterstate requests per solver call.
// Worst case: 78 legal cells × max distinct scores per cell.
// Upper section rows have at most 5 distinct scores + scratch = 6.
// FH has 26 distinct scores + scratch = 27.
// Realistically, with Golden Rule filtering, far fewer.
// 4096 is a generous upper bound that will never be exceeded.
constexpr int kMaxAfterstateRequests = 4096;

struct SolverBuffers {
    AfterstateRequest requests[kMaxAfterstateRequests];
    int request_count;

    double evs[kMaxAfterstateRequests];

    // Internal DP tables used by solver_resolve
    double v0[kNumDiceStates];           // Value at 0 rerolls left
    double v1[kNumDiceStates];           // Value at 1 reroll left

    // For each dice state at layer 0: which request index is the best placement?
    int16_t best_request_idx[kNumDiceStates];

    // For each dice state at layer 1: best hold mask
    int16_t best_mask_v1[kNumDiceStates];

    // "Stop" value at each dice state (best value if placing immediately)
    double stop_value[kNumDiceStates];
    int16_t stop_request_idx[kNumDiceStates];
};
```

`SolverBuffers` is allocated once per game thread and reused across all solver calls within that thread. Zero per-call allocation.

---

## 2. Solver API

### 2.1 Step 1: Get Afterstate Requests

```cpp
/// Determine which afterstates need evaluation.
/// Populates buffers.requests and sets buffers.request_count.
///
/// For each legal placement (column, row), enumerates all distinct scores
/// that could be placed there (filtered by Golden Rule minimum),
/// plus scratch (score = 0). Each unique (column, row, score) becomes
/// one request.
///
/// @param state Current game state (dice + rolls_left + board)
/// @param ctx Current game context (legal placements, golden rule, etc.)
/// @param tables Precomputed tables
/// @param buffers Output: populated with afterstate requests
void solver_get_requests(const GameState& state, const GameContext& ctx,
                         const PrecomputedTables& tables,
                         SolverBuffers& buffers);
```

**Implementation:**

```
request_count = 0

For each legal placement (column, row) from get_legal_placements(player, rolls_left, ctx):

    // Always add scratch option
    requests[request_count++] = {(column, row), 0}

    // Get the Golden Rule minimum for this cell
    min_threshold = ctx.golden_max[column][row]
    
    // Handle SS/LS constraints
    If row == SS:
        If ctx.ls_scratched[player][column]: skip non-zero scores (forced scratch)
        If own LS is filled: cap possible scores below LS value
    If row == LS:
        If ctx.ss_scratched[player][column]: skip non-zero scores (forced scratch)
        If own SS is filled: only scores strictly greater than SS value

    // Get filtered possible scores for this row above the threshold
    count = 0
    scores = get_filtered_scores(row, min_threshold, count, tables)
    
    For each score in filtered scores:
        // Apply SS/LS constraints
        If row == SS and score doesn't satisfy SS constraints: skip
        If row == LS and score doesn't satisfy LS constraints: skip
        
        requests[request_count++] = {(column, row), score}
```

**Important:** The scratch option (score = 0) bypasses the Golden Rule — you can always scratch. It also needs to account for SS/LS mutual destruction (scratching SS means LS will be forced to scratch too, and vice versa). The caller's evaluation function must handle this when computing the afterstate value.

**Deduplication:** Different legal cells might produce the same afterstate if they have the same (column, row, score). This shouldn't happen since each (column, row) pair is unique in the legal placements list. However, the same score value might appear for different cells — that's fine, they're different afterstates because they're in different cells.

### 2.2 Step 2: Resolve Best Action

```cpp
/// Given evaluations for all requested afterstates, compute the optimal action
/// using expectimax dynamic programming over reroll layers.
///
/// @param state Current game state
/// @param ctx Current game context  
/// @param tables Precomputed tables
/// @param buffers Input: requests and evs populated. Internal DP tables used.
/// @return The best action (hold mask or placement) and its expected value
SolverResult solver_resolve(const GameState& state, const GameContext& ctx,
                            const PrecomputedTables& tables,
                            SolverBuffers& buffers);
```

---

## 3. The Expectimax Algorithm

### 3.1 Overview

The solver works in layers from bottom (0 rerolls) upward to the current number of rerolls:

1. **Build V0:** For each of the 252 dice states, determine the best placement and its EV assuming no rerolls remain.
2. **Build V1:** For each dice state, compare stopping (using V0) vs each hold mask (expected value over transitions into V0).
3. **If rolls_left >= 2, build V2:** For each hold mask applied to the current dice, compute expected value over transitions into V1. Compare with stopping (using V0).
4. **Return** the best action at the current layer for the current dice.

### 3.2 Layer 0: No Rerolls Left (V0)

For each of the 252 dice states, find the best placement:

```
For each dice_state_id (0 to 251):
    best_ev = -infinity
    best_req = -1
    
    For each request (i = 0 to request_count-1):
        // Check if this dice state can achieve this score in this row
        actual_dice_score = tables.score_tables.dice_score[dice_state_id][request.row]
        
        If request.score == 0:
            // Scratch is always achievable
            If evs[i] > best_ev:
                best_ev = evs[i]
                best_req = i
        Else if actual_dice_score == request.score:
            // This dice state produces exactly this score for this row
            If evs[i] > best_ev:
                best_ev = evs[i]
                best_req = i
    
    v0[dice_state_id] = best_ev
    best_request_idx[dice_state_id] = best_req
```

**Critical correctness point:** When checking `actual_dice_score == request.score`, we're matching the raw dice score against the requested score. This works because `solver_get_requests` already filtered scores through the Golden Rule and SS/LS constraints — if a score is in the request list, it's achievable and valid. The dice_score table tells us which dice states can produce that exact score.

**Edge case:** For number rows (1s–6s), a dice state might produce a score that's not in the filtered list (below Golden Rule threshold). That's fine — the request for that score was never generated, so the solver naturally won't consider it.

**Edge case:** If no request matches a given dice state (all possible placements for those dice were filtered out), `best_ev` stays at -infinity. However, scratch (score = 0) is always in the request list for every legal cell, and every dice state can scratch. So `best_ev` will always be set to at least the best scratch EV.

### 3.3 Stop Value

At any reroll layer, the player can choose to stop rerolling and place immediately with the current dice. The "stop value" for the current dice is the same as V0 for the current dice state:

```
current_dice_state_id = get_dice_state_id(state.dice, tables)
stop_value = v0[current_dice_state_id]
stop_request = best_request_idx[current_dice_state_id]
```

### 3.4 Layer 1: One Reroll Left (V1)

For each of the 252 dice states, find the best hold mask vs stopping:

```
For each dice_state_id (0 to 251):
    best_ev = v0[dice_state_id]     // Stop value (place with current dice)
    best_mask = -1                   // -1 means "stop"
    
    For each mask (0 to 31):
        held_id = tables.moves[dice_state_id][mask]
        count = 0
        transitions = get_transitions(held_id, count, tables)
        
        ev = 0.0
        For each transition (t) in transitions:
            ev += t.probability * v0[t.target_state_id]
        
        If ev > best_ev:
            best_ev = ev
            best_mask = mask
    
    v1[dice_state_id] = best_ev
    best_mask_v1[dice_state_id] = best_mask
```

### 3.5 Layer 2: Two Rerolls Left

Only computed for the current dice state (not all 252, since we just need the answer for our actual dice):

```
current_id = get_dice_state_id(state.dice, tables)

best_ev = v0[current_id]     // Stop value
best_action_is_place = true
best_mask = 0

For each mask (0 to 31):
    held_id = tables.moves[current_id][mask]
    count = 0
    transitions = get_transitions(held_id, count, tables)
    
    ev = 0.0
    For each transition (t) in transitions:
        ev += t.probability * v1[t.target_state_id]
    
    If ev > best_ev:
        best_ev = ev
        best_mask = mask
        best_action_is_place = false

// Build result
If best_action_is_place:
    return placement from best_request_idx[current_id]
Else:
    return hold_mask = best_mask
```

### 3.6 Handling rolls_left == 0

If `rolls_left == 0`, the player must place immediately. Skip all reroll logic:

```
current_id = get_dice_state_id(state.dice, tables)
// Just return the best placement from V0
return placement from best_request_idx[current_id]
```

### 3.7 Turbo Column Interaction

When `rolls_left == 0`, Turbo cells are already excluded from `get_legal_placements` (handled by the engine in Task 03). So `solver_get_requests` naturally won't include Turbo cells in the request list when `rolls_left == 0`.

For the V0 calculation across all 252 dice states, some of those states correspond to scenarios where the player has used all rolls — but the request list is the same regardless (it was generated once for the current board state). The Turbo exclusion at `rolls_left == 0` only matters for the "stop" decision and when `rolls_left` is actually 0 for the current turn.

Wait — there's a subtlety. The request list is generated based on the current `rolls_left`. But V0 represents "what if these are my final dice" — which means `rolls_left` would be 0 at that point. If the request list was generated with `rolls_left > 0` (including Turbo cells), then V0 correctly includes Turbo as an option because the player chose to stop with rolls remaining.

The only case where Turbo should be excluded is when `rolls_left` is already 0 (the player has no choice — they used all their rolls). This is already handled by `get_legal_placements`. So the current design is correct.

---

## 4. Heuristic Value Function

The heuristic evaluation function computes afterstate values without NN inference:

```cpp
/// Compute heuristic EVs for a batch of afterstate requests.
/// Heuristic: score × coefficient for the placed cell.
///
/// @param board The current board state (before placement)
/// @param ctx The current game context
/// @param requests The afterstate requests
/// @param request_count Number of requests
/// @param evs Output: EV for each request
void heuristic_evaluate(const BoardState& board, const GameContext& ctx,
                        const AfterstateRequest* requests, int request_count,
                        double* evs);
```

**Implementation:**

```cpp
void heuristic_evaluate(const BoardState& board, const GameContext& ctx,
                        const AfterstateRequest* requests, int request_count,
                        double* evs) {
    for (int i = 0; i < request_count; ++i) {
        int col = requests[i].placement.column;
        int score = requests[i].score;
        evs[i] = static_cast<double>(score) * board.coefficients[col];
    }
}
```

This is deliberately simple. It captures the basic insight that higher scores in higher-coefficient columns are better. It doesn't account for:
- Strategic positioning (sacrificing one column to win another)
- Clean column potential
- Upper section bonus thresholds
- SS/LS mutual destruction consequences

These strategic considerations are what the NN will eventually learn.

### 4.1 Heuristic Bot: Complete Turn

The heuristic bot combines the solver with the heuristic evaluation to play a complete turn:

```cpp
/// Play one complete turn using the heuristic bot.
/// Makes hold/reroll decisions until placing a score.
///
/// @param state The game state (modified: dice rerolled, placement applied)
/// @param ctx The game context (modified: caches updated)
/// @param tables Precomputed tables
/// @param buffers Solver buffers (reused across calls)
/// @param rng Random engine for dice rerolling
void heuristic_play_turn(GameState& state, GameContext& ctx,
                          const PrecomputedTables& tables,
                          SolverBuffers& buffers, RNG& rng);
```

**Implementation:**

```
While true:
    // Step 1: Get requests
    solver_get_requests(state, ctx, tables, buffers)
    
    // Step 2: Evaluate with heuristic
    heuristic_evaluate(state.board, ctx, buffers.requests,
                       buffers.request_count, buffers.evs)
    
    // Step 3: Resolve
    result = solver_resolve(state, ctx, tables, buffers)
    
    If result.should_place:
        perform_placement(result.placement, result.score, state, ctx, rng)
        return
    Else:
        If !can_reroll(state, ctx):
            // Shouldn't happen — solver should have chosen to place
            assert(false)
        perform_hold(result.hold_mask, state, rng)
        // Loop back — new dice, resolve again
```

Note: After `perform_hold`, the dice have changed but the board hasn't. The legal placements and Golden Rule thresholds are the same. However, we must call `solver_get_requests` again because `rolls_left` has changed, which affects whether Turbo cells are available (if `rolls_left` dropped to 0).

Actually — the **requests** don't change between rerolls within the same turn (same legal placements, same Golden Rule thresholds). Only the dice change. The request list and EVs are board-dependent, not dice-dependent. The dice only matter in `solver_resolve` (which dice state maps to which V0 entry).

**Optimization:** We can call `solver_get_requests` and `heuristic_evaluate` once per turn, then call `solver_resolve` after each reroll with the same requests/EVs but updated dice. This avoids redundant work on rerolls.

But wait — `rolls_left` affects legal placements (Turbo exclusion at `rolls_left == 0`). If Turbo cells exist and `rolls_left` drops to 0 during the turn, the request list changes. However, this only matters for the "what if I stop now" path in the solver, and V0 uses the full request list (including Turbo) because stopping voluntarily means rolls remain.

Actually no — if `rolls_left == 0`, the engine excludes Turbo from legal placements, so `solver_get_requests` would generate a different (smaller) request list. To support the optimization, we'd need to handle this case specially.

**Simpler approach:** Just call `solver_get_requests` + evaluation + `solver_resolve` on each iteration. The cost is dominated by the DP (252 states × 32 masks × transitions), not by request generation. For the heuristic bot, the evaluation is trivially cheap anyway. For the NN bot (Task 08), the optimization of reusing requests across rerolls will matter more and we can revisit then.

---

## 5. Exploration (Temperature-Based Sampling)

### 5.1 Placement Temperature

When exploration is enabled, instead of returning the argmax placement from V0, the solver samples proportionally using softmax with temperature.

**Critical: Logit transformation.** The V(s) values are win probabilities in [0,1]. Applying softmax directly to values in this compressed range produces near-uniform sampling even at T=1.0 (e.g., V=0.95 vs V=0.05 would be chosen 73% vs 27% — far too much exploration). To restore meaningful dynamic range, values must be converted to logits first: `logit = ln(V / (1-V))`. This maps [0,1] to (-∞,+∞), making the temperature parameter behave as expected.

```cpp
/// Apply softmax temperature to select an action probabilistically.
/// Converts win probabilities to logits before applying softmax.
/// When temperature → 0, this becomes argmax (greedy).
/// When temperature → ∞, this becomes uniform random.
///
/// @param values Array of win probabilities in [0, 1]
/// @param count Number of values
/// @param temperature Softmax temperature (> 0)
/// @param rng Random engine
/// @return Index of the selected value
int softmax_sample(const double* values, int count, double temperature, RNG& rng);
```

**Implementation:**

```cpp
int softmax_sample(const double* values, int count, double temperature, RNG& rng) {
    if (count == 1) return 0;

    // Convert win probabilities to logits for meaningful temperature scaling.
    // Clamp to avoid log(0) or log(inf).
    constexpr double kClampMin = 1e-6;
    constexpr double kClampMax = 1.0 - 1e-6;
    
    double logits[count];  // VLA or use a fixed buffer from SolverBuffers
    double max_logit = -1e18;
    for (int i = 0; i < count; ++i) {
        double v = std::clamp(values[i], kClampMin, kClampMax);
        logits[i] = std::log(v / (1.0 - v));
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    // Softmax with temperature on logits (shift by max for numerical stability)
    double weights[count];
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        weights[i] = std::exp((logits[i] - max_logit) / temperature);
        sum += weights[i];
    }
    
    // Sample proportionally
    double r = rng.uniform_double() * sum;  // uniform in [0, sum)
    double cumulative = 0.0;
    for (int i = 0; i < count - 1; ++i) {
        cumulative += weights[i];
        if (r < cumulative) return i;
    }
    return count - 1;
}
```

**Why logits matter — concrete example:**
- V(best) = 0.95, V(worst) = 0.05, T = 1.0
- Without logits: exp(0.95) / exp(0.05) ≈ 2.46 → worst move chosen 29% of the time
- With logits: logit(0.95) = 2.94, logit(0.05) = -2.94, exp(2.94) / exp(-2.94) ≈ 357 → worst move chosen 0.3% of the time
- T = 1.0 now provides sensible light exploration; T = 2.0+ provides heavier exploration

Note: Need to add `uniform_double()` returning [0, 1) to the RNG class from Task 03.

### 5.2 Hold Mask Temperature

Similarly, at each reroll layer, instead of argmax over hold masks, sample using temperature:

```
// In the reroll layer, instead of:
If ev > best_ev: best_ev = ev; best_mask = mask;

// Collect all mask EVs, then sample:
mask_evs[32] = { ev for each mask }
selected_mask = softmax_sample(mask_evs, 32, hold_temperature, rng)
```

### 5.3 Temperature Configuration

The solver accepts temperature parameters:

```cpp
struct SolverConfig {
    double placement_temperature;   // 0.0 = greedy, > 0 = exploration
    double hold_temperature;        // 0.0 = greedy, > 0 = exploration
    bool exploration_enabled;       // Master switch
};
```

When `exploration_enabled == false` or temperature == 0.0, the solver uses pure argmax (no sampling, no softmax computation). This is used during evaluation games.

---

## 6. Playing a Complete Game

With the heuristic bot and game flow from Task 03, we can now play complete games:

```cpp
/// Play a complete game using the heuristic bot for both players.
///
/// @param rng Random engine
/// @param tables Precomputed tables
/// @return Game result: 1.0 if player 0 wins, 0.0 if player 1 wins, 0.5 if draw
double play_heuristic_game(RNG& rng, const PrecomputedTables& tables);
```

**Implementation:**

```
GameState state
GameContext ctx
SolverBuffers buffers
init_game(state, ctx, rng)

While !is_terminal(state.board):
    heuristic_play_turn(state, ctx, tables, buffers, rng)

return get_game_result(state, ctx)
```

This is the first end-to-end playable bot. It should be able to play thousands of games per second since there's no NN inference involved.

---

## 7. File Organization

```
src/solver/
├── precomputed_tables.h       # (from Task 04)
├── precomputed_tables.cc      # (from Task 04)
├── solver.h                   # SolverBuffers, SolverResult, AfterstateRequest,
│                              # solver_get_requests, solver_resolve, SolverConfig,
│                              # softmax_sample
├── solver.cc                  # Solver implementation
└── CMakeLists.txt

src/heuristic/
├── heuristic_bot.h            # heuristic_evaluate, heuristic_play_turn,
│                              # play_heuristic_game
├── heuristic_bot.cc           # Heuristic bot implementation
└── CMakeLists.txt
```

```cmake
# src/solver/CMakeLists.txt
add_library(solver STATIC
    precomputed_tables.cc
    solver.cc
)
target_include_directories(solver PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(solver PUBLIC engine)

# src/heuristic/CMakeLists.txt
add_library(heuristic STATIC
    heuristic_bot.cc
)
target_include_directories(heuristic PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(heuristic PUBLIC engine solver)
```

---

## 8. Unit Tests

### 8.1 Solver Request Tests (`tests/solver/solver_request_test.cc`)

**Basic request generation:**
- Fresh board, first turn. Verify requests include all legal cells with their possible scores.
- Verify scratch (score = 0) is included for every legal cell.
- Verify request count is within expected bounds (> 0, < kMaxAfterstateRequests).

**Golden Rule filtering:**
- Set up a board where golden_max[col][row] = 12 for row 3s. Verify that scores {3, 6, 9} are NOT in the request list for that cell (below threshold), but {12, 15} ARE.

**SS/LS constraint filtering:**
- LS already scratched in a column → verify SS requests for that column only contain scratch (score = 0).
- SS filled with 25 in a column → verify LS requests only contain scores > 25.
- SS already scratched → verify LS requests only contain scratch.

**Turbo exclusion:**
- Set `rolls_left = 0`. Verify no Turbo column cells appear in requests.
- Set `rolls_left = 1`. Verify Turbo column cells DO appear.

### 8.2 Solver Resolve Tests (`tests/solver/solver_resolve_test.cc`)

**Trivial case — rolls_left = 0:**
- Set up requests with known EVs. Verify the solver returns the placement with the highest EV for the current dice.

**Hold vs stop decision:**
- Set up a scenario where current dice produce a mediocre score (low EV for all placements), but holding certain dice gives high expected improvement. Verify solver chooses to hold.
- Set up a scenario where current dice already produce an excellent score. Verify solver chooses to stop and place.

**Layer consistency:**
- V0[state] should always be <= V1[state] <= V2[state] (more rolls = more opportunity = higher or equal EV).

**Greedy vs exploration:**
- With temperature = 0 (greedy), verify solver always returns the argmax action.
- With high temperature, verify that non-optimal actions are sometimes selected (run multiple times with different RNG seeds, check that not all results are identical).

### 8.3 Heuristic Bot Tests (`tests/heuristic/heuristic_bot_test.cc`)

**Heuristic evaluation:**
- Score 30 in column with coefficient 18 → EV = 540.
- Score 0 (scratch) → EV = 0.
- Verify EVs are computed correctly for a batch of requests.

**Complete game:**
- Play 100 heuristic-vs-heuristic games with different seeds. Verify:
  - All games complete without assertion failures.
  - Results are a mix of wins for player 0 and player 1 (heuristic is symmetric, so approximately 50/50 with variance from dice/coefficients/starting player).
  - No game exceeds the maximum number of turns.

**Heuristic beats random:**
- Play 100 games: heuristic bot vs random bot (random legal placements, random holds). Verify heuristic wins the vast majority (>90%).

### 8.4 Softmax Tests (`tests/solver/softmax_test.cc`)

**Determinism:**
- Same values, same temperature, same RNG seed → same result.

**Logit transformation:**
- Values {0.95, 0.05} with T=1.0: should pick the higher value >99% of the time (logit dynamic range makes this decisive).
- Values {0.6, 0.5, 0.4} with T=1.0: should pick all three with some frequency, but strongly favor 0.6.
- Verify logit transformation: V=0.5 → logit=0, V=0.95 → logit≈2.94, V=0.05 → logit≈-2.94.

**Temperature extremes:**
- Temperature very close to 0 (e.g., 0.001): should almost always pick the highest value. Run 1000 trials, verify >99% pick argmax.
- Very high temperature (e.g., 100.0): should be roughly uniform. Run 10000 trials, verify each option is picked roughly equally (chi-squared test or frequency check).

**Boundary clamping:**
- Values at extremes {1.0, 0.0}: should not produce NaN or infinity (clamped to [1e-6, 1-1e-6]).
- Values all identical {0.5, 0.5, 0.5}: should produce uniform sampling regardless of temperature.

**Numerical stability:**
- Values very close together {0.501, 0.500, 0.499}: should not produce degenerate weights.

---

## 9. Benchmarks

Add to `tests/benchmarks/solver_bench.cc`:

- **BM_SolverGetRequests:** Benchmark request generation from various board states (early game, mid game, late game).
- **BM_SolverResolve:** Benchmark the full DP resolution (V0 + V1 + V2) with pre-populated EVs.
- **BM_HeuristicEvaluate:** Benchmark heuristic evaluation of a batch of requests.
- **BM_HeuristicPlayTurn:** Benchmark a complete heuristic turn (requests + evaluate + resolve + potential rerolls).
- **BM_HeuristicFullGame:** Benchmark a complete heuristic-vs-heuristic game. This establishes the maximum possible game throughput without NN inference.
- **BM_SoftmaxSample:** Benchmark softmax sampling with various array sizes and temperatures.

---

## 10. Definition of Done

This task is complete when:

1. `solver_get_requests` correctly enumerates all valid (cell, score) afterstates, respecting Golden Rule, SS/LS constraints, and Turbo exclusion.
2. `solver_resolve` implements the full expectimax DP (V0, V1, V2 layers) with stop-vs-reroll decisions at each layer.
3. The heuristic evaluation function computes score × coefficient correctly.
4. `heuristic_play_turn` plays a complete turn with correct hold/reroll/place decisions.
5. `play_heuristic_game` runs complete games end-to-end.
6. Softmax temperature-based exploration works correctly at both placement and hold mask levels.
7. The heuristic bot beats a random bot decisively (>90% win rate).
8. All unit tests pass.
9. Benchmarks establish baseline throughput for heuristic games (expected: thousands of games/sec).
10. No heap allocation in any per-call solver function (all buffers pre-allocated).
11. The solver and heuristic libraries compile with no libtorch dependency.
