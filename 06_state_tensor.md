# Task 06: State Tensor Design

## Overview

Design and implement the feature-rich, normalized state observation tensor that the neural network evaluates. This tensor is the bridge between the game engine and the NN — it encodes everything the network needs to predict win probability from a board position.

The tensor is generated from a `BoardState` (no dice, no rolls — pure board position after a placement). It always represents the perspective of the player who just placed (whose opponent moves next).

## Prerequisites

- Task 02 completed (engine data structures, scoring, GameContext)
- Task 04 completed (precomputed tables, transitions)

---

## 1. Probability Precomputation

### 1.1 Single-Turn Achievement Probability

For each row type and each minimum score threshold, precompute the probability of achieving a score >= threshold when dedicating a full turn (3 rolls with optimal holding toward that goal).

This is a mini-solver DP per (row, threshold) pair:

**Layer 0:** For each of the 252 dice states, check if the dice produce a score >= threshold for this row. Result is binary: 1.0 (achieves) or 0.0 (doesn't).

**Layer 1:** For each of the 252 dice states, find the best hold mask that maximizes probability of achieving the target. For each mask, compute the expected value over transitions using the binary V0. Take the max across all masks.

**Layer 2:** Same structure, using V1. The result at the "hold nothing" configuration (reroll all 5 dice) gives the probability from a fresh roll.

**Storage:**

```cpp
struct ProbabilityTables {
    // Probability of achieving score >= threshold in row with 3 rolls (normal)
    // prob_3rolls[row][threshold]
    // Row: 0-12, Threshold: 0-100
    double prob_3rolls[13][101];

    // Same but with 2 rolls (Turbo column)
    double prob_2rolls[13][101];
};
```

### 1.2 Multi-Turn Compounding

At tensor generation time, compute the probability of achieving a score in T attempts:

```
P(success in T turns) = 1 - (1 - P_single)^T
```

Where `P_single` comes from the precomputed table (using `prob_3rolls` for normal columns, `prob_2rolls` for Turbo), and `T = total_turns_remaining` for the player (`78 - placements_made_by_player`).

### 1.3 Initialization

Add `ProbabilityTables` to the startup precomputation in `PrecomputedTables` (from Task 04):

```cpp
// Add to PrecomputedTables:
ProbabilityTables prob_tables;
```

Initialize after the transition tables are built (the mini-solver DP depends on them).

**Performance:** 13 rows × ~100 thresholds = ~1300 mini-solver runs. Each run is 252 states × 32 masks × small transition count. Expected: well under 1 second total.

---

## 2. Tensor Feature Design

### 2.1 Perspective Convention

The tensor is always constructed with:
- **"Me"** = the player who just placed (player index 0 in the tensor)
- **"Opponent"** = the player who moves next (player index 1 in the tensor)

The caller is responsible for passing the correct player perspective. The tensor generation function takes a `player` parameter indicating whose perspective to use.

### 2.2 Feature Layout

All features are `float` values. The tensor is a 1D array of floats. Features are grouped logically but stored contiguously.

---

### Group A: Per-Player, Per-Column Cell Values

For each player (me first, opponent second), for each column (0–5), for each row (0–12):

**Upper cells (rows 0–5):** 3 features per cell

| Feature | Value if empty | Value if filled | Notes |
|---------|---------------|----------------|-------|
| is_filled | 0.0 | 1.0 | Binary indicator |
| score / cell_max | 0.0 | score / max_for_this_row | Per-cell potential (e.g., /5 for 1s, /30 for 6s) |
| score / 30 | 0.0 | score / 30 | Normalized to upper section scale |

**Lower cells (rows 6–12):** 3 features per cell

| Feature | Value if empty | Value if filled | Notes |
|---------|---------------|----------------|-------|
| is_filled | 0.0 | 1.0 | Binary indicator |
| score / cell_max | 0.0 | score / max_for_this_row | Per-cell potential (e.g., /50 for FH, /100 for Y) |
| score / 100 | 0.0 | score / 100 | Normalized to lower section scale |

**Max score per row (for normalization denominators):**

| Row | Max Score |
|-----|-----------|
| 1s | 5 |
| 2s | 10 |
| 3s | 15 |
| 4s | 20 |
| 5s | 25 |
| 6s | 30 |
| SS | 29 |
| LS | 30 |
| FH | 50 |
| K | 54 |
| STR | 50 |
| U8 | 75 |
| Y | 100 |

Count: 2 players × 6 columns × 13 rows × 3 features = **468 features**

---

### Group B: Per-Player, Per-Column Derived Features

For each player (me first, opponent second), for each column (0–5):

| # | Feature | Normalization | Notes |
|---|---------|--------------|-------|
| 1 | Upper section sum | / 100, capped at 1.0 | Current sum of rows 0–5 |
| 2 | Upper potential max sum | / 100, capped at 1.0 | Sum if all unfilled upper cells got max score |
| 3 | Clean column eligibility | 0.0 or 1.0 | upper_sum >= 60 AND no lower scratches |
| 4 | Column duel advantage (positive) | min(1.0, max(0, diff) / 50000) | (my_raw - opp_raw) if positive, else 0 |
| 5 | Column duel disadvantage (positive) | min(1.0, max(0, -diff) / 50000) | (opp_raw - my_raw) if positive, else 0 |
| 6 | Potential column duel advantage | min(1.0, max(0, pot_diff) / 50000) | Same but unfilled cells counted as max |
| 7 | Potential column duel disadvantage | min(1.0, max(0, -pot_diff) / 50000) | Same |
| 8 | Cells remaining | / 13 | Fraction of column unfilled |

Count: 2 players × 6 columns × 8 features = **96 features**

**Notes on duel score computation (features 4–7):**
- "Raw score" = sum of all filled cells in the column + upper section bonus (do not include clean bonus — that depends on the duel context)
- For potential scores, count unfilled cells as their max possible score
- These are computed per player per column, not as a difference between players — each player gets their own advantage/disadvantage values from their own perspective
- Split into positive/negative to work well with ReLU activations

---

### Group C: Per-Player, Per-Column, Per-Cell Probability Features

For each player (me first, opponent second), for each column (0–5), for each row (0–12):

| Feature | Value if filled | Value if empty | Notes |
|---------|----------------|---------------|-------|
| P(non-scratch) | 1.0 | Compounded probability | See computation below |

**Computation for empty cells:**

```
golden_min = golden_max[column][row]  // minimum score needed (Golden Rule)
If golden_min == 0: golden_min = 1    // if no threshold, need at least min non-zero score

// Determine single-turn probability
If column == Turbo:
    p_single = prob_tables.prob_2rolls[row][golden_min]
Else:
    p_single = prob_tables.prob_3rolls[row][golden_min]

// Compound over remaining turns
turns_remaining = 78 - placements_made_by_player
p_compound = 1.0 - pow(1.0 - p_single, turns_remaining)
```

**SS/LS special handling:**
- If the paired cell (SS↔LS) is scratched, P(non-scratch) = 0.0 (forced scratch)
- If LS is already filled in the column, SS's golden_min must also satisfy SS < LS
- If SS is already filled, LS's golden_min must also satisfy LS > SS

Count: 2 players × 6 columns × 13 rows = **156 features**

---

### Group D: Global Features

| # | Feature | Normalization | Count | Notes |
|---|---------|--------------|-------|-------|
| 1 | Column coefficients | / 18 | 6 | Range [0.44, 1.0] |
| 2 | Game progress | cells_filled / 156 | 1 | [0, 1] |
| 3 | Golden Rule max per (col, row) | / max_per_row | 78 | Current threshold for each cell |
| 4 | My total score | min(1.0, score / 50000) | 1 | Sum of all my filled cells |
| 5 | Opponent total score | min(1.0, score / 50000) | 1 | Sum of all opponent's filled cells |
| 6 | My potential max total | min(1.0, score / 50000) | 1 | My score if remaining cells maxed |
| 7 | Opponent potential max total | min(1.0, score / 50000) | 1 | Opponent's score if remaining cells maxed |

Count: 6 + 1 + 78 + 1 + 1 + 1 + 1 = **89 features**

---

### Total Feature Count

| Group | Features |
|-------|----------|
| A: Cell values | 468 |
| B: Column derived | 96 |
| C: Probability | 156 |
| D: Global | 89 |
| **Total** | **809** |

---

## 3. Tensor Generation Function

```cpp
// src/engine/tensor.h

/// Total number of features in the state tensor.
constexpr int kTensorSize = 809;

/// Generate the state observation tensor for a board position.
///
/// @param board The board state to encode
/// @param ctx The game context (for cached derived data)
/// @param player The player whose perspective to use (0 or 1).
///               This player's data goes first in the tensor.
/// @param tables Precomputed tables (for probability lookups and score tables)
/// @param out Output float array of size kTensorSize
void generate_tensor(const BoardState& board, const GameContext& ctx,
                     int player, const PrecomputedTables& tables,
                     float* out);
```

**Implementation outline:**

```
int idx = 0;

// === Group A: Cell values ===
for p in [player, 1-player]:          // "me" first, "opponent" second
    for col in 0..5:
        for row in 0..12:
            cell = board.cells[p][col][row]
            
            if cell == kCellEmpty:
                out[idx++] = 0.0   // is_filled
                out[idx++] = 0.0   // score / cell_max
                out[idx++] = 0.0   // score / section_max
            else:
                out[idx++] = 1.0   // is_filled
                out[idx++] = float(cell) / max_score_per_row[row]
                if row <= 5:
                    out[idx++] = float(cell) / 30.0   // upper section scale
                else:
                    out[idx++] = float(cell) / 100.0  // lower section scale

// === Group B: Column derived ===
for p in [player, 1-player]:
    for col in 0..5:
        // Upper sum and potential
        upper_sum = ctx.upper_sum[p][col]
        upper_potential = upper_sum
        for row in 0..5:
            if board.cells[p][col][row] == kCellEmpty:
                upper_potential += max_score_per_row[row]
        out[idx++] = min(1.0, upper_sum / 100.0)
        out[idx++] = min(1.0, upper_potential / 100.0)
        
        // Clean column eligibility
        clean = (upper_sum >= 60) && !ctx.lower_has_scratch[p][col]
        out[idx++] = clean ? 1.0 : 0.0
        
        // Column duel scores
        my_raw = compute_column_raw_score(board, ctx, p, col)
        opp_raw = compute_column_raw_score(board, ctx, 1-p, col)
        diff = my_raw - opp_raw
        out[idx++] = min(1.0, max(0.0, diff) / 50000.0)    // advantage
        out[idx++] = min(1.0, max(0.0, -diff) / 50000.0)   // disadvantage
        
        // Potential duel scores (empty cells as max)
        my_pot = compute_column_potential_score(board, ctx, p, col)
        opp_pot = compute_column_potential_score(board, ctx, 1-p, col)
        pot_diff = my_pot - opp_pot
        out[idx++] = min(1.0, max(0.0, pot_diff) / 50000.0)
        out[idx++] = min(1.0, max(0.0, -pot_diff) / 50000.0)
        
        // Cells remaining
        remaining = count_empty_cells(board, p, col)
        out[idx++] = float(remaining) / 13.0

// === Group C: Probability features ===
for p in [player, 1-player]:
    placements_made = count_filled_cells(board, p)
    turns_remaining = 78 - placements_made
    for col in 0..5:
        for row in 0..12:
            if board.cells[p][col][row] != kCellEmpty:
                out[idx++] = 1.0   // Already filled
            else:
                // Determine golden rule minimum
                golden_min = ctx.golden_max[col][row]
                if golden_min == 0: golden_min = 1
                
                // SS/LS special cases
                if row == SS and ctx.ls_scratched[p][col]:
                    out[idx++] = 0.0; continue  // Forced scratch
                if row == LS and ctx.ss_scratched[p][col]:
                    out[idx++] = 0.0; continue  // Forced scratch
                if row == SS:
                    // Must also be < own LS if LS is filled
                    ls_val = board.cells[p][col][LS]
                    if ls_val != kCellEmpty:
                        // Additional constraint: SS < ls_val
                        // Adjust golden_min if needed (must score >= golden_min AND < ls_val)
                        if golden_min >= ls_val:
                            out[idx++] = 0.0; continue  // Impossible
                if row == LS:
                    ss_val = board.cells[p][col][SS]
                    if ss_val != kCellEmpty:
                        golden_min = max(golden_min, ss_val + 1)
                
                // Cap threshold at max score for this row
                max_for_row = max_score_per_row[row]
                if golden_min > max_for_row:
                    out[idx++] = 0.0; continue  // Impossible
                
                // Lookup single-turn probability
                if col == kColTurbo:
                    p_single = tables.prob_tables.prob_2rolls[row][golden_min]
                else:
                    p_single = tables.prob_tables.prob_3rolls[row][golden_min]
                
                // Compound over remaining turns
                p_compound = 1.0 - pow(1.0 - p_single, turns_remaining)
                out[idx++] = float(p_compound)

// === Group D: Global features ===
for col in 0..5:
    out[idx++] = float(board.coefficients[col]) / 18.0

out[idx++] = float(board.cells_filled) / 156.0

for col in 0..5:
    for row in 0..12:
        out[idx++] = float(ctx.golden_max[col][row]) / max_score_per_row[row]

// Total scores
my_total = sum_all_filled(board, player)
opp_total = sum_all_filled(board, 1-player)
out[idx++] = min(1.0, float(my_total) / 50000.0)
out[idx++] = min(1.0, float(opp_total) / 50000.0)

// Potential max totals
my_potential = compute_total_potential(board, player)
opp_potential = compute_total_potential(board, 1-player)
out[idx++] = min(1.0, float(my_potential) / 50000.0)
out[idx++] = min(1.0, float(opp_potential) / 50000.0)

assert(idx == kTensorSize);
```

### 3.1 Helper Functions

```cpp
/// Compute raw column score for a player: sum of all filled cells + upper bonus.
/// Does NOT include clean column bonus (that depends on duel context).
int compute_column_raw_score(const BoardState& board, const GameContext& ctx,
                              int player, int column);

/// Compute potential column score: raw score + max possible for unfilled cells + potential upper bonus.
int compute_column_potential_score(const BoardState& board, const GameContext& ctx,
                                    int player, int column);

/// Compute total potential score across all columns for a player.
int compute_total_potential(const BoardState& board, int player);

/// Count empty cells for a player in a column.
int count_empty_cells(const BoardState& board, int player, int column);

/// Count total filled cells for a player (across all columns).
int count_filled_cells(const BoardState& board, int player);

/// Sum all filled cell values for a player across all columns.
int sum_all_filled(const BoardState& board, int player);
```

---

## 4. Performance Considerations

### 4.1 Tensor Generation Cost

The tensor generation is called for every afterstate the solver evaluates. With potentially hundreds of afterstates per turn across hundreds of concurrent games, this is a high-frequency operation.

**Optimization strategies:**
- `max_score_per_row` is a compile-time constant array — no lookup overhead.
- The `pow()` call in probability compounding is expensive. Precompute `pow(1 - p_single, T)` for common values of T, or use a fast approximation. Since T changes only once per placement (not per afterstate), we could precompute `(1 - p_single)^T` for the current T at the start of each solver call and cache it.
- Group A (cell values) is mostly simple divides with compile-time constant denominators — the compiler will convert these to multiplies.
- Group B and D features that depend on column-level aggregates (sums, potentials) can be partially precomputed from GameContext caches.
- Avoid branching in the inner loops where possible — use branchless min/max.

### 4.2 Afterstate Tensor Optimization

When the solver evaluates afterstates, each afterstate differs from the current board by exactly one cell change. Rather than regenerating the entire 809-float tensor from scratch for each afterstate, we could potentially compute a "delta" — but this adds complexity and the full generation is likely fast enough. Profile first, optimize later.

### 4.3 Batch Generation

When generating tensors for multiple afterstates (for batched NN inference), write them contiguously into a pre-allocated float buffer:

```cpp
/// Generate tensors for a batch of afterstates.
/// Each afterstate is defined by a placement applied to the base board.
///
/// @param board Base board state (before any placement)
/// @param ctx Game context for the base state
/// @param player Perspective player
/// @param requests Array of (placement, score) pairs
/// @param request_count Number of afterstates
/// @param tables Precomputed tables
/// @param out Output buffer: request_count × kTensorSize floats, contiguous
void generate_tensor_batch(const BoardState& board, const GameContext& ctx,
                            int player,
                            const AfterstateRequest* requests, int request_count,
                            const PrecomputedTables& tables,
                            float* out);
```

Implementation: for each request, clone BoardState, apply the placement (simple cell write + cells_filled increment), generate tensor into `out + i * kTensorSize`. The GameContext is NOT fully updated for the clone (no cache maintenance) — the tensor generation reads directly from the modified cells where needed and uses the base GameContext for unchanged fields.

**Important:** Since we don't maintain a full GameContext for cloned afterstates, some derived features need to be computed on the fly from the raw cells:
- `golden_max`: use the base ctx values, but check if the new placement raised the max for its (col, row)
- `upper_sum`: recompute from cells for the affected column
- `ss_scratched`/`ls_scratched`: check if the placement triggers mutual destruction
- `lower_has_scratch`: check if the placement is a scratch in the lower section
- `clean column eligibility`: recompute from the above

This is more work per tensor but avoids the cost of full GameContext maintenance for throwaway clones.

---

## 5. File Organization

```
src/engine/
├── ... (existing files)
├── tensor.h              # kTensorSize, generate_tensor, generate_tensor_batch,
│                         # helper function declarations
├── tensor.cc             # Tensor generation implementation
├── probability_tables.h  # ProbabilityTables struct (added to PrecomputedTables)
└── probability_tables.cc # Probability precomputation
```

Update `src/engine/CMakeLists.txt` to include new source files.

Note: Tensor generation lives in the engine library (not in the model library) because it has no libtorch dependency — it produces raw float arrays. The model library (Task 07) wraps these into libtorch tensors.

---

## 6. Unit Tests

### 6.1 Probability Table Tests (`tests/engine/probability_table_test.cc`)

**Known probabilities:**
- P(Yams with any score, 3 rolls) should be small (~4.6% for any specific face).
- P(straight 1-2-3-4-5, 3 rolls) should be moderate.
- P(any number row with score >= 1, 3 rolls) should be very high (>99% for most).
- P(under 8 with score >= 60, 3 rolls) — verify against known/computed value.

**Boundary conditions:**
- Threshold = 0 → probability should be 1.0 for all rows (any score >= 0, including the minimum non-zero).

  Wait — threshold 0 means "need score >= 0," but score 0 means scratch. We want P(non-scratch). So threshold should be 1 (the minimum non-zero score achievable). Verify: `prob_3rolls[row][1]` gives the probability of any non-zero score.

- Threshold > max_for_row → probability should be 0.0.
- Threshold = max_for_row → probability should be > 0 (it's achievable but rare).

**Turbo vs Normal:**
- For all rows and thresholds: `prob_2rolls[row][t] <= prob_3rolls[row][t]` (fewer rolls = lower probability).

**Compounding:**
- P(success in 1 turn) should equal the single-turn probability.
- P(success in many turns) should approach 1.0 for achievable targets.
- P(success in 0 turns) should be 0.0.

### 6.2 Tensor Generation Tests (`tests/engine/tensor_test.cc`)

**Empty board:**
- All `is_filled` features should be 0.0.
- All score features should be 0.0.
- Game progress should be 0.0.
- All golden_max features should be 0.0.
- All probability features should be > 0.0 (every cell is achievable from a fresh game).

**Single placement:**
- Place score 18 in player 0's Free column, 6s row. Verify:
  - `is_filled` = 1.0 for that cell
  - `score / cell_max` = 18/30 = 0.6
  - `score / 30` = 0.6 (upper section)
  - Golden rule max for (Free, 6s) = 18/30 = 0.6
  - Upper sum for player 0, Free column = 18/100 = 0.18
  - Probability feature for that cell = 1.0 (filled)

**Perspective flipping:**
- Generate tensor for player 0, then for player 1 on the same board.
- Verify player 0's cell values in tensor_0 appear as opponent values in tensor_1 and vice versa.

**SS/LS probability edge cases:**
- LS scratched in a column → SS probability feature = 0.0 for that cell.
- SS filled with 25, LS unfilled → LS probability uses threshold max(golden_min, 26).

**Normalization ranges:**
- All features should be in [0.0, 1.0] (no negative values, no values > 1.0).
- Run on various mid-game board states and verify range compliance.

**Tensor size:**
- Verify `idx == kTensorSize` assertion holds for various board states.

### 6.3 Batch Generation Tests (`tests/engine/tensor_batch_test.cc`)

**Consistency:**
- Generate a tensor via `generate_tensor` on a manually cloned+modified board.
- Generate the same tensor via `generate_tensor_batch` with the corresponding request.
- Verify they produce identical output (within floating-point tolerance).

**Multiple afterstates:**
- Generate a batch of 10 afterstates. Verify each tensor in the batch corresponds to the correct placement applied to the base board.

---

## 7. Benchmarks

Add to `tests/benchmarks/tensor_bench.cc`:

- **BM_GenerateTensor:** Benchmark single tensor generation from various board states.
- **BM_GenerateTensorBatch:** Benchmark batch generation for 100, 500, 1000 afterstates.
- **BM_ProbabilityLookup:** Benchmark the probability table lookup + compounding.
- **BM_ProbabilityTableInit:** Benchmark the one-time probability table precomputation.

These benchmarks are critical — tensor generation is on the hot path and its cost directly impacts training throughput.

---

## 8. Definition of Done

This task is complete when:

1. `ProbabilityTables` is precomputed correctly for all rows, thresholds, and both 2-roll/3-roll variants.
2. `generate_tensor` produces a correctly normalized 809-float tensor from any valid board state.
3. All features are in the [0.0, 1.0] range.
4. Perspective flipping works correctly (swapping player produces the expected tensor transformation).
5. `generate_tensor_batch` produces results identical to individual `generate_tensor` calls on cloned boards.
6. SS/LS probability edge cases are handled correctly.
7. All unit tests pass.
8. Benchmarks establish baseline tensor generation throughput.
9. The tensor generation code lives in the engine library with no libtorch dependency.
10. `kTensorSize` is correctly defined and matches the actual number of features generated.
