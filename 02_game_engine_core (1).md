# Task 02: Game Engine Core

## Overview

Implement the core game engine: data structures for board state, scoring calculations, legal move generation, Golden Rule enforcement, SS/LS interlock, column constraints, duel computation, and all associated unit tests. This is the foundation that every other component depends on.

This task produces no ML code, no solver logic, and no game loop orchestration (that's Task 03). It provides the data structures and pure functions that represent and manipulate Pro Yams board state.

## Prerequisites

- Task 01 completed (project scaffolding, CMake, gtest/gbench working)

---

## 1. Data Structures

### 1.1 BoardState (The Cloneable Core)

This is the minimal board representation that the solver clones when evaluating afterstates and that the NN evaluates. It must be trivially copyable, small, and cache-friendly.

```cpp
// src/engine/board_state.h

struct BoardState {
    // Cell values: -1 = empty, 0-100 = filled (0 means scratched)
    // Layout: [player][column][row]
    // Player 0 = current evaluating player, Player 1 = opponent
    // Columns: 0=Down, 1=Free, 2=Up, 3=Mid, 4=Turbo, 5=UpDown
    // Rows: 0=1s, 1=2s, 2=3s, 3=4s, 4=5s, 5=6s,
    //        6=SS, 7=LS, 8=FH, 9=K, 10=STR, 11=U8, 12=Y
    int8_t cells[2][6][13];

    // Column coefficients (shuffled {8,10,12,14,16,18})
    // Same for both players
    int8_t coefficients[6];

    // Whose turn is NEXT (0 or 1)
    int8_t current_player;

    // Total cells filled across both players (0-156)
    // Used for game progress tracking
    uint8_t cells_filled;

    // ~164 bytes total, fits in 3 cache lines
};
static_assert(std::is_trivially_copyable_v<BoardState>);
```

Add a compile-time assertion that `BoardState` is trivially copyable and check its size doesn't exceed 168 bytes (3 × 56-byte cache lines).

### 1.2 GameState (Active Play)

Extends `BoardState` with dice and roll information for active gameplay.

```cpp
// src/engine/game_state.h

struct GameState {
    BoardState board;

    // Current dice values (1-6), only valid during a turn
    int8_t dice[5];

    // Rolls remaining in current turn (0, 1, or 2)
    int8_t rolls_left;

    // ~170 bytes total
};
```

### 1.3 GameContext (Cached Derived Data)

Maintained during actual gameplay, updated incrementally on each placement. Not cloned by the solver.

```cpp
// src/engine/game_context.h

struct Placement {
    int8_t column;
    int8_t row;
};

/// Cache for legal placements. Uses a flat array for fast iteration
/// (the solver's hot path) and a bitset for O(1) membership queries.
/// Zero heap allocation — fully pre-allocated.
struct LegalPlacementCache {
    Placement placements[78];    // Dense array for iteration — max 78 legal cells
    int8_t count;                // Number of valid entries in placements[]
    bool is_legal[6][13];        // O(1) lookup: is (column, row) currently legal?
};

struct GameContext {
    // === Golden Rule cache ===
    // Max score in each (column, row) across both players
    // Updated when a cell is filled
    int8_t golden_max[6][13];           // 78 bytes

    // === Upper section sums ===
    // Running sum of rows 0-5 per player per column
    int16_t upper_sum[2][6];            // 24 bytes

    // === SS/LS scratch status ===
    // Whether SS (row 6) or LS (row 7) is scratched per player per column
    // Used to determine if the paired cell is force-scratched
    bool ss_scratched[2][6];            // 12 bytes
    bool ls_scratched[2][6];            // 12 bytes

    // === Clean column tracking ===
    // Whether any lower section cell (rows 6-12) has been scratched
    // per player per column
    bool lower_has_scratch[2][6];       // 12 bytes

    // === Cached legal placements per player ===
    // Two caches: all placements (including Turbo) and non-Turbo only.
    // Both updated incrementally on each placement.
    // See Task 03 for LegalPlacementCache struct definition and
    // roll-aware get_legal_placements interface.
    LegalPlacementCache legal_all[2];
    LegalPlacementCache legal_no_turbo[2];

    // === Non-turbo cells remaining per player ===
    // Decremented on each non-Turbo placement. When 0, only Turbo cells remain.
    // Used for can_reroll() optimization. See Task 03.
    int8_t non_turbo_cells_remaining[2];
};
```

### 1.4 Constants and Enums

```cpp
// src/engine/constants.h

constexpr int kNumPlayers = 2;
constexpr int kNumColumns = 6;
constexpr int kNumRows = 13;
constexpr int kNumDice = 5;
constexpr int kNumDieSides = 6;
constexpr int kTotalCells = kNumPlayers * kNumColumns * kNumRows;  // 156

// Column indices (fixed assignment)
enum Column : int8_t {
    kColDown = 0,
    kColFree = 1,
    kColUp = 2,
    kColMid = 3,
    kColTurbo = 4,
    kColUpDown = 5
};

// Row indices
enum Row : int8_t {
    kRow1s = 0, kRow2s = 1, kRow3s = 2,
    kRow4s = 3, kRow5s = 4, kRow6s = 5,
    kRowSS = 6, kRowLS = 7, kRowFH = 8,
    kRowK = 9, kRowSTR = 10, kRowU8 = 11, kRowY = 12
};

constexpr int8_t kCellEmpty = -1;

// All possible coefficients
constexpr int8_t kCoefficients[] = {8, 10, 12, 14, 16, 18};
```

---

## 2. SolverTables (Startup Precomputation)

Built once at program startup, shared globally (read-only after initialization). These tables eliminate redundant computation during solver operation.

```cpp
// src/engine/solver_tables.h

struct SolverTables {
    // === Possible scores per row ===
    // All distinct non-zero scores achievable for each row type
    // possible_scores[row] → fixed array of scores
    // possible_count[row] → number of valid scores for this row
    int8_t possible_scores[13][32];     // 32 is generous upper bound
    int8_t possible_count[13];

    // === Scores filtered by Golden Rule minimum ===
    // For each row and each possible minimum threshold,
    // the subset of possible scores >= that threshold.
    // filtered_scores[row][threshold] → array of valid scores
    // filtered_count[row][threshold] → count
    // Max threshold per row varies (max 100 for Yams),
    // so we allocate for the worst case.
    int8_t filtered_scores[13][101][32];
    int8_t filtered_count[13][101];

    // === Dice state → score per row ===
    // For each of the 252 sorted dice states and each row,
    // what raw score does it produce? (0 if doesn't qualify)
    // This does NOT include Golden Rule — just pure dice math.
    int8_t dice_score[252][13];
};
```

### 2.1 Possible Scores Per Row

Enumerate all achievable non-zero scores for each row. These are fixed by the rules:

| Row | Possible non-zero scores |
|-----|-------------------------|
| 1s | 1, 2, 3, 4, 5 |
| 2s | 2, 4, 6, 8, 10 |
| 3s | 3, 6, 9, 12, 15 |
| 4s | 4, 8, 12, 16, 20 |
| 5s | 5, 10, 15, 20, 25 |
| 6s | 6, 12, 18, 24, 30 |
| SS | 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 |
| LS | 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 |
| FH | 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50 |
| K | 34, 38, 42, 46, 50, 54 |
| STR | 45, 50 |
| U8 | 60, 65, 70, 75 |
| Y | 75, 80, 85, 90, 95, 100 |

Note: 0 (scratch) is always implicitly available for every row and does not need to be stored in these tables.

### 2.2 Filtered Scores by Threshold

For each row `r` and each threshold `t` (0 to max score for that row), precompute an array containing all values from `possible_scores[r]` that are `>= t`. This allows the solver to instantly look up "which scores can I place in this row given the current Golden Rule minimum?" without any runtime filtering.

### 2.3 Dice State to Score Mapping

For each of the 252 sorted dice states (enumerated identically to the solver's existing `ID_TO_STATE` table) and each of the 13 rows, compute the raw score. This is pure dice math with no game state dependencies:

- **Rows 0–5 (Numbers):** Sum of dice showing face value `(row + 1)`. Always produces a score (could be 0 if no dice show that number, which is a valid score for the table — it means scratch).
- **Row 6 (SS):** Sum of all dice. Score = sum if sum >= 20 and sum <= 29, else 0.
- **Row 7 (LS):** Sum of all dice. Score = sum if sum >= 20, else 0.
- **Row 8 (FH):** Check for 3-of-a-kind + 2-of-a-kind pattern. Score = 20 + sum if qualifies, else 0.
- **Row 9 (K):** Check for 4-of-a-kind. Score = 30 + (face × 4) if qualifies, else 0.
- **Row 10 (STR):** Check for 1-2-3-4-5 (score 45) or 2-3-4-5-6 (score 50), else 0.
- **Row 11 (U8):** Sum of all dice. Score = 60 + 5 × (8 - sum) if sum <= 8, else 0.
- **Row 12 (Y):** Check for 5-of-a-kind. Score = 75 + 5 × (face - 1) if qualifies, else 0.

**Important edge cases for dice_score:**
- For number rows, if no dice show that face, dice_score = 0 (scratch score).
- LS upper bound: do NOT cap at 30 in the dice table. LS can be any sum >= 20. The cap and SS/LS interlock constraints are applied by `calculate_score` at placement time, not in the precomputed table.
- FH: 5-of-a-kind qualifies as a Full House (it has both a 3-of-a-kind and a 2-of-a-kind subset). Score = 20 + 5 × face.
- K: 5-of-a-kind qualifies as 4-of-a-kind. Score = 30 + face × 4.

### 2.4 Initialization

Implement a function `void init_solver_tables(SolverTables& tables)` that populates all fields. This is called once at program startup. It should use the same dice state enumeration as the solver (sorted ascending, 252 states total) so that dice state IDs are consistent across the engine and solver.

---

## 3. Core Functions

### 3.1 Board Initialization

```cpp
/// Initialize a fresh board with all cells empty and randomized coefficients.
/// Randomly selects the starting player (0 or 1).
/// @param rng A random engine for coefficient shuffling and starting player.
void init_board(BoardState& board, RNG& rng);

/// Initialize the GameContext to match a freshly initialized BoardState.
/// Sets up initial legal placements for both players, zeroes all caches.
void init_context(GameContext& ctx, const BoardState& board);
```

Initial legal placements per column type:
- **Down:** Only row 0 (1s)
- **Free:** All 13 rows
- **Up:** Only row 12 (Yams)
- **Mid:** Row 5 (6s) and row 6 (SS)
- **Turbo:** All 13 rows
- **UpDown:** All 13 rows

### 3.2 Score Calculation

```cpp
/// Calculate the score a player would receive for placing in a specific cell
/// with the given dice.
///
/// This function applies:
/// - Dice-based scoring (what the dice produce for this row)
/// - Golden Rule (score must >= golden_max for this col,row; otherwise returns 0)
/// - SS/LS interlock (returns 0 if paired cell is scratched or constraints violated)
///
/// It does NOT check column constraints (that's get_legal_placements' job).
///
/// @param row The target row (0-12)
/// @param dice Array of 5 dice values (1-6), need not be sorted
/// @param player The player placing (0 or 1)
/// @param column The target column (0-5)
/// @param board The current board state
/// @param ctx The current game context (for golden_max, ss/ls status)
/// @return The score (0 = scratch, >0 = valid score)
int calculate_score(int row, const int8_t dice[5],
                    int player, int column,
                    const BoardState& board, const GameContext& ctx);
```

**Logic for calculate_score:**

1. Compute the raw dice score for this row (can use the precomputed `SolverTables::dice_score` if dice are sorted, or compute inline).

2. If raw dice score is 0, return 0 (dice don't qualify → scratch).

3. **Golden Rule check:** If raw dice score < `ctx.golden_max[column][row]`, return 0 (doesn't meet threshold → scratch).

4. **SS-specific checks (row == 6):**
   a. If `ctx.ls_scratched[player][column]` is true, return 0 (paired LS is scratched → forced scratch).
   b. If LS is already filled in this column for this player (`board.cells[player][column][7] != kCellEmpty`), and raw dice score >= that LS value, return 0 (SS must be strictly less than own LS).
   c. Cap score at 29 (SS max).

5. **LS-specific checks (row == 7):**
   a. If `ctx.ss_scratched[player][column]` is true, return 0 (paired SS is scratched → forced scratch).
   b. If SS is already filled in this column for this player (`board.cells[player][column][6] != kCellEmpty`), and raw dice score <= that SS value, return 0 (LS must be strictly greater than own SS).

6. Return the validated score.

**Important notes:**
- The function does not modify any state. It is a pure query.
- For the solver's use case, this function will not be called on cloned afterstates. The solver uses `SolverTables::dice_score` and Golden Rule lookups directly for mapping dice states to scores. `calculate_score` is primarily used during actual gameplay.

### 3.3 Legal Placement Generation

```cpp
/// Rebuild the legal placements cache for a player from scratch.
/// Called during initialization and can be called for verification.
void rebuild_legal_placements(int player, const BoardState& board, GameContext& ctx);

/// Update the legal placements cache after a placement was made.
/// More efficient than rebuilding — only modifies affected entries.
void update_legal_placements_after_move(int player, int column, int row,
                                         const BoardState& board, GameContext& ctx);
```

**Column constraint logic for legal placement:**

A cell (column, row) is a legal placement for a player if:
1. The cell is empty: `board.cells[player][column][row] == kCellEmpty`
2. The column constraint allows it (see below)

**Column-specific rules:**

- **Down (column 0):** The legal row is the lowest-indexed empty row. Only one row is legal at a time. Scan from row 0 upward to find the first empty cell.

- **Free (column 1):** Any empty cell is legal.

- **Up (column 2):** The legal row is the highest-indexed empty row. Only one row is legal at a time. Scan from row 12 downward to find the first empty cell.

- **Mid (column 3):**
  - If no cells are filled: only row 5 (6s) and row 6 (SS) are legal.
  - Otherwise: any empty cell that has at least one filled neighbor is legal. **Adjacency wraps:** row 0 and row 12 are considered adjacent.

- **Turbo (column 4):** Any empty cell is legal. (The 2-roll restriction is enforced at the turn level, not here.)

- **UpDown (column 5):**
  - If no cells are filled: all 13 rows are legal.
  - Otherwise: any empty cell that has at least one filled neighbor is legal. **Adjacency wraps:** row 0 and row 12 are considered adjacent.

**Circular adjacency helper:**

```cpp
/// Check if a row has a filled neighbor in the given column for the given player.
/// Adjacency wraps: row 0 and row 12 are neighbors.
bool has_filled_neighbor(int player, int column, int row, const BoardState& board);
```

Implementation: check `board.cells[player][column][(row - 1 + 13) % 13]` and `board.cells[player][column][(row + 1) % 13]`. If either is != `kCellEmpty`, return true.

**Updating after a move:**

When player `p` places at (column `c`, row `r`):

1. Remove (c, r) from `legal_all[p]`: scan `placements[]` for the entry, swap with last, decrement count, set `is_legal[c][r] = false`.
2. If column `c` is not Turbo: also remove (c, r) from `legal_no_turbo[p]` the same way.
3. Based on column type, determine newly legal cells:
   - **Down:** Add (c, r+1) if r+1 <= 12 and cell is empty.
   - **Up:** Add (c, r-1) if r-1 >= 0 and cell is empty.
   - **Free/Turbo:** No new cells become legal.
   - **Mid/UpDown:** Check neighbors of (c, r) with wrapping. For each empty neighbor, check `is_legal[c][neighbor_row]` in `legal_all[p]` — if not already legal, add it.
4. For each newly legal cell: add to `legal_all[p]` (set `is_legal`, append to `placements[]`, increment count). If the cell is in a non-Turbo column, also add to `legal_no_turbo[p]`.

The `is_legal` bitset in `LegalPlacementCache` provides O(1) duplicate avoidance for Mid/UpDown neighbor additions.

### 3.4 Applying a Placement

```cpp
/// Apply a placement to the board and update all cached state.
/// This is the main mutation function — everything flows through here.
///
/// @param player The player making the placement (0 or 1)
/// @param column The column (0-5)
/// @param row The row (0-12)
/// @param score The score to place (0 = scratch, >0 = valid score)
/// @param board The board state to modify
/// @param ctx The game context to update
void apply_placement(int player, int column, int row, int score,
                     BoardState& board, GameContext& ctx);
```

**Steps in apply_placement:**

1. **Write the cell:**
   `board.cells[player][column][row] = score;`

2. **Increment cells_filled:**
   `board.cells_filled++;`

3. **Update Golden Rule cache:**
   If `score > ctx.golden_max[column][row]`, update it.

4. **Update upper section sum (rows 0-5 only):**
   If `row <= 5`: `ctx.upper_sum[player][column] += score;`

5. **Update SS/LS scratch tracking:**
   If `row == 6` (SS) and `score == 0`:
     - Set `ctx.ss_scratched[player][column] = true`
     - If LS (row 7) is already filled with a non-zero score:
       Scratch it: `board.cells[player][column][7] = 0;`
       Set `ctx.ls_scratched[player][column] = true;`
       Update `ctx.lower_has_scratch[player][column] = true;`
       Recalculate `ctx.golden_max[column][7]` (it may have decreased)
       Update `ctx.upper_sum` if needed (LS is lower section, so no)
   If `row == 7` (LS) and `score == 0`:
     - Set `ctx.ls_scratched[player][column] = true`
     - If SS (row 6) is already filled with a non-zero score:
       Scratch it: `board.cells[player][column][6] = 0;`
       Set `ctx.ss_scratched[player][column] = true;`
       Update `ctx.lower_has_scratch[player][column] = true;`
       Recalculate `ctx.golden_max[column][6]`

6. **Update clean column tracking (rows 6-12 only):**
   If `row >= 6` and `score == 0`:
     `ctx.lower_has_scratch[player][column] = true;`

7. **Update legal placements:**
   Call `update_legal_placements_after_move(player, column, row, board, ctx);`

**Important: Golden Rule recalculation on mutual destruction.**
When SS/LS mutual destruction scratches an already-filled cell (overwriting a non-zero score with 0), the `golden_max` for that (column, row) might decrease. Recalculate by scanning both players' cells for that (column, row):
```cpp
ctx.golden_max[column][row] = max(
    board.cells[0][column][row] > 0 ? board.cells[0][column][row] : 0,
    board.cells[1][column][row] > 0 ? board.cells[1][column][row] : 0
);
```

Note: After mutual destruction scratches a cell, the opponent's legal placements are not affected (the scratched cell was already filled). However, the changed `golden_max` might affect future `calculate_score` results for that (column, row) — this is handled automatically since `calculate_score` reads `golden_max` on every call.

---

## 4. Duel Computation

```cpp
/// Compute the final duel result for a completed game.
/// Returns the total duel points from player 0's perspective.
/// Positive = player 0 wins, negative = player 1 wins, zero = draw.
///
/// For training: result is converted to 1.0 (win), 0.0 (loss), 0.5 (draw).
///
/// @param board A fully filled board (all 156 cells filled)
/// @param ctx The game context (for upper sums and scratch tracking)
/// @return Total duel points for player 0
int compute_duel(const BoardState& board, const GameContext& ctx);
```

**Duel algorithm (per column):**

For each column `c` (0–5):

### Step 1: Raw Scores

For each player `p`:
```
raw_score[p] = sum of all 13 cells in column c for player p
             + upper_section_bonus(ctx.upper_sum[p][c])
```

Upper section bonus function:
```cpp
int upper_section_bonus(int sum) {
    if (sum >= 100) return 500;
    if (sum >= 90)  return 200;
    if (sum >= 80)  return 100;
    if (sum >= 70)  return 50;
    if (sum >= 60)  return 30;
    return 0;
}
```

### Step 2: Crush Multiplier

```cpp
int crush_multiplier(int my_raw, int opp_raw) {
    if (opp_raw == 0 && my_raw > 0) return 5;
    if (opp_raw > 0 && my_raw >= 5 * opp_raw) return 5;
    if (opp_raw > 0 && my_raw >= 4 * opp_raw) return 4;
    if (opp_raw > 0 && my_raw >= 3 * opp_raw) return 3;
    if (opp_raw > 0 && my_raw >= 2 * opp_raw) return 2;
    return 1;
}
```

Compute crush from player 0's perspective and player 1's perspective. At most one can have a crush > 1 (if both raw scores are > 0, only the higher scorer can crush).

### Step 3: Clean Column Bonus

For each player, check clean column eligibility:
```
is_clean[p] = (ctx.upper_sum[p][c] >= 60) && (!ctx.lower_has_scratch[p][c])
```

The bonus value depends on whether ANY crush multiplier is active in this column:
- If crush > 1 (either direction): clean bonus = 100
- If no crush (both are 1×): clean bonus = 200

```
adjusted_score[p] = raw_score[p] + (is_clean[p] ? clean_bonus : 0)
```

### Step 4: Final Duel Points

```
difference = adjusted_score[0] - adjusted_score[1]
active_crush = max(crush_0, crush_1)  // only one can be > 1
duel_points = difference * active_crush * board.coefficients[c]
```

**Sum across all 6 columns** for the total game result.

### Step 5: Win/Loss/Draw Conversion

```cpp
double game_result(int total_duel_points) {
    if (total_duel_points > 0) return 1.0;   // player 0 wins
    if (total_duel_points < 0) return 0.0;   // player 1 wins
    return 0.5;                               // draw
}
```

---

## 5. Utility Functions

### 5.1 Dice Utilities

```cpp
/// Sort dice in ascending order (in-place).
void sort_dice(int8_t dice[5]);

/// Compute frequency counts for each face value (1-6).
/// counts[0] is unused; counts[1] through counts[6] hold the count for each face.
void dice_counts(const int8_t dice[5], int counts[7]);

/// Compute sum of all dice.
int dice_sum(const int8_t dice[5]);
```

### 5.2 Board Queries

```cpp
/// Check if the game is over (all cells filled).
bool is_terminal(const BoardState& board);

/// Get the number of cells remaining for a player.
int cells_remaining(const BoardState& board, int player);

/// Get the number of cells remaining in a specific column for a player.
int column_cells_remaining(const BoardState& board, int player, int column);
```

---

## 6. File Organization

```
src/engine/
├── constants.h              # Enums, constants
├── board_state.h             # BoardState struct
├── game_state.h              # GameState struct (extends BoardState with dice)
├── game_context.h            # GameContext struct (cached derived data)
├── solver_tables.h           # SolverTables struct
├── solver_tables.cc          # init_solver_tables() implementation
├── scoring.h                 # calculate_score, dice utilities
├── scoring.cc                # Scoring implementation
├── legal_moves.h             # get_legal_placements, update functions
├── legal_moves.cc            # Legal move implementation
├── placement.h               # apply_placement
├── placement.cc              # Placement implementation
├── duel.h                    # compute_duel, upper_section_bonus
├── duel.cc                   # Duel implementation
├── board_init.h              # init_board, init_context
└── board_init.cc             # Initialization implementation
```

Update `src/engine/CMakeLists.txt` to compile all `.cc` files:
```cmake
add_library(engine STATIC
    solver_tables.cc
    scoring.cc
    legal_moves.cc
    placement.cc
    duel.cc
    board_init.cc
)
target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/src)
```

---

## 7. Unit Tests

All tests go in `tests/engine/`. Each test file corresponds to a source module. Use Google Test fixtures where appropriate.

### 7.1 Scoring Tests (`tests/engine/scoring_test.cc`)

**Number rows (0–5):**
- Dice {3,3,3,1,2} → row 3s → score = 9 (three 3s)
- Dice {6,6,6,6,6} → row 6s → score = 30
- Dice {1,2,3,4,5} → row 6s → score = 0 (no 6s)
- Dice {1,1,1,1,1} → row 1s → score = 5

**SS (row 6):**
- Dice {4,4,4,4,4} (sum=20) → score = 20 (meets minimum)
- Dice {3,3,3,3,3} (sum=15) → score = 0 (below minimum 20)
- Dice {6,6,6,6,6} (sum=30) → score = 29 (capped)
- SS when LS is already scratched → forced to 0
- SS with value >= own LS → returns 0

**LS (row 7):**
- Dice {6,6,6,6,6} (sum=30) → score = 30
- Dice {3,3,3,3,3} (sum=15) → score = 0
- LS when SS is already scratched → forced to 0
- LS with value <= own SS → returns 0

**FH (row 8):**
- Dice {3,3,3,2,2} (FH, sum=13) → score = 20 + 13 = 33
- Dice {5,5,5,5,5} (Yams = valid FH, sum=25) → score = 20 + 25 = 45
- Dice {1,2,3,4,5} (no FH) → score = 0

**K (row 9):**
- Dice {4,4,4,4,1} → score = 30 + 16 = 46
- Dice {6,6,6,6,6} (5-of-a-kind = valid K) → score = 30 + 24 = 54
- Dice {3,3,3,2,2} (FH but not K) → score = 0

**STR (row 10):**
- Dice {1,2,3,4,5} → score = 45
- Dice {2,3,4,5,6} → score = 50
- Dice {1,2,3,4,6} → score = 0

**U8 (row 11):**
- Dice {1,1,1,1,1} (sum=5) → score = 60 + 15 = 75
- Dice {1,1,1,1,4} (sum=8) → score = 60 + 0 = 60
- Dice {1,1,1,2,4} (sum=9) → score = 0 (exceeds 8)

**Y (row 12):**
- Dice {6,6,6,6,6} → score = 75 + 25 = 100
- Dice {1,1,1,1,1} → score = 75 + 0 = 75
- Dice {3,3,3,3,2} → score = 0

**Golden Rule tests:**
- Set golden_max[col][row] = 12 for row 3s. Dice {3,3,3,1,1} (score=9) → returns 0 (below threshold).
- Set golden_max[col][row] = 9 for row 3s. Same dice → returns 9 (meets threshold).

### 7.2 Legal Move Tests (`tests/engine/legal_moves_test.cc`)

**Down column:**
- Fresh board → only row 0 is legal
- After filling row 0 → only row 1 is legal
- After filling rows 0–12 → no legal placements

**Up column:**
- Fresh board → only row 12 is legal
- After filling row 12 → only row 11 is legal

**Free column:**
- Fresh board → all 13 rows legal
- After filling row 5 → 12 rows legal (all except row 5)

**Mid column:**
- Fresh board → rows 5 and 6 legal
- After filling row 5 → rows 4 and 6 legal (neighbors of 5)
- After filling rows 5 and 6 → rows 4 and 7 legal
- Wrapping test: fill rows 5 through 0 sequentially (starting from 5, expanding upward). Row 0 is now filled but row 12 is empty and has no filled neighbor via normal adjacency (row 11 is empty). Verify that row 12 is legal because it wraps to row 0 which is filled.

**UpDown column:**
- Fresh board → all 13 rows legal
- After filling row 7 → only rows 6 and 8 legal
- Wrapping test: fill only row 0 → rows 1 and 12 are legal (12 wraps to 0)
- Wrapping test: fill only row 12 → rows 0 and 11 are legal

**Turbo column:**
- Behaves exactly like Free for legal move purposes

### 7.3 Placement Tests (`tests/engine/placement_test.cc`)

**Basic placement:**
- Place score 15 at (player=0, col=1, row=4). Verify cell is set, cells_filled incremented, golden_max updated, legal placements updated.

**SS/LS mutual destruction:**
- Place SS=0 (scratch). Verify ss_scratched flag set.
- Then fill LS with score 25. Verify LS is scratched to 0, ls_scratched flag set, golden_max recalculated.
- Reverse order: scratch LS first, then fill SS → SS forced to 0.
- Scratch SS when LS is empty → ss_scratched=true, LS not yet scratched but will be forced when filled.

**Golden Rule update:**
- Place score 18 in (col=1, row=5). Verify golden_max[1][5] = 18.
- Place score 24 in same (col=1, row=5) for other player. Verify golden_max[1][5] = 24.

**Upper sum tracking:**
- Place several upper section scores, verify upper_sum accumulates correctly.

**Lower scratch tracking:**
- Place score 0 in a lower section row. Verify lower_has_scratch is set.
- Place non-zero scores in all lower rows. Verify lower_has_scratch stays false.

### 7.4 Duel Tests (`tests/engine/duel_test.cc`)

**Simple duel:**
- Set up a completed board where player 0 has raw score 200 and player 1 has raw score 100 in column 0 (coefficient 10). No crush (200 < 2×100). No clean bonuses. Expected: (200 - 100) × 1 × 10 = 1000.

**Crush multiplier:**
- Player 0 raw = 250, player 1 raw = 100 in a column. 250 >= 2×100. Crush = 2×. Expected difference × 2 × coefficient.
- Player 0 raw = 1, player 1 raw = 0. Crush = 5×.

**Clean column bonus:**
- Player 0 has clean column, crush is active → +100 to player 0's adjusted score.
- Player 0 has clean column, no crush → +200.

**Upper section bonus thresholds:**
- Test each threshold: 59→0, 60→30, 70→50, 80→100, 90→200, 100→500.

**Full game duel:**
- Set up a complete board with known values in all 6 columns, compute expected result manually, verify compute_duel matches.

### 7.5 SolverTables Tests (`tests/engine/solver_tables_test.cc`)

**Possible scores:**
- Verify count and values for each row match the expected sets.

**Dice-to-score mapping:**
- For several known dice states, verify scores match expected values across all rows.

**Filtered scores:**
- For row 6s with threshold 13: verify result is {18, 24, 30}.
- For row Yams with threshold 0: verify all 6 Yams scores are included.
- For row Yams with threshold 100: verify only {100} is returned.
- For row Yams with threshold 101: verify empty result.

### 7.6 Board Initialization Tests (`tests/engine/init_test.cc`)

- Verify all cells start as kCellEmpty.
- Verify coefficients are a permutation of {8, 10, 12, 14, 16, 18}.
- Verify current_player is 0 or 1.
- Verify cells_filled starts at 0.
- Verify initial legal placements match column type rules.
- Verify golden_max is all zeros.
- Verify upper_sum is all zeros.

---

## 8. Benchmarks

Add benchmarks in `tests/benchmarks/engine_bench.cc`:

- **BM_CalculateScore:** Benchmark `calculate_score` for various row types.
- **BM_GetLegalPlacements:** Benchmark `rebuild_legal_placements` from a mid-game board state.
- **BM_ApplyPlacement:** Benchmark `apply_placement` including all cache updates.
- **BM_ComputeDuel:** Benchmark `compute_duel` on a completed board.
- **BM_BoardStateCopy:** Benchmark copying a `BoardState` struct (should be ~164 bytes memcpy).
- **BM_InitSolverTables:** Benchmark the one-time solver table initialization.

These benchmarks establish performance baselines. Any future changes to the engine should be validated against these baselines.

---

## 9. Definition of Done

This task is complete when:

1. All data structures (`BoardState`, `GameState`, `GameContext`, `SolverTables`) are implemented with correct sizes and `static_assert` on trivial copyability.
2. `init_solver_tables` populates all precomputed tables correctly (verified by tests).
3. `calculate_score` handles all 13 row types, Golden Rule, and SS/LS interlock (verified by tests).
4. `get_legal_placements` / `rebuild_legal_placements` / `update_legal_placements_after_move` correctly handle all 6 column types including Mid/UpDown circular wrapping (verified by tests).
5. `apply_placement` correctly updates all caches including SS/LS mutual destruction and Golden Rule recalculation (verified by tests).
6. `compute_duel` correctly implements the full duel algorithm including crush multipliers, clean column bonuses, and upper section bonuses (verified by tests).
7. All unit tests pass.
8. All benchmarks run and produce baseline numbers.
9. The engine library compiles with zero libtorch dependencies.
10. No heap allocations in any hot-path function (BoardState copy, calculate_score, legal move checks).
