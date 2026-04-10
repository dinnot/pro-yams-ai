# Task 03: Game Engine — Game Flow

## Overview

Wire the engine components from Task 02 into a complete playable game flow. This task implements dice rolling, the hold/reroll mechanism, turn management, Turbo column enforcement, and the public API that callers (solver, heuristic bot, self-play, UI) use to play a game.

The engine provides mutation functions. The caller is responsible for all decision-making — the engine doesn't know or care whether decisions come from a neural network, a heuristic bot, or a human.

## Prerequisites

- Task 02 completed (all data structures, scoring, legal moves, placement, duel computation)

---

## 1. Additional GameContext Fields

Add the following to `GameContext` (from Task 02):

```cpp
// Add to GameContext:

// Number of non-turbo cells remaining per player.
// Decremented on each placement in columns 0-3 and 5 (all except Turbo/column 4).
// When this reaches 0, only Turbo cells remain.
// Used for can_reroll optimization and Turbo-aware legal placements.
int8_t non_turbo_cells_remaining[2];    // 2 bytes
```

Initialize to 65 per player (13 rows × 5 non-Turbo columns). Decremented in `apply_placement` whenever the placement is in a non-Turbo column.

---

## 2. Updated Legal Placements

### 2.1 Legal Placement Cache Structure

The legal placement cache uses a flat array for fast iteration (the solver's hot path) combined with a bitset for O(1) membership queries:

```cpp
struct LegalPlacementCache {
    Placement placements[78];    // Dense array for iteration — max 78 legal cells
    int8_t count;                // Number of valid entries in placements[]
    bool is_legal[6][13];        // O(1) lookup: is (column, row) currently legal?
};
```

Operations:
- **Iteration:** Loop over `placements[0..count-1]`. Perfect cache locality.
- **Query:** `is_legal[col][row]`. O(1).
- **Removal:** Scan `placements[]` for the entry (N ≤ 78, cache-friendly linear scan), swap with last element, decrement count, clear `is_legal[col][row]`.
- **Addition:** Set `is_legal[col][row] = true`, append to `placements[count]`, increment count. O(1). Check `is_legal` first to avoid duplicates.

Zero heap allocation. The entire structure is pre-allocated.

### 2.2 Two Separate Caches (Option A)

Maintain two caches per player, both updated incrementally on every placement:

```cpp
// Add to GameContext:

// All legal placements (including Turbo column)
LegalPlacementCache legal_all[2];

// Non-Turbo legal placements only (for when rolls_left == 0)
LegalPlacementCache legal_no_turbo[2];
```

Both caches are updated in `update_legal_placements_after_move`. When a placement is made:
- Always update `legal_all` (remove the placed cell, add newly legal neighbors for Mid/UpDown)
- If the placed cell is non-Turbo: update `legal_no_turbo` the same way
- If newly legal neighbors are in non-Turbo columns: add them to `legal_no_turbo` as well
- If the placed cell is in the Turbo column: only update `legal_all` (the cell was never in `legal_no_turbo`)

### 2.3 Roll-Aware Legal Placements

```cpp
/// Get legal placements considering roll state.
/// When rolls_left == 0, returns the non-Turbo cache.
/// Otherwise, returns the full cache.
///
/// @param player The player (0 or 1)
/// @param rolls_left Remaining rerolls (0, 1, or 2)
/// @param ctx The game context
/// @param out_count Output: number of legal placements returned
/// @return Pointer to array of Placement structs
const Placement* get_legal_placements(int player, int rolls_left,
                                       const GameContext& ctx,
                                       int8_t& out_count);
```

**Implementation:**

```cpp
const Placement* get_legal_placements(int player, int rolls_left,
                                       const GameContext& ctx,
                                       int8_t& out_count) {
    if (rolls_left > 0) {
        out_count = ctx.legal_all[player].count;
        return ctx.legal_all[player].placements;
    } else {
        out_count = ctx.legal_no_turbo[player].count;
        return ctx.legal_no_turbo[player].placements;
    }
}
```

O(1) — just returns a pointer. No filtering, no allocation.

### 2.4 can_reroll

```cpp
/// Check if the current player can reroll.
/// Returns false if:
///   - rolls_left == 0
///   - rolls_left == 1 AND only Turbo cells remain (non_turbo_cells_remaining == 0)
///
/// @param state The current game state
/// @param ctx The game context
/// @return true if rerolling is allowed
bool can_reroll(const GameState& state, const GameContext& ctx);
```

**Implementation:**

```cpp
bool can_reroll(const GameState& state, const GameContext& ctx) {
    if (state.rolls_left <= 0) return false;
    if (state.rolls_left == 1 &&
        ctx.non_turbo_cells_remaining[state.board.current_player] == 0) {
        return false;
    }
    return true;
}
```

This is O(1) thanks to the `non_turbo_cells_remaining` counter.

---

## 3. Core Game Flow Functions

### 3.1 Game Initialization

```cpp
/// Initialize a new game. Sets up the board, context, rolls initial dice
/// for the starting player. After this call, the game is ready for the
/// first decision (hold or place).
///
/// @param state Output: the initialized game state (board + dice + rolls_left)
/// @param ctx Output: the initialized game context
/// @param rng Random engine for coefficient shuffling, starting player, and dice
void init_game(GameState& state, GameContext& ctx, RNG& rng);
```

**Steps:**
1. Call `init_board(state.board, rng)` — shuffles coefficients, picks starting player.
2. Call `init_context(ctx, state.board)` — sets up caches, initial legal placements.
3. Initialize `ctx.non_turbo_cells_remaining[0] = 65; ctx.non_turbo_cells_remaining[1] = 65;`
4. Roll all 5 dice: for each die, `state.dice[i] = uniform(1, 6)`.
5. Set `state.rolls_left = 2` (initial roll done, 2 rerolls remaining).

### 3.2 Perform Hold

```cpp
/// Hold the specified dice and reroll the rest.
/// Decrements rolls_left. The caller must check can_reroll() before calling.
///
/// @param hold Bitmask: bit i set means die i is held (not rerolled).
///             E.g., 0b00101 = hold dice 0 and 2, reroll dice 1, 3, 4.
/// @param state The game state to modify (dice and rolls_left updated)
/// @param rng Random engine for rerolling
void perform_hold(uint8_t hold_mask, GameState& state, RNG& rng);
```

**Implementation:**

```cpp
void perform_hold(uint8_t hold_mask, GameState& state, RNG& rng) {
    assert(state.rolls_left > 0);  // Debug-only check

    for (int i = 0; i < 5; ++i) {
        if (!((hold_mask >> i) & 1)) {
            state.dice[i] = uniform_int(rng, 1, 6);
        }
    }
    state.rolls_left--;
}
```

**Notes:**
- `hold_mask = 0b11111` (hold all) is technically valid — the player keeps all dice and loses a reroll. This is sometimes strategically correct.
- `hold_mask = 0b00000` (hold none) is valid — reroll all dice.
- The caller is responsible for calling `can_reroll()` first. In debug builds, the assert catches violations.

### 3.3 Perform Placement

```cpp
/// Place a score in the specified cell, update all game state, switch to the
/// next player, and roll dice for them (if the game is not over).
///
/// @param placement The (column, row) to fill
/// @param score The score to place (0 = scratch, computed by caller via calculate_score)
/// @param state The game state to modify
/// @param ctx The game context to update
/// @param rng Random engine for rolling next player's dice
void perform_placement(const Placement& placement, int score,
                       GameState& state, GameContext& ctx, RNG& rng);
```

**Steps:**

1. **Apply the placement** to the board and update caches:
   Call `apply_placement(state.board.current_player, placement.column, placement.row, score, state.board, ctx);`

2. **Update non-Turbo counter** (if not Turbo column):
   ```cpp
   if (placement.column != kColTurbo) {
       ctx.non_turbo_cells_remaining[state.board.current_player]--;
   }
   ```

3. **Mark legal_no_turbo as dirty:**
   ```cpp
   ctx.legal_no_turbo_dirty[state.board.current_player] = true;
   ```

4. **Check if game is over:**
   ```cpp
   if (state.board.cells_filled >= kTotalCells) {
       // Game over — no dice roll, no player switch
       // Caller should call compute_duel() to get the result
       return;
   }
   ```

5. **Switch to next player:**
   ```cpp
   state.board.current_player = 1 - state.board.current_player;
   ```

6. **Roll dice for the new current player:**
   ```cpp
   for (int i = 0; i < 5; ++i) {
       state.dice[i] = uniform_int(rng, 1, 6);
   }
   state.rolls_left = 2;
   ```

7. **Mark the new player's legal_no_turbo as dirty:**
   ```cpp
   ctx.legal_no_turbo_dirty[state.board.current_player] = true;
   ```

---

## 4. Game Result

```cpp
/// Get the game result after a completed game.
/// Must only be called when is_terminal() returns true.
///
/// @param state The final game state
/// @param ctx The final game context
/// @return 1.0 if player 0 wins, 0.0 if player 1 wins, 0.5 if draw
double get_game_result(const GameState& state, const GameContext& ctx);
```

**Implementation:**
```cpp
double get_game_result(const GameState& state, const GameContext& ctx) {
    assert(is_terminal(state.board));
    int duel_points = compute_duel(state.board, ctx);
    if (duel_points > 0) return 1.0;
    if (duel_points < 0) return 0.0;
    return 0.5;
}
```

---

## 5. Complete Caller Flow

Here's how a caller plays a complete game (pseudocode):

```cpp
GameState state;
GameContext ctx;
RNG rng(seed);

// Initialize — dice are already rolled for starting player
init_game(state, ctx, rng);

while (!is_terminal(state.board)) {
    // Get current player's perspective
    int player = state.board.current_player;

    // Decision loop for this turn
    while (true) {
        // Can we reroll?
        if (can_reroll(state, ctx)) {
            // Ask the decision maker: hold or place?
            // (This is where solver/bot/human logic goes)
            Action action = decide(state, ctx);

            if (action.type == HOLD) {
                perform_hold(action.hold_mask, state, rng);
                continue;  // Go back to decision
            }
        }

        // Place a score
        // Get legal placements (roll-aware)
        int8_t count;
        auto* placements = get_legal_placements(player, state.rolls_left, ctx, count);

        // Decision maker picks a placement
        Placement chosen = choose_placement(placements, count, state, ctx);

        // Calculate score for the chosen cell
        int score = calculate_score(chosen.row, state.dice, player,
                                     chosen.column, state.board, ctx);

        // Execute placement (switches player, rolls dice for next turn)
        perform_placement(chosen, score, state, ctx, rng);
        break;  // Turn is over
    }
}

// Game is over
double result = get_game_result(state, ctx);
```

---

## 6. RNG

Use xoshiro256++ — a fast, high-quality, deterministic PRNG. The reference implementation below is public domain, written by David Blackman and Sebastiano Vigna.

### 6.1 Reference Implementation

Copy the following reference C code into the project and wrap it in a C++ class. Do not modify the core algorithm.

```c
/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#include <stdint.h>

/* This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests we are
   aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}


static uint64_t s[4];

uint64_t next(void) {
	const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

	const uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return result;
}


/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */

void jump(void) {
	static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			next();	
		}
		
	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}



/* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */

void long_jump(void) {
	static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (LONG_JUMP[i] & UINT64_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			next();	
		}
		
	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}
```

### 6.2 C++ Wrapper

Wrap the reference implementation in a class. The key changes from the C version:
- Move the static `s[4]` state into the class as a member (so multiple RNG instances are independent)
- Seed using splitmix64 as recommended by the authors
- Add `uniform_int` using Lemire's fast bounded integer method
- Add `shuffle` using Fisher-Yates

```cpp
// src/engine/rng.h

class RNG {
public:
    /// Construct with a 64-bit seed.
    /// Uses splitmix64 to expand the seed into the 256-bit state,
    /// as recommended by the xoshiro authors.
    explicit RNG(uint64_t seed);

    /// Generate a raw 64-bit random number (the core xoshiro256++ algorithm).
    uint64_t next();

    /// Generate uniform int in [min, max] inclusive.
    /// Uses Lemire's nearly divisionless method for unbiased bounded integers.
    int uniform_int(int min, int max);

    /// Shuffle an array in-place (Fisher-Yates algorithm).
    template<typename T, size_t N>
    void shuffle(std::array<T, N>& arr);

private:
    uint64_t state_[4];
};
```

**Splitmix64 seeding** (used in the constructor to expand a single uint64_t seed into 4 × uint64_t state):

```cpp
// Standard splitmix64 — also public domain, by Sebastiano Vigna
static uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

RNG::RNG(uint64_t seed) {
    state_[0] = splitmix64(seed);
    state_[1] = splitmix64(seed);
    state_[2] = splitmix64(seed);
    state_[3] = splitmix64(seed);
}
```

**Lemire's method for uniform_int:**

```cpp
int RNG::uniform_int(int min, int max) {
    uint32_t range = static_cast<uint32_t>(max - min + 1);
    uint64_t x = next();
    // Use upper 32 bits for the multiply method
    uint32_t hi = static_cast<uint32_t>(x >> 32);
    uint64_t m = static_cast<uint64_t>(hi) * range;
    uint32_t l = static_cast<uint32_t>(m);
    if (l < range) {
        uint32_t t = (-range) % range;
        while (l < t) {
            x = next();
            hi = static_cast<uint32_t>(x >> 32);
            m = static_cast<uint64_t>(hi) * range;
            l = static_cast<uint32_t>(m);
        }
    }
    return min + static_cast<int>(m >> 32);
}
```

### 6.3 Why xoshiro256++ and not std::mt19937

- **Faster:** Fewer operations per random number (~1ns vs ~3ns)
- **Smaller state:** 32 bytes vs 2.5KB (matters if RNG is embedded in game state or passed across threads)
- **Deterministic:** Same seed produces identical sequence across platforms
- **Quality:** Passes all known statistical tests (BigCrush, PractRand)

---

## 7. File Organization

```
src/engine/
├── ... (all files from Task 02)
├── rng.h                # RNG class declaration
├── rng.cc               # RNG implementation
├── game_flow.h          # init_game, perform_hold, perform_placement,
│                        # can_reroll, get_game_result, get_legal_placements (roll-aware)
└── game_flow.cc         # Game flow implementation
```

Update `src/engine/CMakeLists.txt` to include the new source files:
```cmake
add_library(engine STATIC
    solver_tables.cc
    scoring.cc
    legal_moves.cc
    placement.cc
    duel.cc
    board_init.cc
    rng.cc
    game_flow.cc
)
```

---

## 8. Unit Tests

### 8.1 Game Flow Tests (`tests/engine/game_flow_test.cc`)

**Basic game initialization:**
- After `init_game`, verify dice are in range [1,6], `rolls_left == 2`, current_player is 0 or 1, board is empty.
- Verify `non_turbo_cells_remaining` starts at 65 for both players.

**Hold and reroll:**
- Call `perform_hold` with mask `0b11111` (hold all). Verify dice unchanged, `rolls_left` decremented.
- Call `perform_hold` with mask `0b00000` (hold none). Verify all dice changed (probabilistically — run with a seeded RNG and verify the output matches expected values for that seed).
- Call `perform_hold` with mask `0b00101`. Verify dice 0 and 2 unchanged, dice 1, 3, 4 rerolled.

**Placement and player switching:**
- Perform a placement for player 0. Verify:
  - Cell is filled with correct score
  - `current_player` switched to 1
  - New dice are rolled (values in [1,6])
  - `rolls_left` is 2 for the new player
  - `cells_filled` incremented

**Game completion:**
- Play a full game (78 turns × 2 players = 156 placements) using random/deterministic choices. Verify:
  - `is_terminal` returns true after exactly 156 placements
  - `get_game_result` returns 0.0, 0.5, or 1.0
  - No assertions fire during the entire game

**Non-turbo counter:**
- Place in a non-Turbo column. Verify `non_turbo_cells_remaining` decremented.
- Place in Turbo column. Verify `non_turbo_cells_remaining` unchanged.

### 8.2 Turbo Restriction Tests (`tests/engine/turbo_test.cc`)

**Turbo excluded at rolls_left == 0:**
- Set up a board mid-game with Turbo cells available. Set `rolls_left = 0`.
- Call `get_legal_placements`. Verify no Turbo column entries in the result.
- Set `rolls_left = 1`. Verify Turbo entries are present.

**can_reroll basic cases:**
- `rolls_left = 2` → `can_reroll` returns true
- `rolls_left = 1` with non-Turbo cells remaining → `can_reroll` returns true
- `rolls_left = 0` → `can_reroll` returns false

**can_reroll Turbo-only edge case:**
- Set up a board where all non-Turbo cells are filled for the current player (only Turbo cells remain).
- `rolls_left = 1` → `can_reroll` returns false (rerolling would leave rolls_left = 0, and only Turbo cells are available = no legal placements).
- `rolls_left = 2` → `can_reroll` returns true (after reroll, rolls_left = 1, and Turbo cells are legal at rolls_left > 0).

**Full Turbo-only game scenario:**
- Fill all non-Turbo cells for a player. On their next turns, verify:
  - They get 2 rolls total (initial + 1 reroll) before they must place
  - They cannot reroll a second time
  - Turbo placements succeed

### 8.3 RNG Tests (`tests/engine/rng_test.cc`)

**Determinism:**
- Two RNG instances with the same seed produce identical sequences.

**Range:**
- `uniform_int(1, 6)` produces only values 1–6 over 10,000 calls.
- Distribution is approximately uniform (chi-squared test or simple frequency check).

**Shuffle:**
- Shuffling {8, 10, 12, 14, 16, 18} produces a valid permutation.
- Same seed produces same permutation.

### 8.4 Full Game Integration Test (`tests/engine/full_game_test.cc`)

**Random full game:**
- Play a complete game with seeded RNG and random decisions (random legal placements, random holds).
- Verify all invariants hold throughout:
  - Cell values are never overwritten (once filled, stays filled)
  - `cells_filled` matches actual count of non-empty cells
  - `golden_max` is accurate at every step
  - `legal_placements` is accurate at every step (cross-check with `rebuild_legal_placements`)
  - `upper_sum` matches actual upper section sums
  - `ss_scratched` and `ls_scratched` flags are consistent with cell values
  - `non_turbo_cells_remaining` matches actual count
- Run multiple games with different seeds.

**Invariant verification helper:**
```cpp
/// Verify all GameContext caches are consistent with the BoardState.
/// Rebuilds everything from scratch and compares.
/// For use in tests only — not for production.
void verify_invariants(const BoardState& board, const GameContext& ctx);
```

This function:
1. Rebuilds legal placements from scratch, compares with cached version
2. Recomputes golden_max from cells, compares
3. Recomputes upper_sum, compares
4. Recomputes ss/ls_scratched, compares
5. Recomputes lower_has_scratch, compares
6. Recomputes non_turbo_cells_remaining, compares

Call this after every placement in the integration test. It's slow but thorough — exactly what tests are for.

---

## 9. Benchmarks

Add to `tests/benchmarks/engine_bench.cc`:

- **BM_FullGameRandom:** Benchmark playing a complete game with random decisions. This measures the end-to-end engine overhead per game (excluding any NN inference). Target: understand how many games/sec the engine can run in pure simulation mode.
- **BM_PerformHold:** Benchmark a single hold+reroll operation.
- **BM_PerformPlacement:** Benchmark a single placement operation (including all cache updates and dice roll for next player).
- **BM_CanReroll:** Benchmark the can_reroll check (should be near-zero).
- **BM_RNG:** Benchmark dice roll generation (5 dice).

---

## 10. Definition of Done

This task is complete when:

1. `init_game` creates a valid game state with dice rolled for the starting player.
2. `perform_hold` correctly holds/rerolls dice and decrements `rolls_left`.
3. `perform_placement` correctly updates the board, switches players, rolls dice for the next player, and handles game-over detection.
4. `can_reroll` correctly handles the Turbo-only edge case using the `non_turbo_cells_remaining` counter.
5. `get_legal_placements` correctly excludes Turbo cells when `rolls_left == 0`.
6. `get_game_result` returns correct win/loss/draw based on duel computation.
7. The RNG is deterministic, fast, and produces uniform distributions.
8. Full game integration test passes with invariant checking after every placement.
9. Multiple full games with different seeds complete without assertion failures.
10. All unit tests pass.
11. Benchmarks run and establish baseline for pure engine game throughput.
