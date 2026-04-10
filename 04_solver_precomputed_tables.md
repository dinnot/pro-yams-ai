# Task 04: Solver Precomputed Tables

## Overview

Build the precomputed lookup tables that the expectimax solver uses during gameplay. These tables are computed once at program startup and remain read-only for the lifetime of the process. They encode:

- All 252 unique sorted dice states
- The mapping from (dice_state, hold_mask) to held dice configuration
- Transition probability distributions from each held configuration to resulting dice states
- Score tables mapping dice states to scores per row (from Task 02's SolverTables)

This task ports and cleans up the table generation logic from the existing `pro_yams_solver.cc`, removing the OpenSpiel-specific condition/reward system and restructuring for cache-friendly flat arrays.

## Prerequisites

- Task 01 completed (project scaffolding)
- Task 02 completed (engine data structures, SolverTables struct, constants)

---

## 1. Dice State Enumeration

### 1.1 The 252 Sorted Dice States

All possible outcomes of 5 dice, represented as sorted ascending tuples. This is the number of 5-element multisets drawn from {1,2,3,4,5,6}, which is C(10,5) = 252.

Enumeration (identical to existing code):

```cpp
for (int a = 1; a <= 6; ++a)
  for (int b = a; b <= 6; ++b)
    for (int c = b; c <= 6; ++c)
      for (int d = c; d <= 6; ++d)
        for (int e = d; e <= 6; ++e)
          // state = {a, b, c, d, e}, id = sequential counter
```

### 1.2 Bidirectional Lookup

Two lookup structures for converting between dice arrays and state IDs:

```cpp
constexpr int kNumDiceStates = 252;
constexpr int kMaxLinearIndex = 7776;  // 6^5

// State ID (0-251) → sorted dice array
std::array<int, 5> id_to_state[kNumDiceStates];

// Linear index → State ID (-1 if not a valid sorted state)
// Linear index = (d[0]-1)*1296 + (d[1]-1)*216 + (d[2]-1)*36 + (d[3]-1)*6 + (d[4]-1)
int16_t linear_to_id[kMaxLinearIndex];
```

The linearization function:

```cpp
int linearize(const std::array<int, 5>& d) {
    return (d[0]-1)*1296 + (d[1]-1)*216 + (d[2]-1)*36 + (d[3]-1)*6 + (d[4]-1);
}
```

To look up the state ID for an arbitrary (potentially unsorted) set of dice: sort first, then linearize, then index into `linear_to_id`.

### 1.3 Unsorted Dice Lookup Helper

For convenience during gameplay (where dice are not necessarily sorted):

```cpp
/// Get the dice state ID for an arbitrary set of 5 dice values.
/// Sorts internally — does not modify the input.
int get_dice_state_id(const int8_t dice[5]);
```

Implementation: copy dice to a local array, sort, linearize, look up in `linear_to_id`.

---

## 2. Hold Masks and Held Configurations

### 2.1 Hold Masks

There are 32 possible hold masks (0–31) for 5 dice. Bit `i` set means die `i` is held (not rerolled). Mask `0b00000` = reroll all, `0b11111` = hold all.

### 2.2 Held Configurations

A "held configuration" is the unique sorted set of dice values being held. Multiple (dice_state, hold_mask) pairs can map to the same held configuration if they produce the same sorted subset of held values.

Example:
- Dice [1,2,3,4,5] with mask 0b00011 → holding {1,2}
- Dice [1,2,4,5,6] with mask 0b00011 → holding {1,2}
- Same held configuration, same reroll probability distribution.

The number of unique held configurations is computed at init time. It equals the total number of sorted multisubsets of sizes 0–5 drawn from {1,2,3,4,5,6}: C(6,0) + C(7,1) + C(8,2) + C(9,3) + C(10,4) + C(11,5) = 1 + 7 + 28 + 84 + 210 + 462 = 792. However, this is not hardcoded — it is determined dynamically during initialization by counting unique subsets.

### 2.3 MOVES Table

```cpp
// Maps (dice_state_id, hold_mask) → held_config_id
int16_t moves[kNumDiceStates][kNumHoldMasks];  // 252 × 32 = 8064 entries
constexpr int kNumHoldMasks = 32;
```

For each dice state and each hold mask, extract the held dice values, sort them, and look up (or assign) the held configuration ID.

**Initialization** (mirrors existing code logic):

```cpp
// Use a map during init to deduplicate held configurations
std::map<std::vector<int>, int> held_map;
int held_counter = 0;

for (int state = 0; state < kNumDiceStates; ++state) {
    for (int mask = 0; mask < kNumHoldMasks; ++mask) {
        std::vector<int> held_dice;
        for (int b = 0; b < 5; ++b) {
            if ((mask >> b) & 1) {
                held_dice.push_back(id_to_state[state][b]);
            }
        }
        // held_dice is already sorted (dice state is sorted, we extract in order)
        
        auto it = held_map.find(held_dice);
        if (it == held_map.end()) {
            held_map[held_dice] = held_counter;
            moves[state][mask] = held_counter;
            held_counter++;
        } else {
            moves[state][mask] = it->second;
        }
    }
}
// held_counter now contains the total number of unique held configurations
```

---

## 3. Transition Probabilities

### 3.1 Concept

For each held configuration (a sorted subset of 0–5 dice), the transition table describes the probability distribution over the 252 possible dice states after rerolling the unheld dice.

If holding N dice, we reroll (5 - N) dice. Each rerolled die is independently uniform over {1,2,3,4,5,6}. The total number of outcomes is 6^(5-N). Many outcomes map to the same sorted dice state, so we group them and compute probabilities.

Special case: holding all 5 dice (mask = 0b11111) → one transition with probability 1.0 to the current dice state.

### 3.2 Flat Contiguous Storage

For cache-friendly access during the solver's hot loop, store all transitions in a single contiguous array:

```cpp
struct Transition {
    int16_t target_state_id;   // 0–251: which dice state this leads to
    double probability;         // Probability of reaching that state
};

// All transitions stored contiguously, grouped by held_config_id.
// Access: transitions[offset[held_id] .. offset[held_id] + count[held_id] - 1]
std::vector<Transition> all_transitions;   // sized at init time
std::vector<int32_t> transition_offset;     // [held_config_id] → start index
std::vector<int16_t> transition_count;      // [held_config_id] → number of transitions
```

Note: `std::vector` is used here because the total number of transitions is determined at init time. These are allocated once and never resized after initialization. An alternative is to compute the max possible size and use flat arrays, but the one-time vector allocation is negligible.

### 3.3 Initialization

For each held configuration:

1. Determine how many dice need rerolling: `n_reroll = 5 - held_size`.
2. If `n_reroll == 0`: one transition to the held dice state with probability 1.0.
3. Otherwise, enumerate all `6^n_reroll` possible outcomes for the rerolled dice. For each outcome:
   a. Combine held dice + rerolled dice.
   b. Sort the combined set.
   c. Look up the state ID via linearize + `linear_to_id`.
   d. Increment a count for that state ID.
4. Convert counts to probabilities: `probability = count / 6^n_reroll`.
5. Store the (state_id, probability) pairs as transitions.

**Implementation** (mirrors existing code, using recursive enumeration):

```cpp
for (auto const& [held_vec, held_id] : held_map) {
    int n_reroll = 5 - held_vec.size();
    transition_offset[held_id] = all_transitions.size();
    
    if (n_reroll == 0) {
        // Holding all dice — deterministic transition
        std::array<int, 5> full;
        std::copy(held_vec.begin(), held_vec.end(), full.begin());
        int target = linear_to_id[linearize(full)];
        all_transitions.push_back({static_cast<int16_t>(target), 1.0});
        transition_count[held_id] = 1;
        continue;
    }
    
    double total = std::pow(6, n_reroll);
    std::map<int, int> counts;  // state_id → count
    
    // Recursive enumeration of all reroll outcomes
    auto enumerate = [&](auto&& self, std::vector<int>& roll) -> void {
        if (roll.size() == n_reroll) {
            std::vector<int> combined = held_vec;
            combined.insert(combined.end(), roll.begin(), roll.end());
            std::sort(combined.begin(), combined.end());
            std::array<int, 5> arr;
            std::copy(combined.begin(), combined.end(), arr.begin());
            counts[linear_to_id[linearize(arr)]]++;
            return;
        }
        for (int f = 1; f <= 6; ++f) {
            roll.push_back(f);
            self(self, roll);
            roll.pop_back();
        }
    };
    
    std::vector<int> buf;
    enumerate(enumerate, buf);
    
    for (auto const& [state_id, count] : counts) {
        all_transitions.push_back({
            static_cast<int16_t>(state_id),
            static_cast<double>(count) / total
        });
    }
    transition_count[held_id] = counts.size();
}
```

### 3.4 Transition Table Verification

After initialization, verify:
- For each held configuration, the probabilities sum to 1.0 (within floating-point tolerance).
- All `target_state_id` values are in range [0, 251].
- Holding all dice (mask 0b11111) produces exactly one transition with probability 1.0 to the same state.

---

## 4. Score Tables (SolverTables from Task 02)

The `SolverTables` struct defined in Task 02 is populated in this task as part of the same initialization routine. This includes:

### 4.1 dice_score[252][13]

For each dice state and each row, the raw score. See Task 02 Section 2.3 for the complete scoring rules per row.

Implementation: iterate all 252 states, compute dice counts and sum, then apply each row's scoring formula.

### 4.2 possible_scores[13]

The set of all achievable non-zero scores for each row. Hardcoded from the rules (see Task 02 Section 2.1 for the complete table).

### 4.3 filtered_scores[13][101]

For each row and each threshold 0–100, the subset of possible_scores that are >= threshold. Built by iterating possible_scores for each row and filtering.

---

## 5. Combined Initialization

All precomputed tables are initialized in a single function called once at program startup:

```cpp
/// Master data structure holding all precomputed solver/dice tables.
/// Built once at startup, read-only thereafter.
struct PrecomputedTables {
    // === Dice state enumeration ===
    std::array<int, 5> id_to_state[kNumDiceStates];
    int16_t linear_to_id[kMaxLinearIndex];
    
    // === Hold mask → held configuration mapping ===
    int16_t moves[kNumDiceStates][kNumHoldMasks];
    int num_held_configs;  // Total unique held configurations (computed at init)
    
    // === Transition probabilities (flat contiguous storage) ===
    std::vector<Transition> all_transitions;
    std::vector<int32_t> transition_offset;
    std::vector<int16_t> transition_count;
    
    // === Score tables ===
    SolverTables score_tables;
};

/// Initialize all precomputed tables. Call once at program startup.
/// This function is not performance-critical — it runs once and can take
/// a few hundred milliseconds.
void init_precomputed_tables(PrecomputedTables& tables);
```

### 5.1 Initialization Order

1. Enumerate dice states → populate `id_to_state` and `linear_to_id`
2. Enumerate held configurations → populate `moves` table, determine `num_held_configs`
3. Compute transition probabilities → populate `all_transitions`, `transition_offset`, `transition_count`
4. Compute score tables → populate `score_tables` (dice_score, possible_scores, filtered_scores)

### 5.2 Global Access

The `PrecomputedTables` instance should be globally accessible (read-only after init). Options:

- **Global variable** with `init_precomputed_tables()` called in `main()` before anything else.
- **Singleton** with lazy initialization (but we want explicit init for predictable startup timing).

Recommend: a global `const PrecomputedTables& get_tables()` function that asserts initialization has occurred.

```cpp
// In main.cpp:
PrecomputedTables g_tables;
init_precomputed_tables(g_tables);

// Everywhere else:
const auto& tables = get_tables();
int state_id = tables.linear_to_id[linearize(sorted_dice)];
```

---

## 6. Solver Accessor Functions

Provide clean accessor functions that the solver (Task 05) and later components will use, hiding the raw table indexing:

```cpp
/// Get the held configuration ID for a given dice state and hold mask.
int get_held_config(int dice_state_id, int hold_mask,
                    const PrecomputedTables& tables);

/// Get the transitions for a held configuration.
/// Returns a pointer to the first Transition and the count.
const Transition* get_transitions(int held_config_id, int& count,
                                   const PrecomputedTables& tables);

/// Get the dice state ID for an arbitrary (unsorted) set of 5 dice.
int get_dice_state_id(const int8_t dice[5], const PrecomputedTables& tables);

/// Get the raw score for a dice state and row.
/// Does NOT include Golden Rule or SS/LS interlock checks.
int get_dice_score(int dice_state_id, int row,
                   const PrecomputedTables& tables);

/// Get the filtered possible scores for a row given a minimum threshold.
/// Returns a pointer to the scores array and the count.
const int8_t* get_filtered_scores(int row, int min_threshold, int& count,
                                   const PrecomputedTables& tables);
```

These are thin wrappers — they do array indexing and return pointers. Zero overhead, but they provide a clean interface and catch invalid indices in debug builds via assertions.

---

## 7. File Organization

```
src/solver/
├── precomputed_tables.h      # PrecomputedTables struct, Transition struct,
│                              # init function declaration, accessor functions
├── precomputed_tables.cc     # Table initialization implementation
└── CMakeLists.txt            # Updated to build these files
```

```cmake
# src/solver/CMakeLists.txt
add_library(solver STATIC
    precomputed_tables.cc
)
target_include_directories(solver PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(solver PUBLIC engine)
```

---

## 8. Unit Tests

### 8.1 Dice State Tests (`tests/solver/dice_state_test.cc`)

**Enumeration:**
- Verify exactly 252 states are generated.
- Verify first state is {1,1,1,1,1} and last is {6,6,6,6,6}.
- Verify all states are sorted ascending.
- Verify no duplicate states.

**Linearization round-trip:**
- For each state, `linear_to_id[linearize(id_to_state[i])] == i`.
- Verify `linear_to_id` contains exactly 252 valid entries (rest are -1).

**Unsorted dice lookup:**
- `get_dice_state_id({3,1,4,1,5})` should return the same ID as `get_dice_state_id({1,1,3,4,5})`.
- Verify for several known dice combinations.

### 8.2 Hold Mask Tests (`tests/solver/hold_mask_test.cc`)

**MOVES table:**
- Dice state {1,2,3,4,5}, mask 0b00011 (hold dice 0,1 = values 1,2) → verify the held config ID.
- Same held dice from a different state should produce the same held config ID.
- Mask 0b11111 (hold all) for any state should map to a unique held config per state (since the held set equals the full state).
- Mask 0b00000 (hold none) for all states should map to the **same** held config (empty set).

**Held config count:**
- Verify `num_held_configs` is 792 (the theoretical count of sorted multisubsets of size 0–5 from {1–6}).

### 8.3 Transition Tests (`tests/solver/transition_test.cc`)

**Probability sums:**
- For every held configuration, verify transition probabilities sum to 1.0 (within tolerance 1e-10).

**Hold all dice:**
- For each dice state, mask 0b11111: verify exactly one transition with probability 1.0 back to the same state.

**Hold none:**
- Mask 0b00000 (reroll all 5 dice): verify the transition distribution matches the known distribution of 5d6 sorted outcomes. Total outcomes = 6^5 = 7776. The most common state ({1,2,3,4,5} etc. — all distinct) should have probability 120/7776 = 5/324 ≈ 0.01543. The rarest ({x,x,x,x,x}) should have probability 1/7776 ≈ 0.000129.

**Known transition:**
- Holding {6,6,6,6} (4 sixes, rerolling 1 die): should produce exactly 6 transitions:
  - {1,6,6,6,6} with probability 1/6
  - {2,6,6,6,6} with probability 1/6
  - {3,6,6,6,6} with probability 1/6
  - {4,6,6,6,6} with probability 1/6
  - {5,6,6,6,6} with probability 1/6
  - {6,6,6,6,6} with probability 1/6

**Target state validity:**
- All `target_state_id` values across all transitions are in [0, 251].

### 8.4 Score Table Tests (`tests/solver/score_table_test.cc`)

**dice_score spot checks:**
- State {3,3,3,2,2}: row 3s → 9, row FH → 33, row K → 0, row Y → 0.
- State {1,2,3,4,5}: row STR → 45, row FH → 0, row SS → 0 (sum=15 < 20).
- State {6,6,6,6,6}: row Y → 100, row K → 54, row FH → 50, row 6s → 30, row U8 → 0.
- State {1,1,1,1,1}: row Y → 75, row U8 → 75, row 1s → 5.

**Filtered scores:**
- Row 6s, threshold 0 → {6, 12, 18, 24, 30} (5 entries).
- Row 6s, threshold 13 → {18, 24, 30} (3 entries).
- Row 6s, threshold 31 → empty (0 entries).
- Row Y, threshold 100 → {100} (1 entry).
- Row Y, threshold 0 → {75, 80, 85, 90, 95, 100} (6 entries).

**Completeness:**
- For every dice state and every row, verify that `dice_score[state][row]` is either 0 or appears in `possible_scores[row]`.

---

## 9. Benchmarks

Add to `tests/benchmarks/solver_bench.cc`:

- **BM_InitPrecomputedTables:** Benchmark the full initialization. This runs once per program — we want to know the startup cost. Expected: under 1 second.
- **BM_GetDiceStateId:** Benchmark looking up a state ID from unsorted dice (sort + linearize + lookup).
- **BM_TransitionIteration:** Benchmark iterating all transitions for a held configuration and computing a weighted sum (simulates the solver's inner loop).
- **BM_GetFilteredScores:** Benchmark the filtered score lookup for various rows and thresholds.

---

## 10. Definition of Done

This task is complete when:

1. `init_precomputed_tables` runs successfully and populates all tables.
2. Exactly 252 dice states are enumerated with correct bidirectional lookup.
3. The MOVES table correctly maps all 252 × 32 = 8064 (state, mask) pairs to held configuration IDs.
4. All transition probabilities sum to 1.0 per held configuration.
5. Score tables (dice_score, possible_scores, filtered_scores) match expected values.
6. Accessor functions provide clean, assertion-checked access to all tables.
7. All unit tests pass.
8. Benchmarks run and establish baseline timing.
9. The solver library compiles with no libtorch dependency (depends only on engine).
10. Initialization completes in under 1 second on the target hardware.
