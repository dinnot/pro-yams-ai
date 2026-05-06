#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "engine/constants.h"
#include "engine/probability_tables.h"
#include "engine/solver_tables.h"
#include "solver/dp_tables.h"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int kNumDiceStates  = 252;
constexpr int kMaxLinearIndex = 7776;  // 6^5
constexpr int kNumHoldMasks   = 32;   // 2^5

// ---------------------------------------------------------------------------
// Transition — one outcome of rerolling unheld dice for a given held config.
// ---------------------------------------------------------------------------
struct Transition {
    int16_t target_state_id;  // 0-251: resulting dice state
    double  probability;      // Fraction of outcomes landing on this state
};

// ---------------------------------------------------------------------------
// PrecomputedTables — all lookup tables built once at startup.
//
// This struct extends SolverTables (Task 02) with the hold/reroll transition
// tables needed by the expectimax solver (Task 05).
//
// Initialised by init_precomputed_tables(); read-only thereafter.
// ---------------------------------------------------------------------------
struct PrecomputedTables {
    // === Dice state enumeration ===
    // id_to_state[id] — sorted 1-based dice values for state id (0-251)
    std::array<int, 5> id_to_state[kNumDiceStates];

    // linear_to_id[lin] — state id for the sorted 1-based linearisation
    //   lin = (d[0]-1)*1296 + (d[1]-1)*216 + (d[2]-1)*36 + (d[3]-1)*6 + (d[4]-1)
    // Value is -1 for unsorted or otherwise invalid indices.
    int16_t linear_to_id[kMaxLinearIndex];

    // === Hold mask → held configuration mapping ===
    // moves[state_id][hold_mask] — id of the unique held configuration
    // Hold mask bit i set means keep dice[i] (dice are indexed in sorted order).
    int16_t moves[kNumDiceStates][kNumHoldMasks];

    // Total number of unique held configurations (computed at init time, = 462).
    int num_held_configs;

    // === Transition probabilities (flat contiguous storage) ===
    // Grouped by held_config_id for cache-friendly iteration.
    // Access: all_transitions[ transition_offset[hid] .. +transition_count[hid] )
    std::vector<Transition> all_transitions;
    std::vector<int32_t>    transition_offset;  // [held_config_id] → start index
    std::vector<int16_t>    transition_count;   // [held_config_id] → count

    // === Score tables (from Task 02 SolverTables) ===
    SolverTables score_tables;

    // === Probability tables (from Task 06 ProbabilityTables) ===
    // Probability of achieving score >= threshold in a given row with 2 or 3 rolls.
    ProbabilityTables prob_tables;

    // === Dynamic Programming tables (V2.1 tensor) ===
    // Six DP tables: upper P/EV, middle P/EV, lower P/EV.  Auto-loaded from
    // disk cache by init_precomputed_tables when the cache file is present;
    // otherwise left empty (DP-dependent tensor features will be 0).
    DPTables dp_tables;
};

// ---------------------------------------------------------------------------
// Initialise all precomputed tables. Call exactly once before any solver use.
// Not performance-critical (runs once at startup).
// ---------------------------------------------------------------------------
void init_precomputed_tables(PrecomputedTables& tables);

// ---------------------------------------------------------------------------
// Accessor functions — thin wrappers with debug assertions.
// ---------------------------------------------------------------------------

/// Get the held configuration ID for (dice_state_id, hold_mask).
int get_held_config(int dice_state_id, int hold_mask,
                    const PrecomputedTables& tables);

/// Get the transitions for a held configuration.
/// Sets count and returns a pointer to the first Transition.
const Transition* get_transitions(int held_config_id, int& count,
                                  const PrecomputedTables& tables);

/// Get the dice state ID for an arbitrary (unsorted) 5-dice array.
/// Sorts internally; does not modify the input.
int get_dice_state_id(const int8_t dice[kNumDice],
                      const PrecomputedTables& tables);

/// Get the raw dice score for a (state_id, row) pair.
/// Does not apply Golden Rule or SS/LS constraints.
int get_dice_score(int dice_state_id, int row,
                   const PrecomputedTables& tables);

/// Get filtered possible scores for a row above a minimum threshold.
/// Sets count and returns a pointer to the scores array.
const int8_t* get_filtered_scores(int row, int min_threshold, int& count,
                                  const PrecomputedTables& tables);
