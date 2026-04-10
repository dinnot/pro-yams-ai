#pragma once

#include <cstdint>
#include "engine/constants.h"

// ---------------------------------------------------------------------------
// SolverTables — precomputed lookup tables built once at program startup.
//
// These tables eliminate redundant computation during solver operation.
// All tables are read-only after init_solver_tables() returns.
//
// The 252 sorted dice states are enumerated in ascending lexicographic order:
//   state 0  = {1,1,1,1,1}
//   state 1  = {1,1,1,1,2}
//   ...
//   state 251 = {6,6,6,6,6}
// ---------------------------------------------------------------------------
struct SolverTables {
    // === 252 sorted dice states ===
    // dice_states[id][0..4] — the sorted dice values for each state ID
    int8_t dice_states[252][kNumDice];

    // === State ID lookup ===
    // Given 5 sorted dice values (each 1-6), returns the state ID.
    // Indexed as: state_id[d0][d1][d2][d3][d4]  (d0 <= d1 <= d2 <= d3 <= d4)
    // Flat index: d0*6^4 + d1*6^3 + d2*6^2 + d3*6 + d4  (all 0-based)
    // We store it as a 6^5=7776 flat array for O(1) lookup.
    int16_t state_id[7776];   // indexed by packed sorted dice (0-based)

    // === Possible non-zero scores per row ===
    // possible_scores[row][i] — i-th achievable non-zero score for that row
    // possible_count[row]     — number of distinct non-zero scores
    // 0 (scratch) is always implicitly available and not stored here.
    int8_t possible_scores[kNumRows][32];  // 32 generous upper bound
    int8_t possible_count[kNumRows];

    // === Scores filtered by Golden Rule threshold ===
    // filtered_scores[row][threshold][i] — scores >= threshold for this row
    // filtered_count[row][threshold]     — count of such scores
    // threshold range 0-100 (max possible score is 100 for Yams)
    int8_t filtered_scores[kNumRows][101][32];
    int8_t filtered_count[kNumRows][101];

    // === Dice state → raw score per row ===
    // dice_score[state_id][row] — raw score produced by that dice state for
    // that row, BEFORE any Golden Rule or SS/LS checks.
    // 0 means the dice don't qualify (scratch).
    // Note: for number rows (0-5), 0 is a valid scratch score (no dice of that face).
    int8_t dice_score[252][kNumRows];
};

// ---------------------------------------------------------------------------
// Initialize all SolverTables fields.
// Call once at program startup. Thread-safe after return (tables are read-only).
// ---------------------------------------------------------------------------
void init_solver_tables(SolverTables& tables);

// ---------------------------------------------------------------------------
// Compute the raw dice score for a given (sorted) dice array and row.
// This is the same logic used to build dice_score[][] but callable at runtime.
// Does NOT apply Golden Rule or SS/LS constraints.
// ---------------------------------------------------------------------------
int compute_raw_score(const int8_t dice[kNumDice], int row);
