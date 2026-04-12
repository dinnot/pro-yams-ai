#pragma once

#include "engine/constants.h"

// ---------------------------------------------------------------------------
// ProbabilityTables — precomputed per-turn achievement probabilities.
//
// For each row and minimum score threshold, stores the probability of
// achieving score >= threshold when dedicating a full turn with optimal
// holding strategy.
//
// prob_3rolls[row][threshold]: normal column (1 mandatory roll + 2 optional)
// prob_2rolls[row][threshold]: Turbo column  (1 mandatory roll + 1 optional)
//
// threshold range: 0-100.  Threshold 0 always yields 1.0 (every score >= 0).
// If threshold > max achievable score for the row, yields 0.0.
//
// Populated by init_precomputed_tables() in solver/precomputed_tables.cc.
// ---------------------------------------------------------------------------
struct ProbabilityTables {
    double prob_3rolls[kNumRows][101];  // prob_3rolls[row][threshold]
    double prob_2rolls[kNumRows][101];  // prob_2rolls[row][threshold]

    // --- ADD PRECOMPUTED COMPOUND ARRAYS ---
    float prob_3rolls_compound[kNumRows][101][79];
    float prob_2rolls_compound[kNumRows][101][79];
};
