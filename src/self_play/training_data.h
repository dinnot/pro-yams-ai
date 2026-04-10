#pragma once

#include "self_play/game_instance.h"
#include "engine/tensor.h"

// ---------------------------------------------------------------------------
// TDMode — training target computation mode.
// ---------------------------------------------------------------------------
enum class TDMode {
    kTD0,       // Bootstrap from next step's V(s')
    kTDLambda,  // Exponential blending of n-step returns
    kMC         // Monte Carlo: use actual game outcome
};

// ---------------------------------------------------------------------------
// TrainingSample — one (state, target) pair for supervised training.
// ---------------------------------------------------------------------------
struct TrainingSample {
    float  state[kTensorSize];  // Afterstate tensor (from placing player's view)
    double target;               // Target win probability in [0, 1]
};

// ---------------------------------------------------------------------------
// extract_training_samples — extract supervised samples from a completed game.
//
// Each trajectory step becomes one sample. The target is computed according
// to td_mode, with proper perspective flipping (targets are always P(player
// who placed wins)).
//
// @param game        Completed game (trajectory must be fully populated)
// @param td_mode     Target computation mode
// @param td_lambda   Lambda for kTDLambda mode (ignored for kTD0 / kMC)
// @param samples     Pre-allocated output array
// @param max_samples Maximum number of samples to write
// @return            Number of samples written
// ---------------------------------------------------------------------------
int extract_training_samples(const GameInstance& game,
                              TDMode td_mode, double td_lambda,
                              TrainingSample* samples, int max_samples);
