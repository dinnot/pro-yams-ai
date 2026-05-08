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
// `exclude_player` (-1 = none) skips trajectory steps belonging to that player
// when emitting samples; used for past-opponent games so the current model is
// not trained on the older opponent's decisions. Bootstrapping (TD0/TDLambda)
// still walks the full trajectory, so values from the excluded player's steps
// are reachable as bootstrap targets — note that those values come from the
// older model and are mildly biased; MC mode is unaffected.
//
// @param game           Completed game (trajectory must be fully populated)
// @param td_mode        Target computation mode
// @param td_lambda      Lambda for kTDLambda mode (ignored for kTD0 / kMC)
// @param samples        Pre-allocated output array
// @param max_samples    Maximum number of samples to write
// @param exclude_player Player whose steps are skipped (-1 = include all)
// @return               Number of samples written
// ---------------------------------------------------------------------------
int extract_training_samples(const GameInstance& game,
                              TDMode td_mode, double td_lambda,
                              bool use_margin, double margin_scale,
                              bool use_pbrs,
                              TrainingSample* samples, int max_samples,
                              int exclude_player = -1);
