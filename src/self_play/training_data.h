#pragma once

#include "engine/game_traits.h"
#include "engine/tensor.h"
#include "self_play/game_instance.h"

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
// to td_mode, with TEAM-AWARE perspective flipping: targets are always
// P(player-who-placed's TEAM wins). For Yams1v1 every player is their own
// (singleton) team — collapses to the pre-2v2 behaviour bit-for-bit.
//
// In 2v2, bootstrap targets from future trajectory steps do NOT flip when
// the future step's player is on the same team as the placing player (the
// teammate's value already encodes my-team win probability).
//
// `exclude_player` (-1 = none) skips trajectory steps belonging to that player
// when emitting samples; used for past-opponent games so the current model is
// not trained on the older opponent's decisions.
//
// Templated on Traits so the team-aware logic comes from Traits::are_teammates.
// GameInstance itself is currently Yams1v1-only — the Yams2v2 instantiation
// becomes useful once GameInstance is templated in Task 7.
// ---------------------------------------------------------------------------
template <typename Traits = Yams1v1>
int extract_training_samples(const GameInstance& game,
                              TDMode td_mode, double td_lambda,
                              bool use_margin, double margin_scale,
                              bool use_pbrs,
                              TrainingSample* samples, int max_samples,
                              int exclude_player = -1);
