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
// TrainingSampleT<Traits> — one (state, target) pair for supervised training.
// Sized by Traits::kTensorSize so 1v1 and 2v2 samples don't share a type.
// ---------------------------------------------------------------------------
template <typename Traits>
struct TrainingSampleT {
    float  state[Traits::kTensorSize];  // Afterstate tensor (from placing player's view)
    double target;                       // Target win probability / margin
};

using TrainingSample    = TrainingSampleT<Yams1v1>;
using TrainingSample2v2 = TrainingSampleT<Yams2v2>;

// ---------------------------------------------------------------------------
// extract_training_samples — extract supervised samples from a completed game.
//
// Each trajectory step becomes one sample. The target is computed according
// to td_mode, with TEAM-AWARE perspective flipping: targets are always
// P(player-who-placed's TEAM wins). For Yams1v1 every player is their own
// (singleton) team — collapses to the pre-2v2 behaviour bit-for-bit.
//
// `exclude_player` (-1 = none) skips trajectory steps for that player AND its
// teammates (via Traits::are_teammates), so in 2v2 the whole past-opponent
// team is dropped — preventing the current model from training on the older
// opponent's decisions. In 1v1 this collapses to the prior single-seat skip.
// ---------------------------------------------------------------------------
template <typename Traits = Yams1v1>
int extract_training_samples(const GameInstanceT<Traits>& game,
                              TDMode td_mode, double td_lambda,
                              bool use_margin, double margin_scale,
                              bool use_pbrs,
                              TrainingSampleT<Traits>* samples, int max_samples,
                              int exclude_player = -1);
