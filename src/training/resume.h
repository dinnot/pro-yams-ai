#pragma once

#include <string>

#include "engine/game_traits.h"

// Forward declaration to avoid pulling in heavy headers.
template <typename Traits> class TrainingLoopT;

// ---------------------------------------------------------------------------
// find_latest_checkpoint_stem — scan `dir` for checkpoint_step_N.model files
// and return the stem path (dir + "/checkpoint_step_N") for the highest N.
//
// @return  stem path on success, empty string if no checkpoints found.
// ---------------------------------------------------------------------------
std::string find_latest_checkpoint_stem(const std::string& dir);

// ---------------------------------------------------------------------------
// resume_from_checkpoint — restore TrainingLoopT state from the latest
// checkpoint found in `dir`.
//
// Templated on Traits so 1v1 and 2v2 builds both call the same name.
// ---------------------------------------------------------------------------
template <typename Traits>
bool resume_from_checkpoint(TrainingLoopT<Traits>& loop, const std::string& dir);

// ---------------------------------------------------------------------------
// init_from_checkpoint — load only model weights from a checkpoint into the
// TrainingLoopT's trainer.  Training starts from step 0 with a fresh optimizer.
// ---------------------------------------------------------------------------
template <typename Traits>
bool init_from_checkpoint(TrainingLoopT<Traits>& loop, const std::string& path);
