#pragma once

#include <string>

// Forward declaration to avoid pulling in heavy headers.
class TrainingLoop;

// ---------------------------------------------------------------------------
// find_latest_checkpoint_stem — scan `dir` for checkpoint_step_N.model files
// and return the stem path (dir + "/checkpoint_step_N") for the highest N.
//
// @return  stem path on success, empty string if no checkpoints found.
// ---------------------------------------------------------------------------
std::string find_latest_checkpoint_stem(const std::string& dir);

// ---------------------------------------------------------------------------
// resume_from_checkpoint — restore TrainingLoop state from the latest
// checkpoint found in `dir`.
//
// Scans `dir` for checkpoint_step_N.model files, picks the highest N, loads
// the model+optimizer from that checkpoint, and loads the companion .buffer
// file if it exists.
//
// @return true  if a checkpoint was found and loaded
//         false if `dir` contains no checkpoint files
//
// Throws std::runtime_error on I/O or format errors.
// ---------------------------------------------------------------------------
bool resume_from_checkpoint(TrainingLoop& loop, const std::string& dir);

// ---------------------------------------------------------------------------
// init_from_checkpoint — load only model weights from a checkpoint into the
// TrainingLoop's trainer.  Training starts from step 0 with a fresh optimizer.
//
// `path` can be either:
//   - a directory  → the latest checkpoint in that directory is used
//   - a file stem  → e.g. "checkpoints/checkpoint_step_5000"
//
// @return true  if weights were loaded
//         false if the path/directory has no loadable checkpoint
// ---------------------------------------------------------------------------
bool init_from_checkpoint(TrainingLoop& loop, const std::string& path);
