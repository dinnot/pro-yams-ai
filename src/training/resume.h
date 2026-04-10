#pragma once

#include <string>

// Forward declaration to avoid pulling in heavy headers.
class TrainingLoop;

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
