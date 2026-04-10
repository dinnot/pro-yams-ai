#pragma once

#include <string>
#include "training/metrics.h"

// ---------------------------------------------------------------------------
// log_metrics — append one CSV row to the training log.
//
// If the file does not exist or is empty, a header row is written first.
// Columns: step, games_played, buffer_size, loss, temperature, epsilon
// ---------------------------------------------------------------------------
void log_metrics(const std::string& path, const TrainingMetrics& metrics);

// ---------------------------------------------------------------------------
// prune_old_checkpoints — delete checkpoint files for old training steps.
//
// Scans `dir` for files matching checkpoint_step_N.{model,optimizer,buffer}
// and removes those for the oldest steps until at most `max_keep` remain.
// ---------------------------------------------------------------------------
void prune_old_checkpoints(const std::string& dir, int max_keep);
