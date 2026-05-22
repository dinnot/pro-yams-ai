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
// prune_old_checkpoints — prune buffer/optimizer files for old training steps.
//
// Scans `dir` for files matching checkpoint_step_N.{model,optimizer,buffer}.
// For all but the most recent `max_keep` steps, the .optimizer and .buffer
// files are removed to reclaim disk space; the .model file is always kept so
// the full history of trained weights is preserved.
// ---------------------------------------------------------------------------
void prune_old_checkpoints(const std::string& dir, int max_keep);
