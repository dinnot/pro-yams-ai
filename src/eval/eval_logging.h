#pragma once

#include <string>
#include "eval/evaluator.h"

// ---------------------------------------------------------------------------
// log_evaluation — append evaluation results to the eval log CSV.
//
// Creates log_dir/eval_log.csv with a header row on first call;
// appends one data row on each subsequent call.
//
// Columns: timestamp,step,games,nn_wins,heur_wins,draws,win_rate,wr_as_p0,wr_as_p1,avg_margin
// ---------------------------------------------------------------------------
void log_evaluation(const std::string& log_dir, int training_step,
                     const EvalResult& result);

// ---------------------------------------------------------------------------
// log_lr_backoff — record a learning-rate back-off event.
//
// Appends one row to log_dir/lr_backoff_log.csv (header written on first call)
// and echoes a one-line summary to stdout so it's visible in the training log.
//
// Columns: timestamp,step,old_lr,new_lr,best_win_rate
// ---------------------------------------------------------------------------
void log_lr_backoff(const std::string& log_dir, int training_step,
                    double old_lr, double new_lr, double best_win_rate);
