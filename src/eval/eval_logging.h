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
