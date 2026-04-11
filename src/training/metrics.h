#pragma once

// ---------------------------------------------------------------------------
// TrainingMetrics — snapshot of training state for logging and monitoring.
// ---------------------------------------------------------------------------
struct TrainingMetrics {
    int    training_step          = 0;
    int    games_played           = 0;
    int    samples_in_buffer      = 0;
    double loss                   = 0.0;
    double temperature            = 1.0;
    double epsilon                = 0.0;
    double games_per_second       = 0.0;
    long   total_samples_trained  = 0;
    double latest_eval_win_rate   = 0.0;
};
