#pragma once

#include <string>

#include "model/model_config.h"
#include "self_play/coordinator.h"    // SelfPlayConfig
#include "self_play/training_data.h"  // TDMode

// ---------------------------------------------------------------------------
// TrainingConfig — all hyperparameters for the training loop.
// ---------------------------------------------------------------------------
struct TrainingConfig {
    // --- Self-play orchestration ---
    SelfPlayConfig self_play;  // num_workers, num_games, batch params

    // --- Training target ---
    TDMode td_mode   = TDMode::kMC;   // MC targets never go stale in replay buffer
    double td_lambda = 0.0;           // Used only for kTDLambda.
    //
    // 2v2 note: keep td_lambda unchanged from the 1v1 setup (e.g. 0.95). The
    // per-sample TD(λ) variance coefficient is (1-λ)/(1+λ) — a function of λ
    // alone, independent of trajectory length. The 2v2 tuned config lives at
    // config/mlp-512/ (added in the migration's Task 9).

    // --- Duel margin maximization ---
    // When true, targets are tanh(margin/scale) in [-1, 1] instead of win/loss in [0, 1].
    bool   use_duel_margin_maximization = true;
    double duel_margin_maximization_scale = 4000.0;

    // --- Replay buffer ---
    int replay_capacity       = 2'000'000; // Maximum samples stored
    int min_buffer_size       = 10'000;    // Don't train until buffer reaches this size
    int train_batch_size      = 512;       // Samples per gradient step
    double train_steps_per_collect = 0.5; // E.g., 0.5 means 1 step per 2 games

    // --- Model swap and checkpointing ---
    int model_swap_interval   = 100;  // Training steps between inference model updates
    int checkpoint_interval   = 5000; // Training steps between checkpoint saves
    int max_checkpoints       = 5;    // Oldest checkpoints pruned beyond this count

    // --- Exploration (softmax temperature for solver) ---
    double initial_temperature = 1.0;
    double min_temperature     = 0.1;
    double temperature_decay   = 0.9999;  // Applied every training step
    int temperature_decay_start_step = 0; // Steps to wait before decaying temp
    double temperature_decay_start_value = 0.0; // If > 0: jump to this value at decay_start_step, then decay from there

    // --- Heuristic Bootstrapping ---
    double initial_heuristic_weight = 1.0;
    int heuristic_decay_steps       = 50000;
    int heuristic_version           = 2;  // 1 = V1 (greedy), 2 = V2 (DP-driven duel margin)

    // --- Initial epsilon (reserved for future ε-greedy exploration) ---
    double initial_epsilon = 0.0;

    // --- Past-opponent rotation (catastrophic-forgetting mitigation) ---
    // Probability that a freshly-recycled game pits the current model against
    // a randomly-chosen older checkpoint (uniform over the newest 5 on disk).
    // 0.0 disables the feature.
    double past_opponent_probability = 0.0;

    // --- Evaluation ---
    int eval_interval = 1000;   // Training steps between evaluation runs
    int eval_games    = 200;    // Games per evaluation run

    // --- Potential-Based Reward Shaping (PBRS) ---
    bool   use_pbrs          = false;
    double pbrs_upper_reward = 0.1;
    double pbrs_clean_reward = 0.2;

    // --- Developer / Debug ---
    bool debug_mode    = false;
    bool logs_on_start = false; // Log metrics and run eval before first training step

    // --- Paths ---
    std::string checkpoint_dir = "checkpoints";
    std::string log_dir        = ".";              // Directory for CSV logs
    std::string log_path       = "training_log.csv";

    // --- Model architecture ---
    ModelConfig model;
};
