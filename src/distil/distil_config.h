#pragma once

#include <string>

#include "model/model_config.h"
#include "self_play/coordinator.h"  // SelfPlayConfig

// ---------------------------------------------------------------------------
// TeacherKind — which oracle drives distillation.
// ---------------------------------------------------------------------------
enum class TeacherKind {
    kHeuristic,  // HeuristicTeacher<Traits> backed by V1..V17
    kNN          // NNTeacher<Traits> backed by an InferenceEngine
};

// ---------------------------------------------------------------------------
// DistilConfig — all hyperparameters for the distillation training loop.
//
// Parallels TrainingConfig in shape but drops everything that doesn't apply
// to supervised distillation: no TD/MC mode, no temperature schedule, no
// past-opponent rotation, no heuristic blending, no PBRS. Adds teacher
// selection, the replay-buffer parameters, and convergence-driven stopping.
// ---------------------------------------------------------------------------
struct DistilConfig {
    // --- Self-play orchestration ---
    SelfPlayConfig self_play;  // num_workers, num_games, batch params

    // --- Teacher selection ---
    TeacherKind  teacher_kind              = TeacherKind::kHeuristic;
    int          teacher_heuristic_version = 2;        // V1..V17 when kHeuristic
    std::string  teacher_checkpoint_path;              // Required when kNN

    // --- Student model architecture (may differ from teacher's) ---
    ModelConfig  student_model;

    // --- Replay buffer ---
    int    train_batch_size       = 1024;
    // Fixed-capacity ring buffer. Producers block when full; eviction is
    // driven by draw_batch (FIFO, ⌊batch_size / samples_per_train⌋ per
    // draw with a fractional remainder accumulator).
    int    replay_buffer_capacity = 2'000'000;
    // Minimum fill required before the very first draw — gives the trainer
    // a decorrelated warm-up batch. Subsequent draws only require
    // `size >= train_batch_size`.
    int    min_samples_to_start   = 16384;
    // Target expected number of training uses per sample before eviction
    // (must be ≥ 1.0). 1.0 = parity with the old one-pass shuffle queue
    // (each sample consumed once on average). Larger values amortise
    // expensive teacher / worker compute across more gradient updates at
    // the cost of staleness. K=2–4 is a typical sweet spot.
    double samples_per_train      = 1.0;

    // Per-sample keep probability when emitting from the worker into the
    // shuffle queue. Must be in (0, 1]. 1.0 (default) keeps every sample.
    // 0.3 keeps each sample independently with probability 0.3, trading
    // sample throughput per turn for game variance — across a fixed wall
    // budget the trainer sees samples from ~3× more games (and therefore
    // more board states) when set to 0.3 vs 1.0. The retained fraction is
    // approximate (binomial), not exact.
    double samples_per_games_rate = 1.0;

    // --- Checkpointing ---
    int   checkpoint_interval = 5000;
    int   max_checkpoints     = 5;

    // --- Convergence / stopping ---
    int    max_steps                     = 500'000;  // Hard cap.
    int    min_steps                     = 5'000;    // Don't check before this.
    int    eval_interval                 = 1000;
    int    eval_games                    = 200;
    int    reference_heuristic_version   = 2;
    double convergence_win_rate_delta    = 0.02;
    int    convergence_patience          = 3;
    double convergence_match_mse         = -1.0;   // Disabled if < 0.
    int    final_eval_games              = 1000;

    // --- Target normalisation passed to the teacher ---
    // Must match the student's output_activation:
    //   use_duel_margin_maximization=true  ↔ student output_activation="tanh"
    //   use_duel_margin_maximization=false ↔ student output_activation="sigmoid"
    bool   use_duel_margin_maximization   = true;
    double duel_margin_maximization_scale = 4000.0;

    // --- Paths ---
    std::string checkpoint_dir = "checkpoints_distil";
    std::string log_dir        = ".";
    std::string log_path       = "distil_log.csv";

    // --- Developer / Debug ---
    bool debug_mode = false;
};
