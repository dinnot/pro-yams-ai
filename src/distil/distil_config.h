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
// to one-pass supervised distillation: no TD/MC mode, no replay buffer, no
// temperature schedule, no past-opponent rotation, no heuristic blending,
// no PBRS. Adds teacher selection, the shuffle-queue parameters, and
// convergence-driven stopping criteria.
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

    // --- Shuffle queue (replaces replay buffer) ---
    int   shuffle_chunk_size      = 65536;
    int   min_chunk_size_to_start = 16384;
    int   train_batch_size        = 1024;
    // Soft cap on (accumulating + serving_remaining). Workers block on
    // add_batch when the queue fills, so the teacher (and especially an NN
    // teacher sharing the GPU with the trainer) stops burning compute on
    // samples that would never be trained on. The cap is intentionally
    // generous — a 2M-sample window keeps enough game variance for healthy
    // shuffling while preventing runaway memory and resource contention.
    int   max_buffered_samples    = 2'000'000;

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
