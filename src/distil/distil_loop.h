#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "distil/convergence.h"
#include "distil/distil_config.h"
#include "distil/distil_orchestrator.h"
#include "distil/shuffle_queue.h"
#include "distil/teacher.h"
#include "distil/teacher_nn.h"
#include "engine/game_traits.h"
#include "model/trainer.h"
#include "solver/precomputed_tables.h"
#include "training/metrics.h"

// ---------------------------------------------------------------------------
// DistilStopReason — why run() exited. Defined at namespace scope (not nested
// in the template) so helpers / printers / tests don't need to be templated
// on Traits just to name the enum.
// ---------------------------------------------------------------------------
enum class DistilStopReason {
    kNotStopped,    // run() hasn't returned yet, or was never called
    kMaxSteps,      // training_step_ reached config.max_steps
    kConvergence,   // convergence_satisfied() returned true
    kStopSignal,    // stop() was called externally (SIGINT, etc.)
    kQueueDrained,  // draw_batch returned 0 — queue stopped before max_steps
};

const char* stop_reason_str(DistilStopReason r);

// ---------------------------------------------------------------------------
// DistilLoopT<Traits> — top-level distillation loop.
//
// Wires together: student ModelTrainer, Teacher<Traits>, ShuffleQueueT<Traits>,
// and DistilOrchestratorT<Traits>. Owns the run loop:
//
//   queue.draw_batch → student.train_step → (checkpoint?) → (converged?) → loop
//
// Convergence and final-report logic land in later steps; for now the loop
// runs to config.max_steps (or until stop() is called from a signal handler).
// ---------------------------------------------------------------------------
template <typename Traits>
class DistilLoopT {
public:
    DistilLoopT(const DistilConfig& config,
                const PrecomputedTables& tables,
                torch::Device training_device,
                torch::Device inference_device);

    /// Run until max_steps, stop(), or a future convergence criterion fires.
    void run();

    /// Signal the loop to exit after the current iteration. Thread-safe.
    void stop();

    /// Snapshot of current training metrics.
    TrainingMetrics metrics() const;

    // --- Accessors (used by --checkpoint warm-start in main.cpp) ---
    ModelTrainer& trainer() { return *student_trainer_; }

    /// Used by resume_distil_from_checkpoint to restore the training-step
    /// counter so checkpoint / eval modulos line up with the saved step.
    void set_training_step(int step) { training_step_ = step; }

    // --- Diagnostics ---
    long total_turns_processed() const {
        return orchestrator_ ? orchestrator_->total_turns_processed() : 0;
    }
    long total_samples_emitted() const {
        return orchestrator_ ? orchestrator_->total_samples_emitted() : 0;
    }
    /// EMA of train_step's MSE loss. Used as the optional secondary
    /// convergence criterion; 0.0 until the first train_step runs.
    double rolling_match_mse() const { return rolling_mse_.value; }

    // StopReason is the namespace-level DistilStopReason. Keep this typedef
    // so callers can write DistilLoop::StopReason for backward-compat.
    using StopReason = DistilStopReason;

    StopReason stop_reason() const { return stop_reason_; }

    /// Headline numbers from final_report (recomputed at end of run() with
    /// config.final_eval_games — typically larger than eval_games for a
    /// tighter CI on the reported figure). Zero until final_report runs.
    double final_teacher_win_rate() const { return final_teacher_win_rate_; }
    double final_student_win_rate() const { return final_student_win_rate_; }

private:
    void do_training_step(int n);
    void maybe_checkpoint();
    void maybe_evaluate();
    /// Run teacher-vs-reference once at startup and cache the result.
    /// Heuristic teacher only in step 7; NN teacher path lands later.
    void cache_teacher_baseline();
    /// Re-evaluate both teacher and student with config.final_eval_games
    /// for a tight CI on the headline numbers. Print a summary and append
    /// a single row to <log_dir>/distil_final.csv. Always called once at
    /// the end of run(), regardless of stop reason.
    void final_report();

    bool convergence_satisfied() const;

    DistilConfig                                  config_;
    const PrecomputedTables&                      tables_;
    torch::Device                                 train_device_;
    torch::Device                                 infer_device_;

    std::unique_ptr<ModelTrainer>                 student_trainer_;
    std::unique_ptr<Teacher<Traits>>              teacher_;
    NNTeacher<Traits>*                            nn_teacher_view_ = nullptr;
    std::unique_ptr<ShuffleQueueT<Traits>>        queue_;
    std::unique_ptr<DistilOrchestratorT<Traits>>  orchestrator_;

    int  training_step_         = 0;
    long total_samples_trained_ = 0;
    double last_loss_           = 0.0;

    // Eval / convergence state
    double teacher_win_rate_         = 0.0;  // cached once at startup
    double last_student_win_rate_    = 0.0;
    int    consecutive_passes_       = 0;
    distil::EmaState rolling_mse_;           // EMA of train_step's MSE loss
    double final_teacher_win_rate_   = 0.0;  // populated by final_report()
    double final_student_win_rate_   = 0.0;
    DistilStopReason stop_reason_    = DistilStopReason::kNotStopped;

    std::chrono::steady_clock::time_point start_time_;
    std::atomic<bool>                     stop_flag_{false};

    std::vector<float>  train_states_;
    std::vector<double> train_targets_;
};

// Backward-compat aliases.
using DistilLoop    = DistilLoopT<Yams1v1>;
using DistilLoop2v2 = DistilLoopT<Yams2v2>;
