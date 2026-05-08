#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include "engine/rng.h"
#include "model/inference.h"
#include "model/trainer.h"
#include "self_play/game_instance.h"
#include "self_play/orchestrator.h"
#include "solver/precomputed_tables.h"
#include "training/metrics.h"
#include "training/replay_buffer.h"
#include "training/training_config.h"

// ---------------------------------------------------------------------------
// TrainingLoop — orchestrates self-play data generation and gradient steps.
//
// Thread model: run() is called from the main thread.  Internally it owns a
// SelfPlayOrchestrator (N worker threads + 1 coordinator thread) and drives
// the training pipeline:
//
//   collect completed games → extract samples → fill replay buffer
//   → sample mini-batch → ModelTrainer::train_step()
//   → periodically swap inference model and save checkpoints
// ---------------------------------------------------------------------------
class TrainingLoop {
public:
    TrainingLoop(const TrainingConfig& config,
                 const PrecomputedTables& tables,
                 torch::Device training_device,
                 torch::Device inference_device);

    /// Run until `num_steps` training steps have been completed (or stop() is
    /// called).  Blocks until done.
    void run(int num_steps);

    /// Signal the loop to exit after the current iteration.  Thread-safe.
    void stop();

    /// Snapshot of current training metrics.
    TrainingMetrics metrics() const;

    // --- Accessors for resume_from_checkpoint ---
    ModelTrainer&  trainer()        { return *trainer_; }
    ReplayBuffer&  replay_buffer()  { return *buffer_;  }
    void set_training_step(int step)    { training_step_ = step; }
    void set_temperature(double t)      { temperature_   = t;    }
    void set_epsilon(double e)          { epsilon_       = e;    }

private:
    // Returns the number of games collected.
    int collect_completed_games();
    void do_training_step();
    void maybe_swap_model();
    void maybe_checkpoint();
    void maybe_evaluate();

    // Pick a random checkpoint from disk and load its weights into the
    // opponent inference engine. No-op if no checkpoints exist yet.
    void refresh_opponent_model();

    TrainingConfig            config_;
    const PrecomputedTables&  tables_;
    torch::Device             train_device_;
    torch::Device             infer_device_;

    std::unique_ptr<ReplayBuffer>           buffer_;
    std::unique_ptr<ModelTrainer>           trainer_;
    std::shared_ptr<InferenceEngine>        inference_;
    std::shared_ptr<InferenceEngine>        opponent_inference_;  // null when feature disabled
    std::unique_ptr<ModelTrainer>           opponent_loader_;     // owns the past-opponent model + lets us load weights
    bool                                    opponent_ready_ = false;  // true once a checkpoint has been loaded into opponent_inference_
    RNG                                     opponent_rng_{0xA5A5A5A5C3C3C3C3ULL};
    std::unique_ptr<SelfPlayOrchestrator>   orchestrator_;

    SolverConfig  solver_config_;
    double        temperature_     = 1.0;
    double        epsilon_         = 0.0;
    double        heuristic_weight_= 1.0;
    int           training_step_   = 0;
    int           games_played_    = 0;
    long          total_samples_trained_ = 0;
    double        last_loss_       = 0.0;
    
    std::chrono::steady_clock::time_point start_time_;

    RNG           sample_rng_;

    std::atomic<bool> stop_flag_{false};

    double pending_train_steps_ = 0.0;

    double last_eval_win_rate_ = 0.0;

    static constexpr int kMaxCollectBatch = 512;
    GameInstance* collect_buf_[kMaxCollectBatch];
};
