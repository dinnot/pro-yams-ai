#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "engine/game_traits.h"
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
// TrainingLoopT<Traits> — orchestrates self-play data generation and gradient
// steps.
//
// Thread model: run() is called from the main thread.  Internally it owns a
// SelfPlayOrchestratorT (N worker threads + 1 coordinator thread) and drives
// the training pipeline:
//
//   collect completed games → extract samples → fill replay buffer
//   → sample mini-batch → ModelTrainer::train_step()
//   → periodically swap inference model and save checkpoints
// ---------------------------------------------------------------------------
template <typename Traits>
class TrainingLoopT {
public:
    using Instance = GameInstanceT<Traits>;
    using Sample   = TrainingSampleT<Traits>;
    using Buffer   = ReplayBufferT<Traits>;
    using Orchestr = SelfPlayOrchestratorT<Traits>;

    TrainingLoopT(const TrainingConfig& config,
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
    Buffer&        replay_buffer()  { return *buffer_;  }
    void set_training_step(int step)    { training_step_ = step; }
    void set_temperature(double t)      { temperature_   = t;    }
    void set_epsilon(double e)          { epsilon_       = e;    }

private:
    int  collect_completed_games();
    void do_training_step();
    void maybe_swap_model();
    void maybe_checkpoint();
    void maybe_evaluate();

    void refresh_opponent_model();

    TrainingConfig            config_;
    const PrecomputedTables&  tables_;
    torch::Device             train_device_;
    torch::Device             infer_device_;

    std::unique_ptr<Buffer>                 buffer_;
    std::unique_ptr<ModelTrainer>           trainer_;
    std::shared_ptr<InferenceEngine>        inference_;
    std::shared_ptr<InferenceEngine>        opponent_inference_;
    std::unique_ptr<ModelTrainer>           opponent_loader_;
    bool                                    opponent_ready_ = false;
    RNG                                     opponent_rng_{0xA5A5A5A5C3C3C3C3ULL};
    std::unique_ptr<Orchestr>               orchestrator_;

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
    Instance* collect_buf_[kMaxCollectBatch];

    std::vector<float>  train_states_;
    std::vector<double> train_targets_;
};

// Backward-compat aliases.
using TrainingLoop    = TrainingLoopT<Yams1v1>;
using TrainingLoop2v2 = TrainingLoopT<Yams2v2>;
