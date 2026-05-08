#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "self_play/coordinator.h"
#include "self_play/game_instance.h"
#include "self_play/game_queues.h"
#include "self_play/batch_manager.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"
#include "model/inference.h"

// ---------------------------------------------------------------------------
// SelfPlayOrchestrator — top-level self-play manager.
//
// Owns the game pool, queues, worker threads, and coordinator thread.
// Games flow:  available → (worker: kNeedRequests) → BatchManager
//              → (coordinator: GPU inference) → available
//              → (worker: kNeedResolve) → available or completed
// ---------------------------------------------------------------------------
class SelfPlayOrchestrator {
public:
    /// `opponent_inference` is optional. When non-null, a second BatchManager and
    /// a parallel set of coordinator threads route past-opponent requests to it;
    /// games marked with use_past_opponent=true will have one seat played by
    /// this older model.
    SelfPlayOrchestrator(const SelfPlayConfig& config,
                          const PrecomputedTables& tables,
                          InferenceEngine& inference,
                          const SolverConfig& solver_config,
                          InferenceEngine* opponent_inference = nullptr);
    ~SelfPlayOrchestrator();

    /// Create games, launch worker and coordinator threads.
    void start();

    /// Signal shutdown, join all threads.
    void stop();

    /// Collect completed games (non-blocking). Returns number collected.
    int collect_completed(GameInstance** out, int max_count);

    /// Reset a completed game and return it to the available queue.
    /// When use_past_opponent is true, past_opponent_player must be 0 or 1.
    void recycle_game(GameInstance* game, uint64_t new_seed,
                      bool use_past_opponent = false,
                      int past_opponent_player = -1);

    /// Update the solver config used by worker threads (e.g. temperature decay).
    void update_solver_config(const SolverConfig& config) { solver_config_ = config; }

    /// True when this orchestrator was constructed with an opponent inference
    /// engine and is therefore wired for past-opponent games.
    bool has_opponent_inference() const { return opponent_inference_ != nullptr; }

    int available_queue_size() const { return available_queue_.size(); }
    int completed_queue_size() const { return completed_queue_.size(); }

private:
    SelfPlayConfig           config_;
    const PrecomputedTables& tables_;
    InferenceEngine&         inference_;
    InferenceEngine*         opponent_inference_;  // optional
    SolverConfig             solver_config_;

    std::vector<std::unique_ptr<GameInstance>> games_;

    GameQueue available_queue_;
    std::unique_ptr<BatchManager> batch_manager_;
    std::unique_ptr<BatchManager> opponent_batch_manager_;  // optional
    GameQueue completed_queue_;

    std::vector<std::thread> workers_;
    std::vector<std::thread> coordinators_;
    std::vector<std::thread> opponent_coordinators_;
    std::atomic<bool>        shutdown_{false};
};
