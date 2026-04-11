#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "self_play/coordinator.h"
#include "self_play/game_instance.h"
#include "self_play/game_queues.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"
#include "model/inference.h"

// ---------------------------------------------------------------------------
// SelfPlayOrchestrator — top-level self-play manager.
//
// Owns the game pool, queues, worker threads, and coordinator thread.
// Games flow:  available → (worker: kNeedRequests) → pending
//              → (coordinator: GPU inference) → available
//              → (worker: kNeedResolve) → available or completed
// ---------------------------------------------------------------------------
class SelfPlayOrchestrator {
public:
    SelfPlayOrchestrator(const SelfPlayConfig& config,
                          const PrecomputedTables& tables,
                          InferenceEngine& inference,
                          const SolverConfig& solver_config);
    ~SelfPlayOrchestrator();

    /// Create games, launch worker and coordinator threads.
    void start();

    /// Signal shutdown, join all threads.
    void stop();

    /// Collect completed games (non-blocking). Returns number collected.
    int collect_completed(GameInstance** out, int max_count);

    /// Reset a completed game and return it to the available queue.
    void recycle_game(GameInstance* game, uint64_t new_seed);

    /// Update the solver config used by worker threads (e.g. temperature decay).
    void update_solver_config(const SolverConfig& config) { solver_config_ = config; }

    int available_queue_size() const { return available_queue_.size(); }
    int pending_queue_size() const { return pending_queue_.size(); }
    int completed_queue_size() const { return completed_queue_.size(); }

private:
    SelfPlayConfig           config_;
    const PrecomputedTables& tables_;
    InferenceEngine&         inference_;
    SolverConfig             solver_config_;

    std::vector<std::unique_ptr<GameInstance>> games_;

    GameQueue available_queue_;
    GameQueue pending_queue_;
    GameQueue completed_queue_;

    std::vector<std::thread> workers_;
    std::thread              coordinator_;
    std::atomic<bool>        shutdown_{false};
};
