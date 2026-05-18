#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "distil/distil_worker.h"
#include "distil/replay_buffer.h"
#include "distil/teacher.h"
#include "engine/game_traits.h"
#include "self_play/coordinator.h"   // SelfPlayConfig
#include "self_play/game_instance.h"
#include "self_play/game_queues.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// DistilOrchestratorT<Traits> — owns the game pool, the worker threads, and
// the available queue for distillation.
//
// Simpler than SelfPlayOrchestratorT:
//   - No BatchManager and no coordinator threads (heuristic teacher path).
//   - No `completed` queue: workers recycle their own games in-place with a
//     fresh seed after each terminal, so the trainer never sees completed
//     games. NN-teacher variant will reuse the same queue plus BatchManager
//     plumbing (lands in a later step).
// ---------------------------------------------------------------------------
template <typename Traits>
class DistilOrchestratorT {
public:
    using Instance = GameInstanceT<Traits>;

    DistilOrchestratorT(const SelfPlayConfig& sp_config,
                        const PrecomputedTables& tables,
                        Teacher<Traits>& teacher,
                        DistilReplayBufferT<Traits>& replay_buffer,
                        const SolverConfig& solver_config,
                        double samples_per_games_rate = 1.0,
                        uint64_t base_seed = 0xDEADBEEF12345678ULL);
    ~DistilOrchestratorT();

    /// Initialise games and spawn worker threads.
    void start();

    /// Push nullptr sentinels and join all worker threads.
    void stop();

    long total_samples_emitted() const  { return stats_.samples_emitted.load(std::memory_order_relaxed); }
    long total_games_completed() const  { return stats_.games_completed.load(std::memory_order_relaxed); }
    long total_turns_processed() const  { return stats_.turns_processed.load(std::memory_order_relaxed); }
    int  available_queue_size()  const  { return available_queue_.size(); }

private:
    SelfPlayConfig           config_;
    const PrecomputedTables& tables_;
    Teacher<Traits>&         teacher_;
    DistilReplayBufferT<Traits>&   replay_buffer_;
    SolverConfig             solver_config_;
    double                   samples_per_games_rate_;
    uint64_t                 base_seed_;

    std::vector<std::unique_ptr<Instance>> games_;
    GameQueueT<Traits>                     available_queue_;
    std::vector<std::thread>               workers_;
    DistilWorkerStats                      stats_;
};

// Backward-compat aliases.
using DistilOrchestrator    = DistilOrchestratorT<Yams1v1>;
using DistilOrchestrator2v2 = DistilOrchestratorT<Yams2v2>;
