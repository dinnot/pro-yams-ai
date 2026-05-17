#include "distil/distil_orchestrator.h"

#include "engine/game_flow.h"

template <typename Traits>
DistilOrchestratorT<Traits>::DistilOrchestratorT(
    const SelfPlayConfig& sp_config,
    const PrecomputedTables& tables,
    Teacher<Traits>& teacher,
    ShuffleQueueT<Traits>& shuffle_queue,
    const SolverConfig& solver_config,
    uint64_t base_seed)
    : config_(sp_config),
      tables_(tables),
      teacher_(teacher),
      shuffle_queue_(shuffle_queue),
      solver_config_(solver_config),
      base_seed_(base_seed) {}

template <typename Traits>
DistilOrchestratorT<Traits>::~DistilOrchestratorT() {
    stop();
}

template <typename Traits>
void DistilOrchestratorT<Traits>::start() {
    games_.clear();
    games_.reserve(static_cast<size_t>(config_.num_games));
    for (int i = 0; i < config_.num_games; ++i) {
        auto g = std::make_unique<Instance>();
        g->game_id = i;
        uint64_t seed = base_seed_ ^
            (static_cast<uint64_t>(i + 1) * 0x9E3779B97F4A7C15ULL);
        g->rng = RNG(seed);
        init_game<Traits>(g->state, g->ctx, g->rng);
        g->trajectory_length = 0;
        g->phase = GamePhase::kNeedRequests;
        available_queue_.push(g.get());
        games_.push_back(std::move(g));
    }

    workers_.clear();
    workers_.reserve(static_cast<size_t>(config_.num_workers));
    for (int w = 0; w < config_.num_workers; ++w) {
        uint64_t worker_seed = base_seed_ ^
            (static_cast<uint64_t>(w + 1) * 0xC0FFEE1357924680ULL);
        workers_.emplace_back([this, worker_seed]() {
            distil_worker_thread<Traits>(available_queue_, teacher_,
                                         shuffle_queue_, tables_,
                                         solver_config_, worker_seed,
                                         &stats_);
        });
    }
}

template <typename Traits>
void DistilOrchestratorT<Traits>::stop() {
    if (workers_.empty()) return;
    // One nullptr per worker — each pops at most one sentinel and exits.
    for (size_t i = 0; i < workers_.size(); ++i) {
        available_queue_.push(nullptr);
    }
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template class DistilOrchestratorT<Yams1v1>;
template class DistilOrchestratorT<Yams2v2>;
