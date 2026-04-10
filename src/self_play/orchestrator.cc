#include "self_play/orchestrator.h"

#include "self_play/coordinator.h"
#include "self_play/worker.h"
#include "engine/game_flow.h"
#include "engine/tensor.h"

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

SelfPlayOrchestrator::SelfPlayOrchestrator(const SelfPlayConfig& config,
                                             const PrecomputedTables& tables,
                                             InferenceEngine& inference,
                                             const SolverConfig& solver_config)
    : config_(config),
      tables_(tables),
      inference_(inference),
      solver_config_(solver_config) {}

SelfPlayOrchestrator::~SelfPlayOrchestrator() {
    if (!shutdown_.load()) stop();
}

// ---------------------------------------------------------------------------
// start() — pre-allocate games and launch threads.
// ---------------------------------------------------------------------------

void SelfPlayOrchestrator::start() {
    const uint64_t base_seed = 0x1234567890ABCDEFULL;

    games_.reserve(config_.num_games);
    for (int i = 0; i < config_.num_games; ++i) {
        auto game = std::make_unique<GameInstance>();
        game->game_id          = i;
        game->rng              = RNG(base_seed + static_cast<uint64_t>(i) * 6364136223846793005ULL);
        game->trajectory_length = 0;
        game->result           = 0.0;
        init_game(game->state, game->ctx, game->rng);
        game->phase = GamePhase::kNeedRequests;
        available_queue_.push(game.get());
        games_.push_back(std::move(game));
    }

    // Launch worker threads.
    workers_.reserve(config_.num_workers);
    for (int i = 0; i < config_.num_workers; ++i) {
        workers_.emplace_back(worker_thread,
            std::ref(available_queue_), std::ref(pending_queue_),
            std::ref(completed_queue_),
            std::ref(tables_), std::ref(solver_config_),
            std::ref(shutdown_));
    }

    // Launch coordinator thread.
    coordinator_ = std::thread(coordinator_thread,
        std::ref(pending_queue_), std::ref(available_queue_),
        std::ref(inference_), std::ref(config_),
        std::ref(shutdown_));
}

// ---------------------------------------------------------------------------
// stop() — signal shutdown and join all threads.
// ---------------------------------------------------------------------------

void SelfPlayOrchestrator::stop() {
    shutdown_ = true;

    // Push nullptr sentinels to wake up all blocked workers.
    for (int i = 0; i < static_cast<int>(workers_.size()); ++i)
        available_queue_.push(nullptr);

    // Join coordinator (it checks shutdown_ and uses collect() with timeout).
    if (coordinator_.joinable()) coordinator_.join();

    // Join workers.
    for (auto& w : workers_)
        if (w.joinable()) w.join();
}

// ---------------------------------------------------------------------------
// collect_completed — drain the completed queue.
// ---------------------------------------------------------------------------

int SelfPlayOrchestrator::collect_completed(GameInstance** out, int max_count) {
    int count = 0;
    while (count < max_count) {
        GameInstance* g = completed_queue_.try_pop();
        if (g == nullptr) break;
        out[count++] = g;
    }
    return count;
}

// ---------------------------------------------------------------------------
// recycle_game — reset a game and return it to the pool.
// ---------------------------------------------------------------------------

void SelfPlayOrchestrator::recycle_game(GameInstance* game, uint64_t new_seed) {
    game->rng              = RNG(new_seed);
    game->trajectory_length = 0;
    game->result           = 0.0;
    init_game(game->state, game->ctx, game->rng);
    game->phase = GamePhase::kNeedRequests;
    available_queue_.push(game);
}
