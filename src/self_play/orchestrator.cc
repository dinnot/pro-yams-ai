#include "self_play/orchestrator.h"

#include "self_play/coordinator.h"
#include "self_play/worker.h"
#include "engine/game_flow.h"
#include "engine/tensor.h"
#include <filesystem>
#include <fstream>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

template <typename Traits>
SelfPlayOrchestratorT<Traits>::SelfPlayOrchestratorT(
        const SelfPlayConfig& config,
        const PrecomputedTables& tables,
        InferenceEngine& inference,
        const SolverConfig& solver_config,
        InferenceEngine* opponent_inference)
    : config_(config),
      tables_(tables),
      inference_(inference),
      opponent_inference_(opponent_inference),
      solver_config_(solver_config) {}

template <typename Traits>
SelfPlayOrchestratorT<Traits>::~SelfPlayOrchestratorT() {
    if (!shutdown_.load()) stop();
}

// ---------------------------------------------------------------------------
// start()
// ---------------------------------------------------------------------------

template <typename Traits>
void SelfPlayOrchestratorT<Traits>::start() {
    const uint64_t base_seed = 0x1234567890ABCDEFULL;

    games_.reserve(config_.num_games);
    for (int i = 0; i < config_.num_games; ++i) {
        auto game = std::make_unique<Instance>();
        game->game_id          = i;
        game->rng              = RNG(base_seed + static_cast<uint64_t>(i) * 6364136223846793005ULL);
        game->trajectory_length = 0;
        game->result           = 0.0;
        init_game<Traits>(game->state, game->ctx, game->rng);
        game->heuristic_version = random_heuristic_version(game->rng);
        game->phase = GamePhase::kNeedRequests;

        if (i == 0 && solver_config_.debug_mode) {
            game->is_debug_game = true;
            game->debug_log_path = solver_config_.debug_log_path;

            auto parent = std::filesystem::path(game->debug_log_path).parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }

            std::ofstream f(game->debug_log_path, std::ios::trunc);
            f << "=== PRO YAMS AI DEBUG LOG ===\n";
        }

        available_queue_.push(game.get());
        games_.push_back(std::move(game));
    }

    int num_batches = config_.num_coordinators * 2;
    bool use_pinned = inference_.device().is_cuda();
    batch_manager_ = std::make_unique<BatchManagerT<Traits>>(
        num_batches, config_.max_inference_batch, use_pinned, config_.batch_timeout_ms);

    if (opponent_inference_ != nullptr) {
        bool opp_pinned = opponent_inference_->device().is_cuda();
        opponent_batch_manager_ = std::make_unique<BatchManagerT<Traits>>(
            num_batches, config_.max_inference_batch, opp_pinned, config_.batch_timeout_ms);
    }

    workers_.reserve(config_.num_workers);
    BatchManagerT<Traits>* opp_bm_ptr =
        opponent_batch_manager_ ? opponent_batch_manager_.get() : nullptr;
    for (int i = 0; i < config_.num_workers; ++i) {
        workers_.emplace_back(worker_thread<Traits>,
            std::ref(available_queue_), std::ref(*batch_manager_),
            opp_bm_ptr,
            std::ref(completed_queue_),
            std::ref(tables_), std::ref(solver_config_),
            std::ref(shutdown_));
    }

    coordinators_.reserve(config_.num_coordinators);
    for (int i = 0; i < config_.num_coordinators; ++i) {
        coordinators_.emplace_back(coordinator_thread<Traits>,
            std::ref(*batch_manager_), std::ref(available_queue_),
            std::ref(inference_), std::ref(config_),
            std::ref(shutdown_), i);
    }

    if (opponent_batch_manager_) {
        opponent_coordinators_.reserve(config_.num_coordinators);
        for (int i = 0; i < config_.num_coordinators; ++i) {
            opponent_coordinators_.emplace_back(coordinator_thread<Traits>,
                std::ref(*opponent_batch_manager_), std::ref(available_queue_),
                std::ref(*opponent_inference_), std::ref(config_),
                std::ref(shutdown_), 100 + i);
        }
    }
}

// ---------------------------------------------------------------------------
// stop()
// ---------------------------------------------------------------------------

template <typename Traits>
void SelfPlayOrchestratorT<Traits>::stop() {
    shutdown_ = true;

    for (int i = 0; i < static_cast<int>(workers_.size()); ++i)
        available_queue_.push(nullptr);

    if (batch_manager_) {
        batch_manager_->shutdown();
    }
    if (opponent_batch_manager_) {
        opponent_batch_manager_->shutdown();
    }

    for (auto& c : coordinators_)
        if (c.joinable()) c.join();
    for (auto& c : opponent_coordinators_)
        if (c.joinable()) c.join();

    for (auto& w : workers_)
        if (w.joinable()) w.join();
}

// ---------------------------------------------------------------------------
// collect_completed
// ---------------------------------------------------------------------------

template <typename Traits>
int SelfPlayOrchestratorT<Traits>::collect_completed(Instance** out, int max_count) {
    int count = 0;
    while (count < max_count) {
        Instance* g = completed_queue_.try_pop();
        if (g == nullptr) break;
        out[count++] = g;
    }
    return count;
}

// ---------------------------------------------------------------------------
// recycle_game
// ---------------------------------------------------------------------------

template <typename Traits>
void SelfPlayOrchestratorT<Traits>::recycle_game(Instance* game, uint64_t new_seed,
                                                  bool use_past_opponent,
                                                  int past_opponent_player) {
    game->rng              = RNG(new_seed);
    game->trajectory_length = 0;
    game->result           = 0.0;
    game->use_past_opponent    = use_past_opponent && (opponent_inference_ != nullptr);
    game->past_opponent_player = game->use_past_opponent ? past_opponent_player : -1;
    init_game<Traits>(game->state, game->ctx, game->rng);
    game->heuristic_version = random_heuristic_version(game->rng);
    game->phase = GamePhase::kNeedRequests;

    if (game->is_debug_game) {
        auto parent = std::filesystem::path(game->debug_log_path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::ofstream f(game->debug_log_path, std::ios::app);
        f << "\n=== NEW GAME STARTED ===\n";
    }

    available_queue_.push(game);
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template class SelfPlayOrchestratorT<Yams1v1>;
template class SelfPlayOrchestratorT<Yams2v2>;
