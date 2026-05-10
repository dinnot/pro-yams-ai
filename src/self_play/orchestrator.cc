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

SelfPlayOrchestrator::SelfPlayOrchestrator(const SelfPlayConfig& config,
                                             const PrecomputedTables& tables,
                                             InferenceEngine& inference,
                                             const SolverConfig& solver_config,
                                             InferenceEngine* opponent_inference)
    : config_(config),
      tables_(tables),
      inference_(inference),
      opponent_inference_(opponent_inference),
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
        game->heuristic_version = random_heuristic_version(game->rng);
        game->phase = GamePhase::kNeedRequests;

        if (i == 0 && solver_config_.debug_mode) {
            game->is_debug_game = true;
            game->debug_log_path = solver_config_.debug_log_path;

            // Ensure parent directory exists
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

    // Number of active batches must be enough to keep all coordinators and workers busy.
    // 2x coordinators is a good heuristic to allow one to be sent to GPU while workers fill another.
    int num_batches = config_.num_coordinators * 2;
    bool use_pinned = inference_.device().is_cuda();
    batch_manager_ = std::make_unique<BatchManager>(
        num_batches, config_.max_inference_batch, use_pinned, config_.batch_timeout_ms);

    if (opponent_inference_ != nullptr) {
        bool opp_pinned = opponent_inference_->device().is_cuda();
        opponent_batch_manager_ = std::make_unique<BatchManager>(
            num_batches, config_.max_inference_batch, opp_pinned, config_.batch_timeout_ms);
    }

    // Launch worker threads.
    workers_.reserve(config_.num_workers);
    BatchManager* opp_bm_ptr = opponent_batch_manager_ ? opponent_batch_manager_.get() : nullptr;
    for (int i = 0; i < config_.num_workers; ++i) {
        workers_.emplace_back(worker_thread,
            std::ref(available_queue_), std::ref(*batch_manager_),
            opp_bm_ptr,
            std::ref(completed_queue_),
            std::ref(tables_), std::ref(solver_config_),
            std::ref(shutdown_));
    }

    // Launch coordinator threads (current model).
    coordinators_.reserve(config_.num_coordinators);
    for (int i = 0; i < config_.num_coordinators; ++i) {
        coordinators_.emplace_back(coordinator_thread,
            std::ref(*batch_manager_), std::ref(available_queue_),
            std::ref(inference_), std::ref(config_),
            std::ref(shutdown_), i);
    }

    // Launch coordinator threads for the past opponent (if enabled).
    if (opponent_batch_manager_) {
        opponent_coordinators_.reserve(config_.num_coordinators);
        for (int i = 0; i < config_.num_coordinators; ++i) {
            opponent_coordinators_.emplace_back(coordinator_thread,
                std::ref(*opponent_batch_manager_), std::ref(available_queue_),
                std::ref(*opponent_inference_), std::ref(config_),
                std::ref(shutdown_), 100 + i);  // distinct id for logs
        }
    }
}

// ---------------------------------------------------------------------------
// stop() — signal shutdown and join all threads.
// ---------------------------------------------------------------------------

void SelfPlayOrchestrator::stop() {
    shutdown_ = true;

    // Push nullptr sentinels to wake up all blocked workers.
    for (int i = 0; i < static_cast<int>(workers_.size()); ++i)
        available_queue_.push(nullptr);

    if (batch_manager_) {
        batch_manager_->shutdown();
    }
    if (opponent_batch_manager_) {
        opponent_batch_manager_->shutdown();
    }

    // Join coordinators (they check shutdown_ and use pop_with_timeout).
    for (auto& c : coordinators_)
        if (c.joinable()) c.join();
    for (auto& c : opponent_coordinators_)
        if (c.joinable()) c.join();

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

void SelfPlayOrchestrator::recycle_game(GameInstance* game, uint64_t new_seed,
                                         bool use_past_opponent,
                                         int past_opponent_player) {
    game->rng              = RNG(new_seed);
    game->trajectory_length = 0;
    game->result           = 0.0;
    // Only honour past-opponent assignment when wired for it.
    game->use_past_opponent    = use_past_opponent && (opponent_inference_ != nullptr);
    game->past_opponent_player = game->use_past_opponent ? past_opponent_player : -1;
    init_game(game->state, game->ctx, game->rng);
    game->heuristic_version = random_heuristic_version(game->rng);
    game->phase = GamePhase::kNeedRequests;

    if (game->is_debug_game) {
        // Ensure parent directory exists
        auto parent = std::filesystem::path(game->debug_log_path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::ofstream f(game->debug_log_path, std::ios::app);
        f << "\n=== NEW GAME STARTED ===\n";
    }

    available_queue_.push(game);
}
