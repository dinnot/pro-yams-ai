#include <gtest/gtest.h>
#include "self_play/coordinator.h"
#include "self_play/game_instance.h"
#include "self_play/game_queues.h"
#include "model/inference.h"
#include "model/pro_yams_net.h"
#include "engine/tensor.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Shared small model for coordinator tests (CPU only for speed)
// ---------------------------------------------------------------------------
static std::shared_ptr<InferenceEngine> make_inference_cpu() {
    ModelConfig cfg;
    cfg.hidden_layers = 1;
    cfg.hidden_width  = 32;
    auto model = std::make_shared<ProYamsNet>(cfg);
    initialize_weights(*model);
    model->eval();
    return std::make_shared<InferenceEngine>(model, torch::Device(torch::kCPU));
}

// Helpers to create fake game instances with known request counts.
static std::unique_ptr<GameInstance> make_fake_game(int req_count, int id = 0) {
    auto game = std::make_unique<GameInstance>();
    game->game_id = id;
    game->solver_buffers.request_count = req_count;
    // Fill tensor_buffer with identifiable values.
    for (int i = 0; i < req_count * kTensorSize; ++i)
        game->tensor_buffer[i] = static_cast<float>(id + 1);
    return game;
}

// ---------------------------------------------------------------------------
// Batch assembly — games move to available with kNeedResolve + EVs set
// ---------------------------------------------------------------------------
TEST(CoordinatorTest, BatchAssembly_DistributesEVs) {
    auto engine = make_inference_cpu();
    SelfPlayConfig cfg;
    cfg.max_inference_batch = 1024;
    cfg.min_games_per_batch = 2;
    cfg.batch_timeout_ms    = 50;

    GameQueue pending, available;
    std::atomic<bool> shutdown{false};

    auto g0 = make_fake_game(10, 0);
    auto g1 = make_fake_game(20, 1);
    auto g2 = make_fake_game(15, 2);

    pending.push(g0.get());
    pending.push(g1.get());
    pending.push(g2.get());

    // Run coordinator for one iteration then shut down.
    std::thread ct(coordinator_thread,
                   std::ref(pending), std::ref(available),
                   std::ref(*engine), std::ref(cfg),
                   std::ref(shutdown));

    // Wait until all 3 games are in available queue.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (available.size() < 3 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    shutdown = true;
    ct.join();

    EXPECT_EQ(available.size(), 3);
    EXPECT_EQ(pending.size(), 0);

    while (auto* g = available.try_pop()) {
        EXPECT_EQ(g->phase, GamePhase::kNeedResolve);
        // EVs should be in [0, 1] (sigmoid output).
        for (int i = 0; i < g->solver_buffers.request_count; ++i) {
            double ev = g->solver_buffers.evs[i];
            EXPECT_GE(ev, 0.0);
            EXPECT_LE(ev, 1.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Batch overflow — excess games go back to pending
// ---------------------------------------------------------------------------
TEST(CoordinatorTest, BatchOverflow_ExcessGoesBackToPending) {
    auto engine = make_inference_cpu();
    SelfPlayConfig cfg;
    cfg.max_inference_batch = 50;  // very small
    cfg.min_games_per_batch = 1;
    cfg.batch_timeout_ms    = 30;

    GameQueue pending, available;
    std::atomic<bool> shutdown{false};

    // Each game has 30 tensors. max_batch=50 → first game fits, second doesn't.
    auto g0 = make_fake_game(30, 0);
    auto g1 = make_fake_game(30, 1);
    pending.push(g0.get());
    pending.push(g1.get());

    std::thread ct(coordinator_thread,
                   std::ref(pending), std::ref(available),
                   std::ref(*engine), std::ref(cfg),
                   std::ref(shutdown));

    // Wait for at least one game to flow through.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (available.size() < 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    shutdown = true;
    ct.join();

    // At least one game should have been processed; no games should be lost.
    int processed = available.size();
    int remaining = pending.size();
    EXPECT_GT(processed, 0);
    EXPECT_EQ(processed + remaining, 2)
        << "Games must not be lost (processed=" << processed
        << " remaining=" << remaining << ")";
}

// ---------------------------------------------------------------------------
// Timeout — coordinator processes single game even below min_games_per_batch
// ---------------------------------------------------------------------------
TEST(CoordinatorTest, Timeout_ProcessesBelowMinBatch) {
    auto engine = make_inference_cpu();
    SelfPlayConfig cfg;
    cfg.max_inference_batch = 1024;
    cfg.min_games_per_batch = 5;   // need 5, but only 1 will be pushed
    cfg.batch_timeout_ms    = 50;

    GameQueue pending, available;
    std::atomic<bool> shutdown{false};

    auto g = make_fake_game(8, 0);
    pending.push(g.get());

    std::thread ct(coordinator_thread,
                   std::ref(pending), std::ref(available),
                   std::ref(*engine), std::ref(cfg),
                   std::ref(shutdown));

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (available.size() < 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    shutdown = true;
    ct.join();

    EXPECT_GE(available.size(), 1) << "Game should have been processed after timeout";
}
