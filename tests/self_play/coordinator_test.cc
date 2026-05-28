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
#include <filesystem>
#include <fstream>

// ---------------------------------------------------------------------------
// Shared small model for coordinator tests (CPU only for speed).
// Uses the default ModelConfig activation ("tanh") — tests that care about
// EV range use make_inference_cpu_with_activation() instead.
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

static std::shared_ptr<InferenceEngine> make_inference_cpu_with_activation(
        const std::string& activation) {
    ModelConfig cfg;
    cfg.hidden_layers     = 1;
    cfg.hidden_width      = 32;
    cfg.output_activation = activation;
    auto model = std::make_shared<ProYamsNet>(cfg);
    initialize_weights(*model);
    model->eval();
    return std::make_shared<InferenceEngine>(model, torch::Device(torch::kCPU));
}

// Helpers to create fake game instances with known request counts and push to BatchManager.
static std::unique_ptr<GameInstance> make_fake_game(BatchManager& bm, int req_count, int id = 0) {
    auto game = std::make_unique<GameInstance>();
    game->game_id = id;
    game->solver_buffers.request_count = req_count;
    InferenceBatch* batch;
    int offset;
    float* out = bm.reserve(req_count, batch, offset);
    if (out) {
        for (int i = 0; i < req_count * kTensorSize; ++i)
            out[i] = static_cast<float>(id + 1);
        bm.commit(batch, game.get(), offset, req_count);
    }
    return game;
}

// ---------------------------------------------------------------------------
// Batch assembly — parameterized over output activation.
// Each variant verifies that EVs land in the range valid for that activation:
//   sigmoid → [0, 1],  tanh → [-1, 1].
// ---------------------------------------------------------------------------
struct ActivationParam {
    std::string name;
    double      ev_min;
    double      ev_max;
};

class CoordinatorActivationTest
    : public ::testing::TestWithParam<ActivationParam> {};

TEST_P(CoordinatorActivationTest, BatchAssembly_DistributesEVs) {
    const auto& p = GetParam();
    auto engine = make_inference_cpu_with_activation(p.name);

    SelfPlayConfig cfg;
    cfg.max_inference_batch = 1024;
    cfg.min_games_per_batch = 2;
    cfg.batch_timeout_ms    = 50;

    GameQueue available;
    BatchManager bm(1, 1024, false, 50);
    std::atomic<bool> shutdown{false};

    auto g0 = make_fake_game(bm, 10, 0);
    auto g1 = make_fake_game(bm, 20, 1);
    auto g2 = make_fake_game(bm, 15, 2);

    std::thread ct(coordinator_thread<Yams1v1>,
                   std::ref(bm), std::ref(available),
                   std::ref(*engine), std::ref(cfg),
                   std::ref(shutdown), 0, nullptr);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (available.size() < 3 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    shutdown = true;
    bm.shutdown();
    ct.join();

    EXPECT_EQ(available.size(), 3);

    while (auto* g = available.try_pop()) {
        EXPECT_EQ(g->phase, GamePhase::kNeedResolve);
        for (int i = 0; i < g->solver_buffers.request_count; ++i) {
            double ev = g->solver_buffers.evs[i];
            EXPECT_GE(ev, p.ev_min) << "activation=" << p.name;
            EXPECT_LE(ev, p.ev_max) << "activation=" << p.name;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Activations,
    CoordinatorActivationTest,
    ::testing::Values(
        ActivationParam{"sigmoid",  0.0, 1.0},
        ActivationParam{"tanh",    -1.0, 1.0}
    ),
    [](const ::testing::TestParamInfo<ActivationParam>& info) {
        return info.param.name;
    }
);

TEST(CoordinatorTest, BatchOverflow_FlowsToMultipleBatches) {
    auto engine = make_inference_cpu();
    SelfPlayConfig cfg;
    cfg.max_inference_batch = 50;  // very small
    cfg.min_games_per_batch = 1;
    cfg.batch_timeout_ms    = 30;

    GameQueue available;
    BatchManager bm(2, 50, false, 30);
    std::atomic<bool> shutdown{false};

    // Each game has 30 tensors. max_batch=50 → first game fits, second flushes the first.
    auto g0 = make_fake_game(bm, 30, 0);
    auto g1 = make_fake_game(bm, 30, 1);

    std::thread ct(coordinator_thread<Yams1v1>,
                   std::ref(bm), std::ref(available),
                   std::ref(*engine), std::ref(cfg),
                   std::ref(shutdown), 0, nullptr);

    // Wait for BOTH games to flow through.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (available.size() < 2 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    shutdown = true;
    bm.shutdown();
    ct.join();

    EXPECT_EQ(available.size(), 2) << "Both games should be processed since BatchManager handles splitting.";
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

    GameQueue available;
    BatchManager bm(1, 1024, false, 50);
    std::atomic<bool> shutdown{false};

    auto g = make_fake_game(bm, 8, 0);

    std::thread ct(coordinator_thread<Yams1v1>,
                   std::ref(bm), std::ref(available),
                   std::ref(*engine), std::ref(cfg),
                   std::ref(shutdown), 0, nullptr);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (available.size() < 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    shutdown = true;
    bm.shutdown();
    ct.join();

    EXPECT_GE(available.size(), 1) << "Game should have been processed after timeout";
}

// ---------------------------------------------------------------------------
// Logging — respects debug_mode and debug_log_path settings
// ---------------------------------------------------------------------------
TEST(CoordinatorTest, StatsLogging_RespectsDebugSettings) {
    auto engine = make_inference_cpu();
    SelfPlayConfig cfg;
    cfg.max_inference_batch = 1000;
    cfg.min_games_per_batch = 1;
    cfg.batch_timeout_ms    = 1;
    cfg.debug_mode          = true;
    cfg.debug_log_path      = "test_coordinator_metrics.log";

    // Clean up any stale log
    std::filesystem::remove(cfg.debug_log_path);

    GameQueue available;
    BatchManager bm(10, 1000, false, 1); // Timeout is 1ms
    std::atomic<bool> shutdown{false};

    std::thread ct(coordinator_thread<Yams1v1>,
                   std::ref(bm), std::ref(available),
                   std::ref(*engine), std::ref(cfg),
                   std::ref(shutdown), 0, nullptr);

    // Pump 1001 games through to trigger the % 1000 reporting.
    // Each game will be a tiny batch.
    for (int i = 0; i < 1005; ++i) {
        auto g = make_fake_game(bm, 1, i);
        
        // Block until it pops out to avoid flooding the queue too much
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (available.size() == 0 && std::chrono::steady_clock::now() < deadline) {
             std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        available.try_pop(); // clear it
    }

    shutdown = true;
    bm.shutdown();
    ct.join();

    // Verify file exists and has content
    EXPECT_TRUE(std::filesystem::exists(cfg.debug_log_path));
    
    std::ifstream f(cfg.debug_log_path);
    std::string line;
    bool found = false;
    while (std::getline(f, line)) {
        if (line.find("[Coordinator 0 @ iter 1000]") != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "Log file should contain reporting for iteration 1000";

    // Clean up
    std::filesystem::remove(cfg.debug_log_path);
}
