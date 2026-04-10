#include <gtest/gtest.h>
#include "self_play/orchestrator.h"
#include "self_play/training_data.h"
#include "solver/precomputed_tables.h"
#include "model/inference.h"
#include "model/pro_yams_net.h"

#include <chrono>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Shared setup — init once since PrecomputedTables takes ~350ms.
// ---------------------------------------------------------------------------
static PrecomputedTables* g_tables = nullptr;
static std::shared_ptr<InferenceEngine> g_engine;

static void ensure_init() {
    if (!g_tables) {
        // Limit PyTorch threading to avoid TBB/OpenMP initialization overhead.
        // The first forward() call in a new thread can hang for 60+ seconds
        // after another PyTorch process has run (shared TBB state teardown).
        torch::set_num_threads(1);
        at::set_num_interop_threads(1);

        g_tables = new PrecomputedTables();
        init_precomputed_tables(*g_tables);

        ModelConfig cfg;
        cfg.hidden_layers = 1;
        cfg.hidden_width  = 32;
        auto model = std::make_shared<ProYamsNet>(cfg);
        initialize_weights(*model);
        model->eval();
        g_engine = std::make_shared<InferenceEngine>(
            model, torch::Device(torch::kCPU));

        // Warm up PyTorch from multiple threads: TBB initializes per-thread
        // on first use, so warming from new threads pre-initializes the
        // coordinator's TBB context and prevents multi-second first-call latency.
        std::vector<float> dummy(kTensorSize, 0.0f);
        double dummy_out = 0.0;
        g_engine->batch_inference(dummy.data(), 1, &dummy_out);
        // Run warm-up from two background threads to cover coordinator + workers.
        for (int t = 0; t < 2; ++t) {
            std::thread([&] {
                g_engine->batch_inference(dummy.data(), 1, &dummy_out);
            }).join();
        }
    }
}

// ---------------------------------------------------------------------------
// Integration: small pool of games completes end-to-end.
// ---------------------------------------------------------------------------
TEST(IntegrationTest, GamesCompleteWithNN) {
    ensure_init();

    SelfPlayConfig cfg;
    cfg.num_workers         = 2;
    cfg.num_games           = 4;
    cfg.max_inference_batch = 1024;
    cfg.min_games_per_batch = 1;
    cfg.batch_timeout_ms    = 10;

    SolverConfig solver_cfg{0.0, 0.0, false};

    SelfPlayOrchestrator orch(cfg, *g_tables, *g_engine, solver_cfg);
    orch.start();

    // Collect all 4 completed games (with generous timeout).
    std::vector<GameInstance*> completed;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(240);

    while (completed.size() < 4 &&
           std::chrono::steady_clock::now() < deadline) {
        GameInstance* buf[4];
        int n = orch.collect_completed(buf, 4);
        for (int i = 0; i < n; ++i)
            completed.push_back(buf[i]);
        if (n == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    orch.stop();

    ASSERT_EQ(static_cast<int>(completed.size()), 4);

    for (GameInstance* g : completed) {
        EXPECT_EQ(g->trajectory_length, GameInstance::kMaxTrajectorySteps)
            << "Expected 156 trajectory steps for game " << g->game_id;
        double r = g->result;
        EXPECT_TRUE(r == 0.0 || r == 0.5 || r == 1.0)
            << "Invalid result: " << r;
    }
}

// ---------------------------------------------------------------------------
// Training data extraction from completed games.
// ---------------------------------------------------------------------------
TEST(IntegrationTest, TrainingDataExtraction_ValidSamples) {
    ensure_init();

    SelfPlayConfig cfg;
    cfg.num_workers         = 2;
    cfg.num_games           = 2;
    cfg.max_inference_batch = 1024;
    cfg.min_games_per_batch = 1;
    cfg.batch_timeout_ms    = 10;

    SolverConfig solver_cfg{0.0, 0.0, false};

    SelfPlayOrchestrator orch(cfg, *g_tables, *g_engine, solver_cfg);
    orch.start();

    // Wait for at least 1 completed game.
    GameInstance* completed[2] = {nullptr, nullptr};
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(240);
    int n_done = 0;
    while (n_done < 2 && std::chrono::steady_clock::now() < deadline) {
        n_done += orch.collect_completed(completed + n_done, 2 - n_done);
        if (n_done < 2)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    orch.stop();
    ASSERT_GE(n_done, 1);

    GameInstance* g = completed[0];
    ASSERT_NE(g, nullptr);

    int traj_len = g->trajectory_length;
    ASSERT_GT(traj_len, 0);

    std::vector<TrainingSample> samples(traj_len);

    // TD(0)
    int ns = extract_training_samples(*g, TDMode::kTD0, 0.0,
                                       samples.data(), traj_len);
    EXPECT_EQ(ns, traj_len);
    for (int i = 0; i < ns; ++i) {
        EXPECT_GE(samples[i].target, 0.0);
        EXPECT_LE(samples[i].target, 1.0);
    }

    // MC
    std::vector<TrainingSample> mc_samples(traj_len);
    extract_training_samples(*g, TDMode::kMC, 0.0, mc_samples.data(), traj_len);
    // MC targets should differ from TD(0) for at least some steps.
    bool any_diff = false;
    for (int i = 0; i < ns; ++i)
        if (std::abs(samples[i].target - mc_samples[i].target) > 1e-6)
            any_diff = true;
    EXPECT_TRUE(any_diff) << "MC and TD(0) targets should differ";
}

// ---------------------------------------------------------------------------
// Recycle game — recycled game runs again without errors.
// ---------------------------------------------------------------------------
TEST(IntegrationTest, RecycleGame_RunsAgain) {
    ensure_init();

    SelfPlayConfig cfg;
    cfg.num_workers         = 2;
    cfg.num_games           = 2;
    cfg.max_inference_batch = 1024;
    cfg.min_games_per_batch = 1;
    cfg.batch_timeout_ms    = 10;

    SolverConfig solver_cfg{0.0, 0.0, false};

    SelfPlayOrchestrator orch(cfg, *g_tables, *g_engine, solver_cfg);
    orch.start();

    // Collect one completed game.
    GameInstance* done = nullptr;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(240);
    while (!done && std::chrono::steady_clock::now() < deadline) {
        done = nullptr;
        int n = orch.collect_completed(&done, 1);
        if (n == 0) {
            done = nullptr;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    ASSERT_NE(done, nullptr);

    // Recycle it — should complete again.
    orch.recycle_game(done, 0xDEADBEEF);

    GameInstance* done2 = nullptr;
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(240);
    while (!done2 && std::chrono::steady_clock::now() < deadline) {
        int n = orch.collect_completed(&done2, 1);
        if (n == 0) {
            done2 = nullptr;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    orch.stop();
    ASSERT_NE(done2, nullptr);
    EXPECT_EQ(done2->trajectory_length, GameInstance::kMaxTrajectorySteps);
}
