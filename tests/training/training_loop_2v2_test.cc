#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <thread>

#include "engine/game_traits.h"
#include "model/model_config.h"
#include "model/pro_yams_net.h"
#include "solver/precomputed_tables.h"
#include "training/training_loop.h"

// ---------------------------------------------------------------------------
// 2v2 training-loop smoke test (Task 7.G).
//
// Verifies that the templated cascade (game_queues / batch_manager / worker /
// coordinator / orchestrator / training_loop) end-to-end:
//   - constructs a TrainingLoopT<Yams2v2>
//   - plays a few 2v2 games to completion
//   - emits per-step training samples into the buffer
//   - takes a handful of training steps with a finite loss
//
// This is a correctness gate, not a quality bar. It is the smallest test that
// would fail loudly if any layer of the cascade dropped Yams2v2 along the
// way (e.g. a missing explicit instantiation, a tensor-size mismatch, or a
// hardcoded kNumPlayers=2 leak).
// ---------------------------------------------------------------------------

static PrecomputedTables* g_tables_2v2 = nullptr;

static void ensure_tables() {
    if (!g_tables_2v2) {
        // Note: torch::set_num_threads / at::set_num_interop_threads must be
        // called before any libtorch work begins, AND can only be called once
        // per process. The 1v1 training_loop_test fixture in the same binary
        // already does this — so we deliberately skip it here.
        try {
            torch::set_num_threads(1);
        } catch (...) {}
        try {
            at::set_num_interop_threads(1);
        } catch (...) {}
        g_tables_2v2 = new PrecomputedTables();
        init_precomputed_tables(*g_tables_2v2);
    }
}

static TrainingConfig make_small_2v2_config(const std::string& ckpt_dir,
                                             const std::string& log_path) {
    TrainingConfig cfg;
    cfg.self_play.num_workers         = 2;
    cfg.self_play.num_games           = 4;
    cfg.self_play.max_inference_batch = 1024;
    cfg.self_play.min_games_per_batch = 1;
    cfg.self_play.batch_timeout_ms    = 10;

    cfg.td_mode          = TDMode::kMC;
    cfg.replay_capacity  = 10000;
    cfg.min_buffer_size  = 10;
    cfg.train_batch_size = 8;

    cfg.model_swap_interval   = 5;
    cfg.checkpoint_interval   = 1000000;  // effectively disabled
    cfg.max_checkpoints       = 3;

    cfg.initial_temperature = 0.0;
    cfg.min_temperature     = 0.0;
    cfg.temperature_decay   = 1.0;

    // Disable eval — the evaluator is 1v1-only for now.
    cfg.eval_interval = 0;

    cfg.checkpoint_dir = ckpt_dir;
    cfg.log_path       = log_path;

    cfg.model.hidden_layers = 1;
    cfg.model.hidden_width  = 32;
    cfg.model.game_variant  = kGameVariant2v2;
    cfg.model.input_size    = Yams2v2::kTensorSize;

    return cfg;
}

// ---------------------------------------------------------------------------
// Smoke: constructs and runs a few steps on Yams2v2 without crashing.
// ---------------------------------------------------------------------------
TEST(TrainingLoop2v2Test, RunsAFewSteps) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/tl2v2_test_run";
    const std::string log_path = "/tmp/tl2v2_test_run.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    TrainingConfig cfg = make_small_2v2_config(ckpt_dir, log_path);
    TrainingLoopT<Yams2v2> loop(cfg, *g_tables_2v2,
                                torch::Device(torch::kCPU),
                                torch::Device(torch::kCPU));

    loop.run(10);

    TrainingMetrics m = loop.metrics();
    EXPECT_GT(m.games_played, 0) << "Should complete at least one 2v2 game";
    EXPECT_GT(m.samples_in_buffer, 0);
    EXPECT_GE(m.loss, 0.0);
    EXPECT_FALSE(std::isnan(m.loss)) << "Loss must be finite";
    EXPECT_FALSE(std::isinf(m.loss));

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}
