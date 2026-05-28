#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>

#include <torch/torch.h>

#include "distil/distil_config.h"
#include "distil/distil_loop.h"
#include "engine/game_traits.h"
#include "solver/precomputed_tables.h"
#include "test_tables.h"

// Tables + libtorch thread clamping are shared with the other distil test
// files via tests/distil/test_tables.h — at::set_num_interop_threads can
// only be called once per process.
static PrecomputedTables* g_tables = nullptr;
static void ensure_tables() { g_tables = &distil_test::tables(); }

static DistilConfig make_small_distil_config(const std::string& ckpt_dir,
                                             const std::string& log_path) {
    DistilConfig cfg;
    cfg.teacher_kind              = TeacherKind::kHeuristic;
    cfg.teacher_heuristic_version = 2;

    cfg.self_play.num_workers         = 2;
    cfg.self_play.num_games           = 4;
    cfg.self_play.max_inference_batch = 1024;
    cfg.self_play.min_games_per_batch = 1;
    cfg.self_play.batch_timeout_ms    = 10;

    // Small replay buffer so the trainer starts quickly.
    cfg.train_batch_size        = 32;
    cfg.replay_buffer_capacity  = 1024;
    cfg.min_samples_to_start    = 64;
    cfg.samples_per_train       = 1.0;

    cfg.checkpoint_interval = 1'000'000;   // effectively disabled
    cfg.max_checkpoints     = 3;

    cfg.max_steps = 20;
    cfg.min_steps = 0;

    cfg.use_duel_margin_maximization   = true;
    cfg.duel_margin_maximization_scale = 4000.0;

    cfg.checkpoint_dir = ckpt_dir;
    cfg.log_path       = log_path;
    cfg.log_dir        = ".";

    // Small student so the test stays cheap.
    cfg.student_model.hidden_layers     = 1;
    cfg.student_model.hidden_width      = 32;
    cfg.student_model.output_activation = "tanh";
    cfg.student_model.architecture      = "mlp";
    // game_variant + input_size default to Yams1v1 — overridden in 2v2 test.

    return cfg;
}

// ---------------------------------------------------------------------------
// Yams1v1 — runs to max_steps with finite loss
// ---------------------------------------------------------------------------
TEST(DistilLoopTest, RunsAFewSteps) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/distil_loop_test_1v1";
    const std::string log_path = "/tmp/distil_loop_test_1v1.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    DistilConfig cfg = make_small_distil_config(ckpt_dir, log_path);

    DistilLoop loop(cfg, *g_tables,
                    torch::Device(torch::kCPU),
                    torch::Device(torch::kCPU));
    loop.run();

    TrainingMetrics m = loop.metrics();
    EXPECT_EQ(m.training_step, cfg.max_steps);
    EXPECT_GT(m.total_samples_trained, 0);
    EXPECT_GE(m.loss, 0.0);
    EXPECT_TRUE(std::isfinite(m.loss));

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}

// ---------------------------------------------------------------------------
// stop() unblocks the loop even when most time is spent in draw_batch
// ---------------------------------------------------------------------------
TEST(DistilLoopTest, StopExitsEarly) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/distil_loop_test_stop";
    const std::string log_path = "/tmp/distil_loop_test_stop.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    DistilConfig cfg = make_small_distil_config(ckpt_dir, log_path);
    cfg.max_steps = 1'000'000;   // never reached without stop()

    auto loop = std::make_unique<DistilLoop>(cfg, *g_tables,
                                             torch::Device(torch::kCPU),
                                             torch::Device(torch::kCPU));
    std::thread stopper([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        loop->stop();
    });

    loop->run();
    stopper.join();

    EXPECT_LT(loop->metrics().training_step, 1'000'000);

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}

// ---------------------------------------------------------------------------
// Checkpoint round-trip on a tight interval
// ---------------------------------------------------------------------------
TEST(DistilLoopTest, CheckpointWritesToDisk) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/distil_loop_test_ckpt";
    const std::string log_path = "/tmp/distil_loop_test_ckpt.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    DistilConfig cfg = make_small_distil_config(ckpt_dir, log_path);
    cfg.checkpoint_interval = 10;
    cfg.max_steps           = 20;

    DistilLoop loop(cfg, *g_tables,
                    torch::Device(torch::kCPU),
                    torch::Device(torch::kCPU));
    loop.run();

    // Expect at least one .model file in ckpt_dir.
    bool found = false;
    for (auto& e : std::filesystem::directory_iterator(ckpt_dir)) {
        if (e.path().extension() == ".model") { found = true; break; }
    }
    EXPECT_TRUE(found) << "No .model checkpoint written";

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}

// ---------------------------------------------------------------------------
// Yams2v2 smoke
// ---------------------------------------------------------------------------
TEST(DistilLoopTest, RunsAFewSteps_2v2) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/distil_loop_test_2v2";
    const std::string log_path = "/tmp/distil_loop_test_2v2.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    DistilConfig cfg = make_small_distil_config(ckpt_dir, log_path);
    cfg.student_model.game_variant = kGameVariant2v2;
    cfg.student_model.input_size   = Yams2v2::kTensorSize;
    cfg.max_steps                  = 8;

    DistilLoop2v2 loop(cfg, *g_tables,
                       torch::Device(torch::kCPU),
                       torch::Device(torch::kCPU));
    loop.run();

    TrainingMetrics m = loop.metrics();
    EXPECT_EQ(m.training_step, cfg.max_steps);
    EXPECT_GT(m.total_samples_trained, 0);
    EXPECT_GE(m.loss, 0.0);
    EXPECT_TRUE(std::isfinite(m.loss));

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}

// ---------------------------------------------------------------------------
// NN teacher: a missing checkpoint surfaces as a runtime_error.
// ---------------------------------------------------------------------------
TEST(DistilLoopTest, NNTeacherMissingCheckpointThrows) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/distil_loop_test_nn_missing";
    const std::string log_path = "/tmp/distil_loop_test_nn_missing.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    DistilConfig cfg = make_small_distil_config(ckpt_dir, log_path);
    cfg.teacher_kind = TeacherKind::kNN;
    cfg.teacher_checkpoint_path = "/tmp/no_such_path_xyz";

    // Underlying torch loader throws c10::Error (derived from std::exception
    // but not from std::runtime_error), so widen the expected type.
    EXPECT_THROW({
        DistilLoop loop(cfg, *g_tables,
                        torch::Device(torch::kCPU),
                        torch::Device(torch::kCPU));
    }, std::exception);

    std::filesystem::remove_all(ckpt_dir);
}
