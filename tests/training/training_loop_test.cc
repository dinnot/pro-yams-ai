#include <gtest/gtest.h>
#include "training/training_loop.h"
#include "training/resume.h"
#include "solver/precomputed_tables.h"
#include "model/pro_yams_net.h"

#include <chrono>
#include <filesystem>
#include <thread>

// ---------------------------------------------------------------------------
// Shared setup — init PrecomputedTables once (~350ms).
// ---------------------------------------------------------------------------
static PrecomputedTables* g_tables = nullptr;

static void ensure_tables() {
    if (!g_tables) {
        // Limit PyTorch threading to avoid TBB/OpenMP initialization overhead.
        // The first forward() call in a new thread can hang for 60+ seconds
        // after another PyTorch process has run (shared TBB state teardown).
        torch::set_num_threads(1);
        at::set_num_interop_threads(1);

        g_tables = new PrecomputedTables();
        init_precomputed_tables(*g_tables);
    }
}

static TrainingConfig make_small_config(const std::string& ckpt_dir,
                                         const std::string& log_path) {
    TrainingConfig cfg;
    cfg.self_play.num_workers         = 2;
    cfg.self_play.num_games           = 4;
    cfg.self_play.max_inference_batch = 1024;
    cfg.self_play.min_games_per_batch = 1;
    cfg.self_play.batch_timeout_ms    = 10;

    cfg.td_mode          = TDMode::kMC;
    cfg.replay_capacity  = 10000;
    cfg.min_buffer_size  = 10;   // train as soon as 10 samples collected
    cfg.train_batch_size = 8;

    cfg.model_swap_interval   = 5;
    cfg.checkpoint_interval   = 1000000;  // effectively disabled for most tests
    cfg.max_checkpoints       = 3;

    cfg.initial_temperature = 0.0;
    cfg.min_temperature     = 0.0;
    cfg.temperature_decay   = 1.0;

    cfg.checkpoint_dir = ckpt_dir;
    cfg.log_path       = log_path;

    cfg.model.hidden_layers = 1;
    cfg.model.hidden_width  = 32;

    return cfg;
}

// ---------------------------------------------------------------------------
// Run for N steps, verify metrics make sense.
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, RunsNSteps) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/tl_test_run";
    const std::string log_path = "/tmp/tl_test_run.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    TrainingConfig cfg = make_small_config(ckpt_dir, log_path);
    TrainingLoop loop(cfg, *g_tables,
                      torch::Device(torch::kCPU),
                      torch::Device(torch::kCPU));

    loop.run(20);

    TrainingMetrics m = loop.metrics();
    EXPECT_EQ(m.training_step, 20);
    EXPECT_GT(m.games_played, 0);
    EXPECT_GT(m.samples_in_buffer, 0);
    EXPECT_GE(m.loss, 0.0);

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}

// ---------------------------------------------------------------------------
// stop() exits the loop early.
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, StopExitsEarly) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/tl_test_stop";
    const std::string log_path = "/tmp/tl_test_stop.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    TrainingConfig cfg = make_small_config(ckpt_dir, log_path);
    // Ask for many steps — we'll stop early.
    auto loop = std::make_unique<TrainingLoop>(cfg, *g_tables,
                                               torch::Device(torch::kCPU),
                                               torch::Device(torch::kCPU));

    std::thread stopper([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        loop->stop();
    });

    loop->run(1'000'000);  // would never finish without stop()
    stopper.join();

    // Should have stopped before reaching 1M steps.
    EXPECT_LT(loop->metrics().training_step, 1'000'000);

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}

// ---------------------------------------------------------------------------
// Temperature decays over training steps.
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, TemperatureDecays) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/tl_test_temp";
    const std::string log_path = "/tmp/tl_test_temp.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    TrainingConfig cfg = make_small_config(ckpt_dir, log_path);
    cfg.initial_temperature = 1.0;
    cfg.min_temperature     = 0.01;
    cfg.temperature_decay   = 0.9;  // decays fast

    TrainingLoop loop(cfg, *g_tables,
                      torch::Device(torch::kCPU),
                      torch::Device(torch::kCPU));

    loop.run(50);

    double final_temp = loop.metrics().temperature;
    // After 50 steps with decay=0.9: 1.0 * 0.9^50 ≈ 0.005 → clamped to 0.01
    EXPECT_LT(final_temp, 1.0);
    EXPECT_GE(final_temp, cfg.min_temperature - 1e-9);

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}

// ---------------------------------------------------------------------------
// Checkpoint save/load round-trip via resume_from_checkpoint.
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, Checkpoint_ResumeRestoresStep) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/tl_test_resume";
    const std::string log_path = "/tmp/tl_test_resume.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    TrainingConfig cfg = make_small_config(ckpt_dir, log_path);
    cfg.checkpoint_interval = 10;  // save every 10 steps
    cfg.max_checkpoints     = 2;

    // First run: train 10 steps → triggers checkpoint at step 10
    {
        TrainingLoop loop(cfg, *g_tables,
                          torch::Device(torch::kCPU),
                          torch::Device(torch::kCPU));
        loop.run(10);
        EXPECT_EQ(loop.metrics().training_step, 10);
    }

    // Second run: resume and confirm step is restored
    {
        TrainingLoop loop2(cfg, *g_tables,
                           torch::Device(torch::kCPU),
                           torch::Device(torch::kCPU));
        bool resumed = resume_from_checkpoint(loop2, ckpt_dir);
        EXPECT_TRUE(resumed);
        EXPECT_EQ(loop2.metrics().training_step, 10);
    }

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}

// ---------------------------------------------------------------------------
// resume_from_checkpoint returns false when directory has no checkpoints.
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, Resume_NoCheckpoints_ReturnsFalse) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/tl_test_noresume";
    const std::string log_path = "/tmp/tl_test_noresume.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::create_directories(ckpt_dir);

    TrainingConfig cfg = make_small_config(ckpt_dir, log_path);
    TrainingLoop loop(cfg, *g_tables,
                      torch::Device(torch::kCPU),
                      torch::Device(torch::kCPU));

    EXPECT_FALSE(resume_from_checkpoint(loop, ckpt_dir));

    std::filesystem::remove_all(ckpt_dir);
}

// ---------------------------------------------------------------------------
// Buffer fills up across multiple recycled games.
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, BufferFillsUp) {
    ensure_tables();

    const std::string ckpt_dir = "/tmp/tl_test_buffer";
    const std::string log_path = "/tmp/tl_test_buffer.csv";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);

    TrainingConfig cfg = make_small_config(ckpt_dir, log_path);
    cfg.min_buffer_size  = 500;
    cfg.train_batch_size = 32;

    TrainingLoop loop(cfg, *g_tables,
                      torch::Device(torch::kCPU),
                      torch::Device(torch::kCPU));
    loop.run(10);

    EXPECT_GT(loop.metrics().samples_in_buffer, 0);

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove(log_path);
}
