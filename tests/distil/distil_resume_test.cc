#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <torch/torch.h>

#include "distil/distil_config.h"
#include "distil/distil_loop.h"
#include "distil/distil_resume.h"
#include "engine/game_traits.h"
#include "solver/precomputed_tables.h"
#include "test_tables.h"

static PrecomputedTables* g_tables = nullptr;
static void ensure_tables() { g_tables = &distil_test::tables(); }

static DistilConfig make_small_cfg(const std::string& ckpt_dir,
                                   const std::string& log_dir) {
    DistilConfig cfg;
    cfg.teacher_kind              = TeacherKind::kHeuristic;
    cfg.teacher_heuristic_version = 2;
    cfg.reference_heuristic_version = 2;

    cfg.self_play.num_workers         = 2;
    cfg.self_play.num_games           = 4;
    cfg.self_play.max_inference_batch = 1024;
    cfg.self_play.min_games_per_batch = 1;
    cfg.self_play.batch_timeout_ms    = 10;

    cfg.train_batch_size        = 32;
    cfg.replay_buffer_capacity  = 1024;
    cfg.min_samples_to_start    = 64;
    cfg.samples_per_train       = 1.0;

    cfg.checkpoint_interval = 5;            // tight so we save quickly
    cfg.max_checkpoints     = 5;
    cfg.max_steps           = 15;
    cfg.min_steps           = 0;
    cfg.eval_interval       = 1'000'000;    // never fires
    cfg.eval_games          = 4;
    cfg.final_eval_games    = 8;
    cfg.convergence_win_rate_delta = -1.0;
    cfg.convergence_patience       = 3;

    cfg.use_duel_margin_maximization   = true;
    cfg.duel_margin_maximization_scale = 4000.0;

    cfg.checkpoint_dir = ckpt_dir;
    cfg.log_dir        = log_dir;
    cfg.log_path       = log_dir + "/distil_log.csv";

    cfg.student_model.hidden_layers     = 1;
    cfg.student_model.hidden_width      = 16;
    cfg.student_model.output_activation = "tanh";
    cfg.student_model.architecture      = "mlp";
    return cfg;
}

// ---------------------------------------------------------------------------
// Resume round-trip: run → save → resume → step counter restored.
// ---------------------------------------------------------------------------

TEST(DistilResumeTest, RoundTripRestoresTrainingStep) {
    ensure_tables();

    const std::string ckpt = "/tmp/distil_resume_test_roundtrip";
    const std::string logs = ckpt + "_logs";
    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);

    DistilConfig cfg = make_small_cfg(ckpt, logs);

    // First run: train 15 steps (last checkpoint at step 15).
    {
        DistilLoop loop(cfg, *g_tables, torch::kCPU, torch::kCPU);
        loop.run();
        EXPECT_EQ(loop.metrics().training_step, 15);
    }

    // Confirm at least one .model + .optimizer pair exists.
    bool have_model = false, have_optim = false;
    for (auto& e : std::filesystem::directory_iterator(ckpt)) {
        if (e.path().extension() == ".model")     have_model = true;
        if (e.path().extension() == ".optimizer") have_optim = true;
    }
    ASSERT_TRUE(have_model);
    ASSERT_TRUE(have_optim);

    // Second run: a fresh loop that resumes.
    DistilConfig cfg2 = cfg;
    cfg2.max_steps = 25;   // would normally run 25 from 0; resume should continue at 15
    DistilLoop loop2(cfg2, *g_tables, torch::kCPU, torch::kCPU);
    ASSERT_TRUE(resume_distil_from_checkpoint(loop2, ckpt));
    EXPECT_EQ(loop2.metrics().training_step, 15)
        << "Resume should set training_step to the saved step";

    loop2.run();
    EXPECT_EQ(loop2.metrics().training_step, 25)
        << "Continuation should reach the new max_steps";

    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);
}

// ---------------------------------------------------------------------------
// Missing checkpoint dir → returns false (not an exception).
// ---------------------------------------------------------------------------

TEST(DistilResumeTest, EmptyDirReturnsFalse) {
    ensure_tables();

    const std::string ckpt = "/tmp/distil_resume_test_empty";
    const std::string logs = ckpt + "_logs";
    std::filesystem::remove_all(ckpt);
    std::filesystem::create_directories(ckpt);
    std::filesystem::remove_all(logs);

    DistilConfig cfg = make_small_cfg(ckpt, logs);
    DistilLoop loop(cfg, *g_tables, torch::kCPU, torch::kCPU);

    EXPECT_FALSE(resume_distil_from_checkpoint(loop, ckpt));
    EXPECT_EQ(loop.metrics().training_step, 0)
        << "training_step should be untouched when no checkpoint to resume from";

    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);
}

// ---------------------------------------------------------------------------
// Missing .optimizer sibling → throws (refuses to silently restart Adam).
// ---------------------------------------------------------------------------

TEST(DistilResumeTest, MissingOptimizerThrows) {
    ensure_tables();

    const std::string ckpt = "/tmp/distil_resume_test_no_optim";
    const std::string logs = ckpt + "_logs";
    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);

    DistilConfig cfg = make_small_cfg(ckpt, logs);
    {
        DistilLoop loop(cfg, *g_tables, torch::kCPU, torch::kCPU);
        loop.run();
    }

    // Delete every .optimizer file — model files left intact.
    for (auto& e : std::filesystem::directory_iterator(ckpt)) {
        if (e.path().extension() == ".optimizer") {
            std::filesystem::remove(e.path());
        }
    }

    DistilLoop loop2(cfg, *g_tables, torch::kCPU, torch::kCPU);
    EXPECT_THROW(resume_distil_from_checkpoint(loop2, ckpt), std::runtime_error);

    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);
}

// ---------------------------------------------------------------------------
// init_distil_from_checkpoint: weights-only path; training_step stays at 0.
// ---------------------------------------------------------------------------

TEST(DistilResumeTest, InitFromCheckpoint_DoesNotRestoreStep) {
    ensure_tables();

    const std::string ckpt = "/tmp/distil_resume_test_init";
    const std::string logs = ckpt + "_logs";
    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);

    DistilConfig cfg = make_small_cfg(ckpt, logs);
    {
        DistilLoop loop(cfg, *g_tables, torch::kCPU, torch::kCPU);
        loop.run();
    }

    DistilLoop loop2(cfg, *g_tables, torch::kCPU, torch::kCPU);
    ASSERT_TRUE(init_distil_from_checkpoint(loop2, ckpt));
    EXPECT_EQ(loop2.metrics().training_step, 0)
        << "init_from_checkpoint loads weights only — step stays at 0";

    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);
}
