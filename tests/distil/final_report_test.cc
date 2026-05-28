#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include <torch/torch.h>

#include "distil/distil_config.h"
#include "distil/distil_loop.h"
#include "engine/game_traits.h"
#include "solver/precomputed_tables.h"
#include "test_tables.h"

// Tables + libtorch thread clamping live in tests/distil/test_tables.h —
// see distil_loop_test.cc for rationale.
static PrecomputedTables* g_tables = nullptr;
static void ensure_tables() { g_tables = &distil_test::tables(); }

static DistilConfig make_tiny_cfg(const std::string& ckpt_dir,
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

    cfg.checkpoint_interval = 1'000'000;  // disabled
    cfg.max_checkpoints     = 3;

    cfg.max_steps     = 10;
    cfg.min_steps     = 0;
    cfg.eval_interval = 1'000'000;        // never fires per-interval eval
    cfg.eval_games    = 10;
    cfg.final_eval_games = 16;            // tiny so the test stays fast
    cfg.convergence_win_rate_delta = -1.0; // disable convergence
    cfg.convergence_patience       = 3;

    cfg.use_duel_margin_maximization   = true;
    cfg.duel_margin_maximization_scale = 4000.0;

    cfg.checkpoint_dir = ckpt_dir;
    cfg.log_dir        = log_dir;
    cfg.log_path       = log_dir + "/distil_log.csv";

    cfg.student_model.hidden_layers     = 1;
    cfg.student_model.hidden_width      = 32;
    cfg.student_model.output_activation = "tanh";
    cfg.student_model.architecture      = "mlp";
    return cfg;
}

// ---------------------------------------------------------------------------
// final_report always runs and populates rates, regardless of stop reason.
// ---------------------------------------------------------------------------

TEST(FinalReportTest, FiresOnMaxSteps_PopulatesRatesAndStopReason) {
    ensure_tables();

    const std::string ckpt = "/tmp/distil_final_test_maxsteps_ckpt";
    const std::string logs = "/tmp/distil_final_test_maxsteps_logs";
    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);

    DistilConfig cfg = make_tiny_cfg(ckpt, logs);
    DistilLoop loop(cfg, *g_tables,
                    torch::Device(torch::kCPU),
                    torch::Device(torch::kCPU));
    loop.run();

    EXPECT_EQ(loop.stop_reason(), DistilLoop::StopReason::kMaxSteps);
    // Heuristic-vs-heuristic always lands in [0,1]; never NaN.
    EXPECT_TRUE(std::isfinite(loop.final_teacher_win_rate()));
    EXPECT_TRUE(std::isfinite(loop.final_student_win_rate()));
    EXPECT_GE(loop.final_teacher_win_rate(), 0.0);
    EXPECT_LE(loop.final_teacher_win_rate(), 1.0);
    EXPECT_GE(loop.final_student_win_rate(), 0.0);
    EXPECT_LE(loop.final_student_win_rate(), 1.0);

    // CSV row written.
    const std::string csv = logs + "/distil_final.csv";
    ASSERT_TRUE(std::filesystem::exists(csv));
    std::ifstream f(csv);
    std::string header, row;
    std::getline(f, header);
    std::getline(f, row);
    EXPECT_NE(header.find("stop_reason"), std::string::npos);
    EXPECT_NE(row.find("max_steps"), std::string::npos);

    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);
}

TEST(FinalReportTest, FiresOnStopSignal) {
    ensure_tables();

    const std::string ckpt = "/tmp/distil_final_test_signal_ckpt";
    const std::string logs = "/tmp/distil_final_test_signal_logs";
    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);

    DistilConfig cfg = make_tiny_cfg(ckpt, logs);
    cfg.max_steps = 1'000'000;   // would never finish without external stop

    auto loop = std::make_unique<DistilLoop>(cfg, *g_tables,
                                             torch::Device(torch::kCPU),
                                             torch::Device(torch::kCPU));

    std::thread stopper([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        loop->stop();
    });

    loop->run();
    stopper.join();

    // Either kStopSignal or kQueueDrained is acceptable — both reflect that
    // the loop exited because the external stop fired (the second one happens
    // when stop_flag and queue.stop race such that the drawer notices the
    // queue first).
    auto r = loop->stop_reason();
    EXPECT_TRUE(r == DistilLoop::StopReason::kStopSignal ||
                r == DistilLoop::StopReason::kQueueDrained)
        << "Unexpected stop reason after external stop()";

    EXPECT_LT(loop->metrics().training_step, 1'000'000);
    EXPECT_TRUE(std::isfinite(loop->final_teacher_win_rate()));
    EXPECT_TRUE(std::isfinite(loop->final_student_win_rate()));

    ASSERT_TRUE(std::filesystem::exists(logs + "/distil_final.csv"));

    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);
}

TEST(FinalReportTest, FiresOnConvergence) {
    ensure_tables();

    const std::string ckpt = "/tmp/distil_final_test_conv_ckpt";
    const std::string logs = "/tmp/distil_final_test_conv_logs";
    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);

    // Force convergence on the very first eval: huge delta, patience 1.
    DistilConfig cfg = make_tiny_cfg(ckpt, logs);
    cfg.max_steps                  = 100'000;
    cfg.eval_interval              = 5;      // first eval at step 5
    cfg.convergence_win_rate_delta = 1.0;    // any gap < 1.0 (always true)
    cfg.convergence_patience       = 1;

    DistilLoop loop(cfg, *g_tables,
                    torch::Device(torch::kCPU),
                    torch::Device(torch::kCPU));
    loop.run();

    EXPECT_EQ(loop.stop_reason(), DistilLoop::StopReason::kConvergence);
    EXPECT_LT(loop.metrics().training_step, 1000)
        << "Should converge in the first few eval intervals";
    EXPECT_TRUE(std::isfinite(loop.final_teacher_win_rate()));
    EXPECT_TRUE(std::isfinite(loop.final_student_win_rate()));

    // Confirm CSV row records the convergence reason.
    std::ifstream f(logs + "/distil_final.csv");
    std::string header, row;
    std::getline(f, header);
    std::getline(f, row);
    EXPECT_NE(row.find("convergence"), std::string::npos);

    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);
}

// ---------------------------------------------------------------------------
// final_report uses final_eval_games, which can differ from eval_games.
// Verify the CSV row reports `final_eval_games` (not the per-interval count).
// ---------------------------------------------------------------------------

TEST(FinalReportTest, CsvReportsFinalEvalGamesCount) {
    ensure_tables();

    const std::string ckpt = "/tmp/distil_final_test_games_ckpt";
    const std::string logs = "/tmp/distil_final_test_games_logs";
    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);

    DistilConfig cfg = make_tiny_cfg(ckpt, logs);
    cfg.eval_games       = 5;
    cfg.final_eval_games = 23;   // distinctive prime, easy to spot

    DistilLoop loop(cfg, *g_tables,
                    torch::Device(torch::kCPU),
                    torch::Device(torch::kCPU));
    loop.run();

    std::ifstream f(logs + "/distil_final.csv");
    std::string header, row;
    std::getline(f, header);
    std::getline(f, row);
    // Last column is final_eval_games; comparing the suffix is enough.
    EXPECT_NE(row.find(",23"), std::string::npos)
        << "CSV row didn't contain final_eval_games=23: " << row;

    std::filesystem::remove_all(ckpt);
    std::filesystem::remove_all(logs);
}
