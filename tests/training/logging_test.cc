#include <gtest/gtest.h>
#include "training/logging.h"
#include "training/metrics.h"

#include <filesystem>
#include <fstream>
#include <string>

// ---------------------------------------------------------------------------
// log_metrics
// ---------------------------------------------------------------------------

TEST(LoggingTest, LogMetrics_CreatesFile) {
    const std::string path = "/tmp/training_log_test.csv";
    std::filesystem::remove(path);

    TrainingMetrics m;
    m.training_step     = 100;
    m.games_played      = 50;
    m.samples_in_buffer = 10000;
    m.loss              = 0.0123;
    m.temperature       = 0.9;
    m.epsilon           = 0.0;

    log_metrics(path, m);

    EXPECT_TRUE(std::filesystem::exists(path));

    std::ifstream f(path);
    std::string line1, line2;
    ASSERT_TRUE(std::getline(f, line1));
    ASSERT_TRUE(std::getline(f, line2));

    // Check header contains expected columns
    EXPECT_NE(line1.find("step"), std::string::npos);
    EXPECT_NE(line1.find("loss"), std::string::npos);

    // Check data row contains step number
    EXPECT_NE(line2.find("100"), std::string::npos);

    std::filesystem::remove(path);
}

TEST(LoggingTest, LogMetrics_AppendsRows) {
    const std::string path = "/tmp/training_log_append_test.csv";
    std::filesystem::remove(path);

    TrainingMetrics m1, m2;
    m1.training_step = 100;
    m2.training_step = 200;

    log_metrics(path, m1);
    log_metrics(path, m2);

    std::ifstream f(path);
    int line_count = 0;
    std::string line;
    while (std::getline(f, line)) ++line_count;

    // header + 2 data rows
    EXPECT_EQ(line_count, 3);

    std::filesystem::remove(path);
}

TEST(LoggingTest, LogMetrics_NoDoubleHeader) {
    const std::string path = "/tmp/training_log_header_test.csv";
    std::filesystem::remove(path);

    for (int i = 0; i < 5; ++i) {
        TrainingMetrics m;
        m.training_step = i * 100;
        log_metrics(path, m);
    }

    std::ifstream f(path);
    int header_count = 0;
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("step", 0) == 0) ++header_count;

    EXPECT_EQ(header_count, 1);

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// prune_old_checkpoints
// ---------------------------------------------------------------------------

static void touch(const std::string& path) {
    std::ofstream f(path);
    f << "x";
}

TEST(LoggingTest, PruneOldCheckpoints_RemovesOldest) {
    const std::string dir = "/tmp/prune_ckpt_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    // Create fake checkpoint files for steps 100, 200, 300, 400, 500
    for (int s : {100, 200, 300, 400, 500}) {
        std::string stem = dir + "/checkpoint_step_" + std::to_string(s);
        touch(stem + ".model");
        touch(stem + ".optimizer");
        touch(stem + ".buffer");
    }

    prune_old_checkpoints(dir, 3);

    // Steps 100, 200: optimizer/buffer pruned, but .model kept permanently.
    for (int s : {100, 200}) {
        EXPECT_TRUE(std::filesystem::exists(
            dir + "/checkpoint_step_" + std::to_string(s) + ".model"))
            << "Step " << s << " .model should be kept";
        EXPECT_FALSE(std::filesystem::exists(
            dir + "/checkpoint_step_" + std::to_string(s) + ".optimizer"))
            << "Step " << s << " .optimizer should be pruned";
        EXPECT_FALSE(std::filesystem::exists(
            dir + "/checkpoint_step_" + std::to_string(s) + ".buffer"))
            << "Step " << s << " .buffer should be pruned";
    }
    // Steps 300, 400, 500: full checkpoint (model + optimizer + buffer) kept.
    for (int s : {300, 400, 500}) {
        EXPECT_TRUE(std::filesystem::exists(
            dir + "/checkpoint_step_" + std::to_string(s) + ".model"))
            << "Step " << s << " .model should be kept";
        EXPECT_TRUE(std::filesystem::exists(
            dir + "/checkpoint_step_" + std::to_string(s) + ".optimizer"))
            << "Step " << s << " .optimizer should be kept";
        EXPECT_TRUE(std::filesystem::exists(
            dir + "/checkpoint_step_" + std::to_string(s) + ".buffer"))
            << "Step " << s << " .buffer should be kept";
    }

    std::filesystem::remove_all(dir);
}

TEST(LoggingTest, PruneOldCheckpoints_Noop_WhenBelowLimit) {
    const std::string dir = "/tmp/prune_ckpt_noop_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    touch(dir + "/checkpoint_step_100.model");
    touch(dir + "/checkpoint_step_100.optimizer");

    prune_old_checkpoints(dir, 5);

    EXPECT_TRUE(std::filesystem::exists(dir + "/checkpoint_step_100.model"));

    std::filesystem::remove_all(dir);
}

TEST(LoggingTest, PruneOldCheckpoints_NonexistentDir) {
    // Should not throw.
    EXPECT_NO_THROW(prune_old_checkpoints("/tmp/nonexistent_ckpt_dir_xyz", 3));
}
