#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "config/config_loader.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string write_tmp_yaml(const std::string& content) {
    char path[] = "/tmp/yams_cfg_XXXXXX.yaml";
    // Use mkstemp-style: write directly.
    std::string p = std::string("/tmp/yams_cfg_test_") +
                    std::to_string(reinterpret_cast<uintptr_t>(&content)) +
                    ".yaml";
    std::ofstream f(p);
    f << content;
    return p;
}

// ---------------------------------------------------------------------------
// default_config
// ---------------------------------------------------------------------------
TEST(ConfigLoader, DefaultConfigHasExpectedNumSteps) {
    AppConfig cfg = default_config();
    EXPECT_EQ(cfg.num_steps, 100000);
}

TEST(ConfigLoader, DefaultConfigHasExpectedTrainingDefaults) {
    AppConfig cfg = default_config();
    EXPECT_EQ(cfg.training.replay_capacity,  2'000'000);
    EXPECT_EQ(cfg.training.min_buffer_size,  10'000);
    EXPECT_EQ(cfg.training.train_batch_size, 512);
    EXPECT_DOUBLE_EQ(cfg.training.initial_temperature, 1.0);
    EXPECT_EQ(cfg.training.td_mode, TDMode::kMC);
}

TEST(ConfigLoader, DefaultConfigHasSelfPlayDefaults) {
    AppConfig cfg = default_config();
    EXPECT_EQ(cfg.training.self_play.num_workers, 16);
    EXPECT_EQ(cfg.training.self_play.num_games,   512);
}

TEST(ConfigLoader, DefaultConfigHasModelDefaults) {
    AppConfig cfg = default_config();
    EXPECT_EQ(cfg.training.model.hidden_layers, 3);
    EXPECT_EQ(cfg.training.model.hidden_width,  256);
}

// ---------------------------------------------------------------------------
// load_config: override a scalar
// ---------------------------------------------------------------------------
TEST(ConfigLoader, LoadOverridesNumSteps) {
    std::string p = write_tmp_yaml("num_steps: 42\n");
    AppConfig cfg = load_config(p);
    std::remove(p.c_str());
    EXPECT_EQ(cfg.num_steps, 42);
}

TEST(ConfigLoader, LoadOverridesTrainingField) {
    std::string p = write_tmp_yaml(
        "training:\n"
        "  train_batch_size: 128\n");
    AppConfig cfg = load_config(p);
    std::remove(p.c_str());
    EXPECT_EQ(cfg.training.train_batch_size, 128);
    // Other fields keep their defaults.
    EXPECT_EQ(cfg.training.replay_capacity, 2'000'000);
}

TEST(ConfigLoader, LoadOverridesSelfPlayField) {
    std::string p = write_tmp_yaml(
        "training:\n"
        "  self_play:\n"
        "    num_workers: 4\n");
    AppConfig cfg = load_config(p);
    std::remove(p.c_str());
    EXPECT_EQ(cfg.training.self_play.num_workers, 4);
    EXPECT_EQ(cfg.training.self_play.num_games, 512);  // default preserved
}

TEST(ConfigLoader, LoadOverridesModelField) {
    std::string p = write_tmp_yaml(
        "training:\n"
        "  model:\n"
        "    hidden_width: 512\n");
    AppConfig cfg = load_config(p);
    std::remove(p.c_str());
    EXPECT_EQ(cfg.training.model.hidden_width, 512);
}

TEST(ConfigLoader, LoadParsesTdModeMC) {
    std::string p = write_tmp_yaml(
        "training:\n"
        "  td_mode: mc\n");
    AppConfig cfg = load_config(p);
    std::remove(p.c_str());
    EXPECT_EQ(cfg.training.td_mode, TDMode::kMC);
}

TEST(ConfigLoader, LoadParsesTdModeTD0) {
    std::string p = write_tmp_yaml(
        "training:\n"
        "  td_mode: td0\n");
    AppConfig cfg = load_config(p);
    std::remove(p.c_str());
    EXPECT_EQ(cfg.training.td_mode, TDMode::kTD0);
}

TEST(ConfigLoader, LoadParsesTdModeTDLambda) {
    std::string p = write_tmp_yaml(
        "training:\n"
        "  td_mode: tdlambda\n"
        "  td_lambda: 0.9\n");
    AppConfig cfg = load_config(p);
    std::remove(p.c_str());
    EXPECT_EQ(cfg.training.td_mode, TDMode::kTDLambda);
    EXPECT_DOUBLE_EQ(cfg.training.td_lambda, 0.9);
}

// ---------------------------------------------------------------------------
// load_config: error paths
// ---------------------------------------------------------------------------
TEST(ConfigLoader, MissingFileThrows) {
    EXPECT_THROW(load_config("/nonexistent/path/cfg.yaml"), std::runtime_error);
}

TEST(ConfigLoader, InvalidTdModeThrows) {
    std::string p = write_tmp_yaml(
        "training:\n"
        "  td_mode: bogus\n");
    EXPECT_THROW(load_config(p), std::runtime_error);
    std::remove(p.c_str());
}
