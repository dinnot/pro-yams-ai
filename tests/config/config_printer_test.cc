#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "config/app_config.h"
#include "config/config_printer.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string print_to_string(const AppConfig& cfg) {
    std::ostringstream oss;
    print_config(cfg, oss);
    return oss.str();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
TEST(ConfigPrinter, ContainsNumSteps) {
    AppConfig cfg;
    cfg.num_steps = 99999;
    EXPECT_NE(print_to_string(cfg).find("num_steps: 99999"), std::string::npos);
}

TEST(ConfigPrinter, ContainsTrainingSection) {
    AppConfig cfg;
    std::string out = print_to_string(cfg);
    EXPECT_NE(out.find("training:"), std::string::npos);
}

TEST(ConfigPrinter, ContainsReplayCapacity) {
    AppConfig cfg;
    cfg.training.replay_capacity = 1234567;
    EXPECT_NE(print_to_string(cfg).find("1234567"), std::string::npos);
}

TEST(ConfigPrinter, ContainsSelfPlaySection) {
    AppConfig cfg;
    std::string out = print_to_string(cfg);
    EXPECT_NE(out.find("self_play:"), std::string::npos);
    EXPECT_NE(out.find("num_workers:"), std::string::npos);
}

TEST(ConfigPrinter, ContainsModelSection) {
    AppConfig cfg;
    std::string out = print_to_string(cfg);
    EXPECT_NE(out.find("model:"), std::string::npos);
    EXPECT_NE(out.find("learning_rate:"), std::string::npos);
}

TEST(ConfigPrinter, TdModeShownAsMC) {
    AppConfig cfg;
    cfg.training.td_mode = TDMode::kMC;
    EXPECT_NE(print_to_string(cfg).find("td_mode: mc"), std::string::npos);
}

TEST(ConfigPrinter, TdModeShownAsTD0) {
    AppConfig cfg;
    cfg.training.td_mode = TDMode::kTD0;
    EXPECT_NE(print_to_string(cfg).find("td_mode: td0"), std::string::npos);
}

TEST(ConfigPrinter, TdModeShownAsTDLambda) {
    AppConfig cfg;
    cfg.training.td_mode = TDMode::kTDLambda;
    EXPECT_NE(print_to_string(cfg).find("td_mode: tdlambda"), std::string::npos);
}

TEST(ConfigPrinter, CustomTemperatureShown) {
    AppConfig cfg;
    cfg.training.initial_temperature = 2.5;
    EXPECT_NE(print_to_string(cfg).find("initial_temperature: 2.5"), std::string::npos);
}

TEST(ConfigPrinter, CheckpointDirShown) {
    AppConfig cfg;
    cfg.training.checkpoint_dir = "my_checkpoints";
    EXPECT_NE(print_to_string(cfg).find("my_checkpoints"), std::string::npos);
}
