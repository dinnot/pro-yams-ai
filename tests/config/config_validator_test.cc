#include <gtest/gtest.h>

#include "config/app_config.h"
#include "config/config_validator.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static AppConfig valid_cfg() {
    return AppConfig{};  // All defaults are valid.
}

// ---------------------------------------------------------------------------
// Valid config
// ---------------------------------------------------------------------------
TEST(ConfigValidator, DefaultConfigPasses) {
    ValidationResult r = validate_config(valid_cfg());
    EXPECT_TRUE(r.ok);
    EXPECT_TRUE(r.errors.empty());
}

// ---------------------------------------------------------------------------
// num_steps
// ---------------------------------------------------------------------------
TEST(ConfigValidator, ZeroNumStepsFails) {
    AppConfig cfg = valid_cfg();
    cfg.num_steps = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, NegativeNumStepsFails) {
    AppConfig cfg = valid_cfg();
    cfg.num_steps = -1;
    EXPECT_FALSE(validate_config(cfg).ok);
}

// ---------------------------------------------------------------------------
// Replay buffer
// ---------------------------------------------------------------------------
TEST(ConfigValidator, ZeroReplayCapacityFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.replay_capacity = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, MinBufferExceedsCapacityFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.replay_capacity = 100;
    cfg.training.min_buffer_size = 200;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, BatchSizeExceedsMinBufferFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.min_buffer_size  = 100;
    cfg.training.train_batch_size = 200;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, ZeroTrainBatchSizeFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.train_batch_size = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

// ---------------------------------------------------------------------------
// Intervals and checkpointing
// ---------------------------------------------------------------------------
TEST(ConfigValidator, ZeroModelSwapIntervalFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.model_swap_interval = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, ZeroCheckpointIntervalFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.checkpoint_interval = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, ZeroMaxCheckpointsFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.max_checkpoints = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

// ---------------------------------------------------------------------------
// Temperature
// ---------------------------------------------------------------------------
TEST(ConfigValidator, ZeroInitialTemperatureFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.initial_temperature = 0.0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, MinTemperatureExceedsInitialFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.min_temperature     = 2.0;
    cfg.training.initial_temperature = 1.0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, TemperatureDecayAboveOneFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.temperature_decay = 1.1;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, TemperatureDecayZeroFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.temperature_decay = 0.0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, TemperatureDecayOfOneIsValid) {
    AppConfig cfg = valid_cfg();
    cfg.training.temperature_decay = 1.0;
    EXPECT_TRUE(validate_config(cfg).ok);
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------
TEST(ConfigValidator, NegativeEvalIntervalFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.eval_interval = -1;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, ZeroEvalIntervalIsValid) {
    // 0 = disabled, which is intentionally allowed.
    AppConfig cfg = valid_cfg();
    cfg.training.eval_interval = 0;
    EXPECT_TRUE(validate_config(cfg).ok);
}

TEST(ConfigValidator, ZeroEvalGamesFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.eval_games = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

// ---------------------------------------------------------------------------
// Self-play
// ---------------------------------------------------------------------------
TEST(ConfigValidator, ZeroNumWorkersFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.self_play.num_workers = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, ZeroNumGamesFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.self_play.num_games = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, SmallMaxInferenceBatchFails) {
    AppConfig cfg = valid_cfg();
    // Must be >= kMaxAfterstates (512) to prevent buffer overflow.
    cfg.training.self_play.max_inference_batch = 511;
    EXPECT_FALSE(validate_config(cfg).ok);
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
TEST(ConfigValidator, ZeroHiddenLayersFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.model.hidden_layers = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, ZeroHiddenWidthFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.model.hidden_width = 0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

TEST(ConfigValidator, ZeroLearningRateFails) {
    AppConfig cfg = valid_cfg();
    cfg.training.model.learning_rate = 0.0;
    EXPECT_FALSE(validate_config(cfg).ok);
}

// ---------------------------------------------------------------------------
// Error accumulation
// ---------------------------------------------------------------------------
TEST(ConfigValidator, MultipleErrorsAccumulated) {
    AppConfig cfg = valid_cfg();
    cfg.num_steps                   = -1;
    cfg.training.train_batch_size   = 0;
    cfg.training.model.hidden_width = 0;
    ValidationResult r = validate_config(cfg);
    EXPECT_FALSE(r.ok);
    EXPECT_GE(r.errors.size(), 3u);
}
