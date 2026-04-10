#pragma once

#include "training/training_config.h"

// ---------------------------------------------------------------------------
// AppConfig — top-level application configuration.
//
// Wraps TrainingConfig and adds run-level parameters not specific to
// a single training session (e.g. how long to train).
// ---------------------------------------------------------------------------
struct AppConfig {
    TrainingConfig training;
    int num_steps = 100000;  // Gradient steps to run in train mode
    std::string mode = "info";
    std::string config_path;
    std::string checkpoint_path;
    uint64_t seed = 42;
};
