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
    std::string checkpoint_path;  // Init weights from this checkpoint (fresh training)
    std::string resume_path;      // Resume full training state from this checkpoint dir
    uint64_t seed = 42;
};
