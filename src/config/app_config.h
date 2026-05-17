#pragma once

#include "distil/distil_config.h"
#include "training/training_config.h"

// ---------------------------------------------------------------------------
// AppConfig — top-level application configuration.
//
// Wraps the per-mode configs (TrainingConfig for `train` / `eval`,
// DistilConfig for `distil`) plus a few run-level parameters. The active
// substruct is selected by `mode`; the unused one keeps its defaults.
// ---------------------------------------------------------------------------
struct AppConfig {
    TrainingConfig training;
    DistilConfig   distil;
    int num_steps = 100000;  // Gradient steps in train mode (synonym for distil.max_steps in distil mode)
    std::string mode = "info";
    std::string config_path;
    std::string checkpoint_path;  // Init weights from this checkpoint (fresh training)
    std::string resume_path;      // Resume full training state from this checkpoint dir
    uint64_t seed = 42;
};
