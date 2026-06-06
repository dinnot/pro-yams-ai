#pragma once

#include <string>

#include "engine/game_traits.h"  // Yams1v1::kTensorSize, kTensorVersionLatest

// ---------------------------------------------------------------------------
// Game variant tag for the trained model. Persisted to the checkpoint so a
// 1v1 model can't be silently loaded into a 2v2 process (or vice versa).
//   1 → Yams1v1 (tensor size 986, 2 players)
//   2 → Yams2v2 (tensor size 2126, 4 players)
// ---------------------------------------------------------------------------
constexpr int kGameVariant1v1 = 1;
constexpr int kGameVariant2v2 = 2;

// ---------------------------------------------------------------------------
// ModelConfig — hyperparameters for ProYamsNet construction.
// ---------------------------------------------------------------------------
struct ModelConfig {
    int    game_variant   = kGameVariant1v1;  // 1 = Yams1v1, 2 = Yams2v2
    // Tensor layout version this model consumes (append-only; see game_traits.h).
    // Default is the latest layout for a fresh 1v1 model. NOTE: readers of OLD
    // checkpoints do NOT rely on this default — trainer.cc reads the persisted
    // tag with an explicit fallback of 1 (pre-Group-G checkpoints predate it).
    int    tensor_version = kTensorVersionLatest;
    // Must equal tensor_size_for_version(tensor_version, game_variant) — enforced
    // by config_validator. Default mirrors (1v1, latest).
    int    input_size     = Yams1v1::kTensorSize;
    int    hidden_layers  = 3;      // Number of hidden layers
    int    hidden_width   = 256;    // Neurons per hidden layer
    double learning_rate  = 0.001;  // Adam optimizer learning rate
    bool   debug_mode     = false;  // Print sample tensor health
    std::string debug_log_path = "";
    std::string output_activation = "tanh";    // "tanh" or "sigmoid"
    std::string loss_function     = "mse";    // "mse" or "bce"
    std::string architecture      = "mlp"; // "mlp" or "resnet"
};
