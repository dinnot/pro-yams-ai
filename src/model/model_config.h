#pragma once

#include <string>

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
    int    input_size     = 986;    // kTensorSize (Yams1v1::kTensorSize for 1v1,
                                    // Yams2v2::kTensorSize for 2v2 — must match
                                    // tensor generation)
    int    hidden_layers  = 3;      // Number of hidden layers
    int    hidden_width   = 256;    // Neurons per hidden layer
    double learning_rate  = 0.001;  // Adam optimizer learning rate
    bool   debug_mode     = false;  // Print sample tensor health
    std::string debug_log_path = "";
    std::string output_activation = "tanh";    // "tanh" or "sigmoid"
    std::string loss_function     = "mse";    // "mse" or "bce"
    std::string architecture      = "mlp"; // "mlp" or "resnet"
};
