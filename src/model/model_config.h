#pragma once

#include <string>

// ---------------------------------------------------------------------------
// ModelConfig — hyperparameters for ProYamsNet construction.
// ---------------------------------------------------------------------------
struct ModelConfig {
    int    input_size     = 809;    // kTensorSize, must match tensor generation
    int    hidden_layers  = 3;      // Number of hidden layers
    int    hidden_width   = 256;    // Neurons per hidden layer
    double learning_rate  = 0.001;  // Adam optimizer learning rate
    bool   debug_mode     = false;  // Print sample tensor health
    std::string debug_log_path = "";
    std::string output_activation = "tanh";    // "tanh" or "sigmoid"
    std::string loss_function     = "mse";    // "mse" or "bce"
    std::string architecture      = "resnet"; // "mlp" or "resnet"
};
