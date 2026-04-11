#pragma once

// ---------------------------------------------------------------------------
// ModelConfig — hyperparameters for ProYamsNet construction.
// ---------------------------------------------------------------------------
struct ModelConfig {
    int    input_size     = 809;    // kTensorSize, must match tensor generation
    int    hidden_layers  = 3;      // Number of hidden layers
    int    hidden_width   = 256;    // Neurons per hidden layer
    double learning_rate  = 0.001;  // Adam optimizer learning rate
    bool   debug_mode     = false;  // Print sample tensor health
};
