#pragma once

#include <torch/torch.h>
#include <vector>
#include "model/model_config.h"

// ---------------------------------------------------------------------------
// ProYamsNet — feedforward MLP for win-probability prediction.
//
// Architecture:
//   Input(input_size) → [Linear → ReLU] × hidden_layers → Linear → Sigmoid
//   Output: P(Win) ∈ (0, 1)
//
// Usage: std::shared_ptr<ProYamsNet> — do NOT use TORCH_MODULE macro, as this
// model is owned via shared_ptr throughout the codebase.
// ---------------------------------------------------------------------------
class ProYamsNet : public torch::nn::Module {
public:
    /// Construct the network with the given configuration.
    explicit ProYamsNet(const ModelConfig& config);

    /// Forward pass.
    /// @param x Input tensor of shape [batch_size, input_size]
    /// @return Win probability tensor of shape [batch_size, 1], values in (0,1)
    torch::Tensor forward(torch::Tensor x);

    /// Access the configuration used to build this model.
    const ModelConfig& config() const { return config_; }

private:
    ModelConfig config_;
    std::vector<torch::nn::Linear> hidden_layers_;
    torch::nn::Linear output_layer_{nullptr};
    int num_hidden_;
};

// ---------------------------------------------------------------------------
// Free function: Xavier uniform weight initialization.
// Call after construction for reproducible training starts.
// ---------------------------------------------------------------------------
void initialize_weights(ProYamsNet& model);

// ---------------------------------------------------------------------------
// Free function: Get the best available device (CUDA first, then CPU).
// ---------------------------------------------------------------------------
torch::Device get_device();
