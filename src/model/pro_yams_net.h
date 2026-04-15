#pragma once

#include <torch/torch.h>
#include <vector>
#include "model/model_config.h"

// ---------------------------------------------------------------------------
// ResBlockImpl — residual block: two linear layers with LayerNorm and a skip.
// ---------------------------------------------------------------------------
struct ResBlockImpl : torch::nn::Module {
    torch::nn::Linear lin1{nullptr}, lin2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    ResBlockImpl(int dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResBlock);

// ---------------------------------------------------------------------------
// ProYamsNet — feedforward network for win-probability prediction.
//
// Architecture (resnet):
//   Input(input_size) → Linear+ReLU → [ResBlock] × (hidden_layers-1) → Linear → activation
// Architecture (mlp):
//   Input(input_size) → [Linear+ReLU] × hidden_layers → Linear → activation
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
    torch::nn::Sequential hidden_blocks_{nullptr};
    torch::nn::Linear output_layer_{nullptr};
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
