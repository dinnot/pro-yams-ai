#include "model/pro_yams_net.h"

// ---------------------------------------------------------------------------
// ResBlockImpl implementation
// ---------------------------------------------------------------------------

ResBlockImpl::ResBlockImpl(int dim) {
    lin1  = register_module("lin1",  torch::nn::Linear(dim, dim));
    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    lin2  = register_module("lin2",  torch::nn::Linear(dim, dim));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
}

torch::Tensor ResBlockImpl::forward(torch::Tensor x) {
    auto residual = x;
    x = torch::relu(norm1->forward(lin1->forward(x)));
    x = norm2->forward(lin2->forward(x));
    return torch::relu(x + residual);
}

// ---------------------------------------------------------------------------
// ProYamsNet implementation
// ---------------------------------------------------------------------------

ProYamsNet::ProYamsNet(const ModelConfig& config) : config_(config) {
    hidden_blocks_ = register_module("hidden_blocks", torch::nn::Sequential());

    // Projection layer: input_size → hidden_width
    hidden_blocks_->push_back(torch::nn::Linear(config.input_size, config.hidden_width));
    hidden_blocks_->push_back(torch::nn::ReLU());

    // Remaining hidden layers
    for (int i = 1; i < config.hidden_layers; ++i) {
        if (config.architecture == "resnet") {
            hidden_blocks_->push_back(ResBlock(config.hidden_width));
        } else {
            hidden_blocks_->push_back(torch::nn::Linear(config.hidden_width, config.hidden_width));
            hidden_blocks_->push_back(torch::nn::ReLU());
        }
    }

    // Output layer: hidden_width → 1
    output_layer_ = register_module("output", torch::nn::Linear(config.hidden_width, 1));
}

torch::Tensor ProYamsNet::forward_logits(torch::Tensor x) {
    x = hidden_blocks_->forward(x);
    return output_layer_->forward(x);   // raw pre-activation outputs
}

torch::Tensor ProYamsNet::forward(torch::Tensor x) {
    x = forward_logits(std::move(x));
    if (config_.output_activation == "sigmoid") {
        x = torch::sigmoid(x);
    } else {
        x = torch::tanh(x);
    }
    return x;
}

// ---------------------------------------------------------------------------
// initialize_weights: Xavier uniform + zero biases (skips LayerNorm).
// For ResNet, lin2 in each ResBlock is scaled near zero (ReZero-style) so
// each block starts as a near-identity transformation for stable training.
// ---------------------------------------------------------------------------
void initialize_weights(ProYamsNet& model) {
    for (auto& module : model.modules(/*include_self=*/false)) {
        if (auto* linear = module->as<torch::nn::LinearImpl>()) {
            torch::nn::init::xavier_uniform_(linear->weight);
            torch::nn::init::zeros_(linear->bias);
        }
    }
    // Scale down norm2's gamma in each ResBlock so the residual branch starts
    // near-zero (LayerNorm would undo a scaling on lin2.weight, so we suppress
    // norm2's output directly via its learnable gamma parameter).
    for (auto& module : model.modules(/*include_self=*/false)) {
        if (auto* resblock = module->as<ResBlockImpl>()) {
            resblock->norm2->weight.data().mul_(0.01f);
        }
    }
}

// ---------------------------------------------------------------------------
// get_device: CUDA if available, else CPU
// ---------------------------------------------------------------------------
torch::Device get_device() {
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
}
