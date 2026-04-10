#include "model/pro_yams_net.h"

// ---------------------------------------------------------------------------
// ProYamsNet implementation
// ---------------------------------------------------------------------------

ProYamsNet::ProYamsNet(const ModelConfig& config)
    : config_(config), num_hidden_(config.hidden_layers) {

    // First hidden layer: input_size → hidden_width
    hidden_layers_.push_back(
        register_module("hidden_0",
            torch::nn::Linear(config.input_size, config.hidden_width)));

    // Subsequent hidden layers: hidden_width → hidden_width
    for (int i = 1; i < config.hidden_layers; ++i) {
        hidden_layers_.push_back(
            register_module("hidden_" + std::to_string(i),
                torch::nn::Linear(config.hidden_width, config.hidden_width)));
    }

    // Output layer: hidden_width → 1
    output_layer_ = register_module("output",
        torch::nn::Linear(config.hidden_width, 1));
}

torch::Tensor ProYamsNet::forward(torch::Tensor x) {
    for (int i = 0; i < num_hidden_; ++i) {
        x = torch::relu(hidden_layers_[i]->forward(x));
    }
    x = torch::sigmoid(output_layer_->forward(x));
    return x;
}

// ---------------------------------------------------------------------------
// initialize_weights: Xavier uniform + zero biases
// ---------------------------------------------------------------------------
void initialize_weights(ProYamsNet& model) {
    for (auto& module : model.modules(/*include_self=*/false)) {
        if (auto* linear = module->as<torch::nn::LinearImpl>()) {
            torch::nn::init::xavier_uniform_(linear->weight);
            torch::nn::init::zeros_(linear->bias);
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
