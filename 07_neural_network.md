# Task 07: Neural Network

## Overview

Implement the neural network model using libtorch: a configurable feedforward MLP with sigmoid output that predicts win probability. This task covers the model definition, forward pass, inference interface, serialization, and GPU management.

The model library is the only component that depends on libtorch. It provides a clean interface that accepts raw float arrays from the engine/solver and returns win probabilities, keeping the libtorch boundary well-contained.

## Prerequisites

- Task 01 completed (project scaffolding, libtorch integration)
- Task 06 completed (tensor design, kTensorSize defined)

---

## 1. Model Definition

### 1.1 Architecture

Feedforward MLP with configurable depth and width:

```
Input (kTensorSize = 986)
  → Linear(986, hidden_width) → ReLU
  → Linear(hidden_width, hidden_width) → ReLU
  → ... (repeated for hidden_layers total)
  → Linear(hidden_width, 1) → Sigmoid
  → Output: P(Win) ∈ [0, 1]
```

### 1.2 Model Configuration

```cpp
// src/model/model_config.h

struct ModelConfig {
    int input_size = 986;         // kTensorSize, should match tensor generation
    int hidden_layers = 3;        // Number of hidden layers
    int hidden_width = 256;       // Neurons per hidden layer
    double learning_rate = 0.001; // Adam learning rate
};
```

### 1.3 libtorch Module

```cpp
// src/model/pro_yams_net.h

class ProYamsNet : public torch::nn::Module {
public:
    /// Construct the network from configuration.
    explicit ProYamsNet(const ModelConfig& config);

    /// Forward pass. Input shape: [batch_size, input_size].
    /// Output shape: [batch_size, 1], values in [0, 1].
    torch::Tensor forward(torch::Tensor x);

private:
    std::vector<torch::nn::Linear> hidden_layers_;
    torch::nn::Linear output_layer_{nullptr};
    int num_hidden_;
};
```

**Implementation:**

```cpp
ProYamsNet::ProYamsNet(const ModelConfig& config)
    : num_hidden_(config.hidden_layers) {
    
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
```

### 1.4 Weight Initialization

Use Xavier (Glorot) uniform initialization for weights and zero initialization for biases. This is a good default for ReLU networks and avoids the vanishing/exploding gradient problem at initialization.

```cpp
/// Initialize network weights using Xavier uniform.
void initialize_weights(ProYamsNet& model);
```

**Implementation:**

```cpp
void initialize_weights(ProYamsNet& model) {
    for (auto& module : model.modules(/*include_self=*/false)) {
        if (auto* linear = module->as<torch::nn::LinearImpl>()) {
            torch::nn::init::xavier_uniform_(linear->weight);
            torch::nn::init::zeros_(linear->bias);
        }
    }
}
```

---

## 2. Inference Interface

### 2.1 Single-Batch Inference

The primary interface for the solver and self-play system. Accepts raw float arrays, returns win probabilities. libtorch is completely hidden behind this interface.

```cpp
// src/model/inference.h

class InferenceEngine {
public:
    /// Construct with a model and device.
    /// The model is moved to the specified device (CPU or CUDA).
    InferenceEngine(std::shared_ptr<ProYamsNet> model, torch::Device device);

    /// Run inference on a batch of state tensors.
    /// Input: raw float array of shape [batch_size × kTensorSize], contiguous.
    /// Output: win probabilities written to output array, one per sample.
    ///
    /// @param input Raw float tensor data (batch_size * kTensorSize floats)
    /// @param batch_size Number of samples in the batch
    /// @param output Output array (batch_size doubles)
    void batch_inference(const float* input, int batch_size, double* output);

    /// Swap the underlying model (for model updates during training).
    /// Thread-safe: uses atomic pointer swap.
    void swap_model(std::shared_ptr<ProYamsNet> new_model);

    /// Get the current device.
    torch::Device device() const { return device_; }

private:
    std::shared_ptr<ProYamsNet> model_;
    torch::Device device_;
    std::mutex inference_mutex_;  // Protects model_ during swap
};
```

**Implementation of batch_inference:**

```cpp
void InferenceEngine::batch_inference(const float* input, int batch_size,
                                       double* output) {
    // Wrap raw float array as a CPU tensor (no copy)
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto input_tensor = torch::from_blob(
        const_cast<float*>(input),
        {batch_size, kTensorSize},
        options
    );

    torch::Tensor result;
    {
        torch::NoGradGuard no_grad;  // Disable gradient computation
        std::lock_guard<std::mutex> lock(inference_mutex_);
        
        // Move to GPU, run forward pass, move result back to CPU
        auto gpu_input = input_tensor.to(device_);
        auto gpu_output = model_->forward(gpu_input);
        result = gpu_output.to(torch::kCPU).contiguous();
    }

    // Copy results to output array
    auto accessor = result.accessor<float, 2>();
    for (int i = 0; i < batch_size; ++i) {
        output[i] = static_cast<double>(accessor[i][0]);
    }
}
```

### 2.2 Thread Safety

During training, the inference thread and the training thread use separate model instances. The `swap_model` method atomically replaces the inference model with updated weights:

```cpp
void InferenceEngine::swap_model(std::shared_ptr<ProYamsNet> new_model) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    model_ = std::move(new_model);
}
```

The mutex ensures that a forward pass in progress completes before the model is swapped. Since forward passes are fast (sub-millisecond for typical batch sizes), the contention is minimal.

### 2.3 Model Cloning for Swap

When training produces updated weights, we need to create a new model instance with those weights for the inference engine:

```cpp
/// Create a deep copy of a model, moved to the specified device.
/// Used to create the inference model from the training model.
std::shared_ptr<ProYamsNet> clone_model(const ProYamsNet& source,
                                         torch::Device target_device);
```

**Implementation:**

```cpp
std::shared_ptr<ProYamsNet> clone_model(const ProYamsNet& source,
                                         torch::Device target_device) {
    // Serialize to a buffer
    std::ostringstream buffer;
    torch::save(source, buffer);
    
    // Deserialize into a new instance
    auto clone = std::make_shared<ProYamsNet>(source.config());
    std::istringstream input(buffer.str());
    torch::load(*clone, input);
    
    clone->to(target_device);
    clone->eval();  // Set to evaluation mode
    return clone;
}
```

---

## 3. Training Interface

### 3.1 Training Wrapper

A wrapper that manages the model, optimizer, and provides the training step interface:

```cpp
// src/model/trainer.h

class ModelTrainer {
public:
    /// Construct trainer with model config and device.
    ModelTrainer(const ModelConfig& config, torch::Device device);

    /// Perform one training step on a batch of (state, target) pairs.
    ///
    /// @param states Raw float array [batch_size × kTensorSize]
    /// @param targets Target win probabilities [batch_size]
    /// @param batch_size Number of samples
    /// @return Training loss for this batch
    double train_step(const float* states, const double* targets, int batch_size);

    /// Get the current model (for cloning to inference engine).
    const ProYamsNet& model() const { return *model_; }
    
    /// Get a clone of the current model on the specified device.
    std::shared_ptr<ProYamsNet> clone_for_inference(torch::Device device);

    /// Save checkpoint (model weights + optimizer state + metadata).
    void save_checkpoint(const std::string& path, int training_step,
                          double temperature, double epsilon);

    /// Load checkpoint and restore all state.
    void load_checkpoint(const std::string& path, int& training_step,
                          double& temperature, double& epsilon);

    /// Get current training step count.
    int training_step() const { return training_step_; }

private:
    std::shared_ptr<ProYamsNet> model_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    torch::Device device_;
    ModelConfig config_;
    int training_step_ = 0;
};
```

### 3.2 Training Step Implementation

```cpp
double ModelTrainer::train_step(const float* states, const double* targets,
                                 int batch_size) {
    model_->train();  // Set to training mode

    // Create input tensor
    auto state_tensor = torch::from_blob(
        const_cast<float*>(states),
        {batch_size, kTensorSize},
        torch::kFloat32
    ).to(device_);

    // Create target tensor
    // Targets are doubles, need to convert to float tensor
    auto target_tensor = torch::zeros({batch_size, 1}, torch::kFloat32);
    for (int i = 0; i < batch_size; ++i) {
        target_tensor[i][0] = static_cast<float>(targets[i]);
    }
    target_tensor = target_tensor.to(device_);

    // Forward pass
    auto prediction = model_->forward(state_tensor);

    // MSE Loss
    auto loss = torch::mse_loss(prediction, target_tensor);

    // Backward pass
    optimizer_->zero_grad();
    loss.backward();
    optimizer_->step();

    training_step_++;

    return loss.item<double>();
}
```

### 3.3 Optimizer

Use Adam with configurable learning rate. Adam is the standard choice for this type of value function learning — it handles the non-stationary target distribution well and requires minimal tuning.

```cpp
// In ModelTrainer constructor:
optimizer_ = std::make_unique<torch::optim::Adam>(
    model_->parameters(),
    torch::optim::AdamOptions(config.learning_rate)
);
```

---

## 4. Serialization

### 4.1 Checkpoint Format

A checkpoint saves everything needed to resume training:

```cpp
void ModelTrainer::save_checkpoint(const std::string& path, int training_step,
                                    double temperature, double epsilon) {
    torch::serialize::OutputArchive archive;
    
    // Model weights
    model_->save(archive);
    
    // Optimizer state
    // Note: torch::optim::Adam state serialization
    std::ostringstream opt_buffer;
    torch::save(*optimizer_, opt_buffer);
    
    // Metadata
    archive.write("training_step", torch::tensor(training_step));
    archive.write("temperature", torch::tensor(temperature));
    archive.write("epsilon", torch::tensor(epsilon));
    archive.write("hidden_layers", torch::tensor(config_.hidden_layers));
    archive.write("hidden_width", torch::tensor(config_.hidden_width));
    archive.write("input_size", torch::tensor(config_.input_size));
    
    archive.save_to(path + ".model");
    
    // Save optimizer separately (libtorch optimizer serialization)
    torch::save(*optimizer_, path + ".optimizer");
}
```

### 4.2 Checkpoint Loading

```cpp
void ModelTrainer::load_checkpoint(const std::string& path, int& training_step,
                                    double& temperature, double& epsilon) {
    // Load model weights
    torch::serialize::InputArchive archive;
    archive.load_from(path + ".model");
    model_->load(archive);
    model_->to(device_);
    
    // Load optimizer state
    torch::load(*optimizer_, path + ".optimizer");
    
    // Load metadata
    torch::Tensor t;
    archive.read("training_step", t);
    training_step = t.item<int>();
    archive.read("temperature", t);
    temperature = t.item<double>();
    archive.read("epsilon", t);
    epsilon = t.item<double>();
    
    // Verify architecture matches
    archive.read("hidden_layers", t);
    assert(t.item<int>() == config_.hidden_layers);
    archive.read("hidden_width", t);
    assert(t.item<int>() == config_.hidden_width);
    
    training_step_ = training_step;
}
```

### 4.3 Checkpoint Management

Keep the last N checkpoints (configurable). On each save, delete the oldest if over the limit:

```cpp
/// Manage checkpoint files: save new, prune old.
/// Checkpoints are named: checkpoint_step_{step}.model / .optimizer
void save_managed_checkpoint(ModelTrainer& trainer, const std::string& dir,
                              int step, double temperature, double epsilon,
                              int max_checkpoints);
```

---

## 5. GPU Management

### 5.1 Device Selection

```cpp
/// Get the best available device (CUDA if available, else CPU).
torch::Device get_device();
```

**Implementation:**

```cpp
torch::Device get_device() {
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
}
```

### 5.2 Memory Considerations

During simultaneous self-play + training, the GPU holds:
- The training model (weights + gradients + optimizer state)
- The inference model (weights only, no gradients)
- Batch tensors for inference and training

For a 3-layer, 256-width MLP: ~340K parameters × 4 bytes = ~1.4MB per model. With optimizer state (Adam stores 2 momentum terms per parameter): ~4.2MB for training. Plus the inference copy: ~1.4MB. Total model memory: ~7MB — negligible on the 5080 (16GB VRAM).

Batch tensors: 512 samples × 986 features × 4 bytes = ~2.0MB per batch. Also negligible.

GPU memory is not a constraint for this architecture.

---

## 6. File Organization

```
src/model/
├── model_config.h          # ModelConfig struct
├── pro_yams_net.h           # ProYamsNet module declaration
├── pro_yams_net.cc          # Module implementation, weight initialization
├── inference.h              # InferenceEngine class
├── inference.cc             # Inference implementation
├── trainer.h                # ModelTrainer class
├── trainer.cc               # Training step, checkpoint save/load
└── CMakeLists.txt
```

```cmake
# src/model/CMakeLists.txt
add_library(model STATIC
    pro_yams_net.cc
    inference.cc
    trainer.cc
)
target_include_directories(model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(model PUBLIC ${TORCH_LIBRARIES} engine)
```

The `engine` dependency is for `kTensorSize`. No other engine code is used.

---

## 7. Unit Tests

### 7.1 Model Construction Tests (`tests/model/model_test.cc`)

**Basic construction:**
- Create a ProYamsNet with default config. Verify it has the correct number of parameters.
- Create with various configs (2 layers × 128, 4 layers × 512). Verify parameter counts.

**Forward pass shape:**
- Input shape [1, 986] → output shape [1, 1].
- Input shape [64, 986] → output shape [64, 1].
- Input shape [512, 986] → output shape [512, 1].

**Output range:**
- Run forward on random input. Verify all outputs are in (0, 1) — sigmoid guarantees this but good to test.

**Determinism:**
- Same model, same input → same output.
- Different models (different random init) → different output.

### 7.2 Inference Tests (`tests/model/inference_test.cc`)

**batch_inference correctness:**
- Create model, run forward pass via libtorch directly, then via `batch_inference`. Verify results match within floating-point tolerance.

**Batch size variations:**
- Test with batch sizes 1, 16, 64, 256, 512, 1024.

**Model swap:**
- Run inference with model A, swap to model B, run inference again. Verify results change.

### 7.3 Training Tests (`tests/model/trainer_test.cc`)

**Loss decreases:**
- Create a simple dataset (random inputs with known targets). Run 100 training steps. Verify loss decreases over time.

**Overfitting test:**
- Create a tiny dataset (10 samples). Train for 1000 steps. Verify the model can memorize it (loss approaches 0).

**Checkpoint round-trip:**
- Train for 50 steps. Save checkpoint. Create new trainer, load checkpoint. Verify:
  - Training step count matches.
  - Model produces identical output on same input.
  - Temperature and epsilon values are restored.
  - Further training continues from where it left off (loss doesn't spike).

### 7.4 Weight Initialization Tests (`tests/model/init_test.cc`)

**Xavier init:**
- After initialization, verify weight values have reasonable variance (not all zeros, not exploding).
- Run forward pass on random input — output should be near 0.5 (uninformed prior).

---

## 8. Benchmarks

Add to `tests/benchmarks/model_bench.cc`:

- **BM_ForwardPass:** Benchmark forward pass for batch sizes 1, 64, 256, 512, 1024 on GPU.
- **BM_ForwardPassCPU:** Same on CPU for comparison.
- **BM_TrainStep:** Benchmark one training step (forward + backward + optimizer step) for various batch sizes.
- **BM_BatchInference:** Benchmark the full `batch_inference` pipeline (CPU tensor creation → GPU transfer → forward → GPU to CPU → extract results).
- **BM_ModelClone:** Benchmark cloning a model for inference swap.
- **BM_CheckpointSave:** Benchmark saving a checkpoint to disk.
- **BM_CheckpointLoad:** Benchmark loading a checkpoint from disk.

These benchmarks establish how fast inference and training are on the 5080, which directly determines achievable training throughput.

---

## 9. Definition of Done

This task is complete when:

1. `ProYamsNet` constructs correctly with configurable layers and width.
2. Forward pass produces correct output shapes with values in (0, 1).
3. `InferenceEngine::batch_inference` correctly wraps raw float arrays, runs GPU inference, and returns results.
4. `ModelTrainer::train_step` performs forward + backward + optimizer step and returns loss.
5. Loss decreases over training steps on synthetic data.
6. Checkpoints save and load correctly, preserving model weights, optimizer state, and metadata.
7. Architecture configuration (layers, width) is validated on checkpoint load.
8. Model swap via `InferenceEngine::swap_model` works correctly.
9. Xavier weight initialization produces reasonable initial outputs (~0.5).
10. All unit tests pass.
11. Benchmarks establish baseline GPU inference and training throughput.
12. The model library's public interface uses raw float arrays — no libtorch types exposed.
