#pragma once

#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include "model/pro_yams_net.h"
#include "model/inference.h"

// ---------------------------------------------------------------------------
// ModelTrainer — manages model + Adam optimizer for one training thread.
//
// Provides a simple train_step() interface and checkpoint serialization.
// All libtorch types are kept internal; callers use raw float/double arrays.
// ---------------------------------------------------------------------------
class ModelTrainer {
public:
    /// Construct a fresh trainer with Xavier-initialized model on the given device.
    ModelTrainer(const ModelConfig& config, torch::Device device);

    /// Perform one training step (forward + backward + Adam step).
    ///
    /// @param states     Input float array [batch_size × kTensorSize]
    /// @param targets    Target win probabilities [batch_size] in [0, 1]
    /// @param batch_size Number of samples
    /// @return MSE loss for this batch
    double train_step(const float* states, const double* targets, int batch_size);

    /// Access the current model (const; use clone_for_inference for a copy).
    const ProYamsNet& model() const { return *model_; }

    /// Non-const model access (e.g., for evaluation with model.eval()).
    ProYamsNet& model_mut() { return *model_; }

    /// Raw model pointer (for callers that need a pointer, e.g. run_evaluation).
    ProYamsNet* model_ptr() { return model_.get(); }

    /// Device this trainer and model reside on.
    torch::Device device() const { return device_; }

    /// Create a deep copy of the current model for inference use.
    std::shared_ptr<ProYamsNet> clone_for_inference(torch::Device device);

    /// Save a checkpoint: model weights + optimizer state + metadata.
    /// Writes two files: path + ".model" and path + ".optimizer".
    void save_checkpoint(const std::string& path, int training_step,
                          double temperature, double epsilon);

    /// Load a checkpoint and restore all state.
    /// On return, training_step / temperature / epsilon hold the saved values.
    /// Supports both the current (hidden_blocks) and legacy (hidden_N) formats.
    void load_checkpoint(const std::string& path, int& training_step,
                          double& temperature, double& epsilon);

    /// Load only the model weights from a checkpoint (no optimizer state).
    /// Useful for inference-only contexts where the optimizer is not needed and
    /// the .optimizer file may contain CUDA tensors incompatible with CPU.
    void load_weights(const std::string& path);

    /// Read architecture metadata from a checkpoint without loading weights.
    /// Missing fields fall back to sensible defaults (hidden_width=256,
    /// hidden_layers=3, architecture="mlp").  Safe to call before constructing
    /// a ModelTrainer so the trainer is built with the right shape.
    static ModelConfig config_from_checkpoint(const std::string& path);

    /// Total number of train_step() calls made so far.
    int training_step_count() const { return training_step_; }

private:
    std::shared_ptr<ProYamsNet>        model_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    torch::Device                       device_;
    ModelConfig                         config_;
    int                                 training_step_ = 0;

    // High-priority CUDA stream reserved for training so train kernels are
    // scheduled ahead of inference batches that share the GPU. Inference
    // coordinators use normal-priority pool streams (see coordinator.cc).
    std::optional<c10::cuda::CUDAStream> train_stream_;
};

// ---------------------------------------------------------------------------
// save_managed_checkpoint — save a checkpoint and prune old ones.
//
// Checkpoints are named: checkpoint_step_{step}.model / .optimizer
// Old checkpoints beyond max_checkpoints are deleted.
// ---------------------------------------------------------------------------
void save_managed_checkpoint(ModelTrainer& trainer, const std::string& dir,
                              int step, double temperature, double epsilon,
                              int max_checkpoints = 5);
