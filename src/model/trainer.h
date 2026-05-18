#pragma once

#include <limits>
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

    /// Submit one training step (forward + backward + Adam step) and return
    /// the most-recently-computed loss.
    ///
    /// The current step's GPU work is queued asynchronously; its loss tensor
    /// is materialised at the start of the NEXT train_step() call (or on
    /// demand via wait_until_step_complete()). This deferred-sync pattern
    /// lets the caller's CPU-side batch sampling overlap with the GPU's
    /// compute of the previous batch — otherwise the GPU sits idle waiting
    /// for the CPU to sample the next batch, and CPU sits idle waiting on
    /// loss.item() inside train_step.
    ///
    /// Returns:
    ///   - NaN on the very first call (no completed step yet)
    ///   - on subsequent calls, the loss of the most recently MATERIALISED
    ///     step. That's the previous train_step if no intervening
    ///     wait_until_step_complete() forced an earlier materialisation; if
    ///     one did, the cached value from that wait is returned (the just-
    ///     submitted step's loss is not yet computed).
    ///
    /// Memory safety: the .item() at the start of the next call fully
    /// synchronises the train stream, so the pinned staging buffer is safe
    /// to overwrite by the time we memcpy into it.
    double train_step(const float* states, const double* targets, int batch_size);

    /// Block until the most recent train_step's GPU work has fully completed
    /// and return the most-recently-computed loss (cached if no step is
    /// pending). Returns NaN only if no train_step has ever been called.
    ///
    /// Idempotent — safe to call any number of times. Callers should invoke
    /// this before reading model state that depends on the latest update
    /// (e.g. running evaluation). save_checkpoint / clone_for_inference /
    /// load_* call it internally.
    double wait_until_step_complete();

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

    // Pinned-memory staging buffers, reused across train_step calls. Pinned
    // memory enables async CPU→GPU transfer (non_blocking=true), so the
    // model.forward() call can be queued without waiting for the transfer
    // to complete. Allocated lazily on first train_step; grown if a larger
    // batch arrives.
    //
    // Single buffer is sufficient: with deferred .item() (see train_step
    // docs), wait_until_step_complete() syncs the train stream before we
    // overwrite the buffer on the next call, so the buffer is free at that
    // point. Double-buffering would only be needed if we wanted >1 batch in
    // flight on the GPU at once.
    torch::Tensor pinned_state_buffer_;   // [max_batch_size, input_size] float32
    torch::Tensor pinned_target_buffer_;  // [max_batch_size, 1] float32
    int           pinned_capacity_ = 0;

    // Loss tensor from the most recently SUBMITTED (but not yet materialised)
    // training step. .item() on this tensor blocks until the train stream is
    // drained, so it's both the loss value and the "GPU step is done"
    // barrier. Detached so it doesn't keep the autograd graph alive.
    torch::Tensor pending_loss_tensor_;
    bool          has_pending_loss_ = false;

    // Cached scalar loss of the most-recently MATERIALISED training step
    // (updated whenever we .item() the pending tensor). Returned by
    // train_step / wait_until_step_complete when there's nothing pending so
    // callers always get a stable last-known value rather than NaN.
    double last_finalized_loss_ = std::numeric_limits<double>::quiet_NaN();
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
