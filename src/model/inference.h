#pragma once

#include <memory>
#include <mutex>
#include <torch/torch.h>
#include "model/pro_yams_net.h"

// ---------------------------------------------------------------------------
// InferenceEngine — thread-safe wrapper for running model inference.
//
// Accepts raw float arrays (produced by generate_tensor_batch) and writes
// win probabilities to a double output array.  libtorch types are fully
// hidden from callers.
// ---------------------------------------------------------------------------
class InferenceEngine {
public:
    /// Construct with a model already moved to the desired device.
    InferenceEngine(std::shared_ptr<ProYamsNet> model, torch::Device device);

    /// Run inference on a batch of state tensors.
    ///
    /// @param input      Raw float array of size batch_size × kTensorSize
    /// @param batch_size Number of samples
    /// @param output     Output array (batch_size doubles) — written in place
    void batch_inference(const float* input, int batch_size, double* output);

    /// Run inference on a pre-built CPU tensor (e.g. pinned memory).
    /// Uses non-blocking GPU transfer when possible (pinned → async DMA).
    ///
    /// @param input_tensor  CPU tensor of shape [batch_size, kTensorSize]
    /// @param batch_size    Number of samples
    /// @param output        Output array (batch_size doubles) — written in place
    void batch_inference(torch::Tensor input_tensor, int batch_size, double* output);

    /// Atomically swap the underlying model.
    /// Safe to call while another thread is inside batch_inference.
    void swap_model(std::shared_ptr<ProYamsNet> new_model);

    /// Enable dummy mode for benchmarking (bypasses PyTorch).
    void set_dummy_mode(bool dummy) { dummy_mode_ = dummy; }

    /// The device this engine runs on.
    torch::Device device() const { return device_; }

private:
    std::shared_ptr<ProYamsNet> model_;
    torch::Device device_;
    std::mutex inference_mutex_;
    bool dummy_mode_ = false;
};

// ---------------------------------------------------------------------------
// clone_model — create a deep copy of a model on a target device.
//
// Used to move updated training weights into the inference engine.
// Copies all named parameters by value; buffers (none in this architecture)
// would be copied identically.
// ---------------------------------------------------------------------------
std::shared_ptr<ProYamsNet> clone_model(const ProYamsNet& source,
                                         torch::Device target_device);
