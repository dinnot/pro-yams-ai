#include "model/inference.h"

#include <cassert>
#include "engine/tensor.h"  // kTensorSize

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

InferenceEngine::InferenceEngine(std::shared_ptr<ProYamsNet> model,
                                  torch::Device device)
    : model_(std::move(model)), device_(device) {
    model_->to(device_);
    model_->eval();
}

void InferenceEngine::batch_inference(const float* input, int batch_size,
                                       double* output) {
    assert(batch_size > 0);

    // Wrap raw float array as a CPU tensor (zero-copy)
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto input_tensor = torch::from_blob(
        const_cast<float*>(input),
        {batch_size, kTensorSize},
        options
    );

    batch_inference(input_tensor, batch_size, output);
}

void InferenceEngine::batch_inference(torch::Tensor input_tensor, int batch_size,
                                       double* output) {
    assert(batch_size > 0);

    if (dummy_mode_) {
        for (int i = 0; i < batch_size; ++i) {
            output[i] = 0.5;
        }
        return;
    }

    torch::Tensor result;
    std::shared_ptr<ProYamsNet> local_model;

    // Only lock to safely copy the shared_ptr
    {
        std::lock_guard<std::mutex> lock(inference_mutex_);
        local_model = model_;
    }

    // Execute GPU forward pass lock-free — local_model keeps the old model
    // alive even if swap_model() runs concurrently.
    // When input_tensor is pinned memory, non_blocking=true enables async DMA
    // to GPU, freeing the CPU thread while the transfer completes.
    {
        torch::NoGradGuard no_grad;
        bool async_transfer = device_.is_cuda() && input_tensor.is_pinned();
        auto gpu_input  = input_tensor.to(device_, /*non_blocking=*/async_transfer);
        auto gpu_output = local_model->forward(gpu_input);
        result = gpu_output.to(torch::kCPU).contiguous();
    }

    // Extract results into output array
    const float* result_ptr = result.data_ptr<float>();
    for (int i = 0; i < batch_size; ++i) {
        output[i] = static_cast<double>(result_ptr[i]);
    }
}

void InferenceEngine::swap_model(std::shared_ptr<ProYamsNet> new_model) {
    std::shared_ptr<ProYamsNet> old_model;
    {
        std::lock_guard<std::mutex> lock(inference_mutex_);
        old_model = std::move(model_);
        model_ = std::move(new_model);
    }
    // old_model destroyed here, outside the critical section —
    // CUDA memory deallocation won't block batch_inference.
}

// ---------------------------------------------------------------------------
// clone_model
// ---------------------------------------------------------------------------

std::shared_ptr<ProYamsNet> clone_model(const ProYamsNet& source,
                                         torch::Device target_device) {
    auto clone = std::make_shared<ProYamsNet>(source.config());

    // Copy all named parameters by value (deep copy)
    torch::NoGradGuard no_grad;
    auto src_params = source.named_parameters();
    auto dst_params = clone->named_parameters();
    for (const auto& pair : src_params) {
        dst_params[pair.key()].data().copy_(pair.value().data());
    }

    // Copy all named buffers (BatchNorm stats etc — none in this arch, but safe)
    auto src_bufs = source.named_buffers();
    auto dst_bufs = clone->named_buffers();
    for (const auto& pair : src_bufs) {
        dst_bufs[pair.key()].data().copy_(pair.value().data());
    }

    clone->to(target_device);
    clone->eval();
    return clone;
}
