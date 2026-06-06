#include "model/trainer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>

#include <c10/cuda/CUDAGuard.h>

#include "engine/tensor.h"  // kTensorSize

// ---------------------------------------------------------------------------
// ModelTrainer
// ---------------------------------------------------------------------------

ModelTrainer::ModelTrainer(const ModelConfig& config, torch::Device device)
    : device_(device), config_(config) {
    model_ = std::make_shared<ProYamsNet>(config);
    initialize_weights(*model_);
    model_->to(device_);
    model_->train();

    optimizer_ = std::make_unique<torch::optim::Adam>(
        model_->parameters(),
        torch::optim::AdamOptions(config.learning_rate)
    );
    learning_rate_ = config.learning_rate;

    if (device_.is_cuda()) {
        train_stream_ = c10::cuda::getStreamFromPool(
            /*isHighPriority=*/true, device_.index());
    }
}

double ModelTrainer::train_step(const float* states, const double* targets,
                                  int batch_size) {
    assert(batch_size > 0);
    model_->train();

    std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
    if (train_stream_.has_value()) {
        stream_guard.emplace(*train_stream_);
    }

    // ---- Step 1: materialise the PREVIOUS step's loss. -------------------
    // .item() issues a D2H copy on train_stream_ and then synchronises the
    // stream — by the time it returns, every op submitted by the previous
    // train_step (H2D, forward, backward, optimizer.step) is complete. So
    // the pinned buffer below is safe to overwrite, and any model state we
    // touched has been committed.
    if (has_pending_loss_) {
        last_finalized_loss_ = pending_loss_tensor_.item<double>();
        pending_loss_tensor_ = torch::Tensor();
        has_pending_loss_    = false;
    }

    // Stage input + target tensors in PINNED CPU memory so the subsequent
    // .to(device, non_blocking=true) transfers can overlap with the next
    // train step's CPU work (and with the queued GPU compute that depends
    // on them). Buffers are allocated lazily and grown if a larger batch
    // arrives. Without pinning, .to(device) is implicitly synchronous and
    // every train step blocks ~5-10ms on the CPU→GPU transfer for big
    // batches (16K × 2126 floats ≈ 139 MB).
    if (batch_size > pinned_capacity_) {
        pinned_state_buffer_ = torch::empty(
            {batch_size, config_.input_size},
            torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
        pinned_target_buffer_ = torch::empty(
            {batch_size, 1},
            torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
        pinned_capacity_ = batch_size;
    }
    auto state_view  = pinned_state_buffer_.narrow(0, 0, batch_size);
    auto target_view = pinned_target_buffer_.narrow(0, 0, batch_size);

    std::memcpy(state_view.data_ptr<float>(), states,
                static_cast<size_t>(batch_size) * config_.input_size * sizeof(float));

    const bool use_tanh = (config_.output_activation != "sigmoid");
    {
        float* tptr = target_view.data_ptr<float>();
        for (int i = 0; i < batch_size; ++i) {
            float val = static_cast<float>(targets[i]);
            if (std::isnan(val)) val = use_tanh ? 0.0f : 0.5f;
            tptr[i] = use_tanh ? std::max(-1.0f, std::min(1.0f, val))
                               : std::max( 0.0f, std::min(1.0f, val));
        }
    }

    auto state_tensor  = state_view.to(device_,  /*non_blocking=*/true);
    auto target_tensor = target_view.to(device_, /*non_blocking=*/true);

    // Forward pass + loss.
    // For BCE we route through forward_logits so the loss can use the
    // numerically-stable binary_cross_entropy_with_logits, which folds the
    // sigmoid into a log-sum-exp and stops gradient blow-up when the
    // sigmoid saturates near 0 or 1. The debug block below still wants
    // post-sigmoid probabilities, so we recompute them on the fly when
    // we took the logits path.
    const bool use_bce = (config_.loss_function != "mse");
    torch::Tensor prediction;
    torch::Tensor loss;
    if (use_bce) {
        auto logits = model_->forward_logits(state_tensor);
        loss = torch::binary_cross_entropy_with_logits(logits, target_tensor);
        if (config_.debug_mode && training_step_ % 1000 == 0) {
            prediction = torch::sigmoid(logits);
        }
    } else {
        prediction = model_->forward(state_tensor);
        loss = torch::mse_loss(prediction, target_tensor);
    }

    if (config_.debug_mode && training_step_ % 1000 == 0) {
        auto pred_cpu = prediction.to(torch::kCPU);
        auto targ_cpu = target_tensor.to(torch::kCPU);
        
        double sum_win = 0.0, sum_loss = 0.0;
        int count_win = 0, count_loss = 0;
        
        const float* p_ptr = pred_cpu.data_ptr<float>();
        const float* t_ptr = targ_cpu.data_ptr<float>();
        
        const float win_threshold = use_tanh ? 0.0f : 0.5f;
        for (int i = 0; i < batch_size; ++i) {
            if (p_ptr[i] > win_threshold) { sum_win += p_ptr[i]; count_win++; }
            else                          { sum_loss += p_ptr[i]; count_loss++; }
        }

        double avg_win = count_win > 0 ? sum_win / count_win : 0.0;
        double avg_loss = count_loss > 0 ? sum_loss / count_loss : 0.0;

        std::stringstream ss;
        ss << "\n--- Debug Batch @ Step " << training_step_ << " ---\n"
           << config_.loss_function << " Loss: " << loss.item<double>() << "\n"
           << "Avg Pred (Win):  " << avg_win << " (" << count_win << " samples)\n"
           << "Avg Pred (Loss): " << avg_loss << " (" << count_loss << " samples)\n";

        if (!config_.debug_log_path.empty()) {
            std::ofstream f(config_.debug_log_path, std::ios::app);
            if (f.is_open()) f << ss.str();
        } else {
            std::cout << ss.str();
        }
    }

    // Backward + update
    optimizer_->zero_grad();
    loss.backward();
    optimizer_->step();

    ++training_step_;

    // Defer the .item() — store the loss tensor and let the next train_step
    // (or wait_until_step_complete) materialise it. Detach so we don't hold
    // the autograd graph alive across calls.
    pending_loss_tensor_ = loss.detach();
    has_pending_loss_    = true;

    return last_finalized_loss_;
}

void ModelTrainer::set_learning_rate(double lr) {
    for (auto& group : optimizer_->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
    }
    learning_rate_ = lr;
}

double ModelTrainer::wait_until_step_complete() {
    if (has_pending_loss_) {
        std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
        if (train_stream_.has_value()) {
            stream_guard.emplace(*train_stream_);
        }
        last_finalized_loss_ = pending_loss_tensor_.item<double>();
        pending_loss_tensor_ = torch::Tensor();
        has_pending_loss_    = false;
    }
    return last_finalized_loss_;
}

std::shared_ptr<ProYamsNet> ModelTrainer::clone_for_inference(
    torch::Device device) {
    // The clone reads model parameters, which the deferred optimizer.step()
    // from the most recent train_step may still be writing to. Sync first.
    wait_until_step_complete();
    return clone_model(*model_, device);
}

void ModelTrainer::save_checkpoint(const std::string& path,
                                    int training_step,
                                    double temperature,
                                    double epsilon) {
    // Drain any in-flight train step so the serialised parameters + optimizer
    // state reflect the most recent update (and not a half-applied step).
    wait_until_step_complete();

    // --- Save model weights + metadata ---
    torch::serialize::OutputArchive model_archive;
    model_->save(model_archive);

    // Metadata tensors
    model_archive.write("training_step",
                         torch::tensor(static_cast<int64_t>(training_step)));
    model_archive.write("temperature",
                         torch::tensor(temperature));
    model_archive.write("epsilon",
                         torch::tensor(epsilon));
    model_archive.write("hidden_layers",
                         torch::tensor(static_cast<int64_t>(config_.hidden_layers)));
    model_archive.write("hidden_width",
                         torch::tensor(static_cast<int64_t>(config_.hidden_width)));
    model_archive.write("input_size",
                         torch::tensor(static_cast<int64_t>(config_.input_size)));
    // game_variant: 1 = Yams1v1, 2 = Yams2v2. Pinned so 1v1 weights cannot be
    // silently loaded into a 2v2 process.
    model_archive.write("game_variant",
                         torch::tensor(static_cast<int64_t>(config_.game_variant)));
    // tensor_version: which append-only tensor layout this model consumes.
    // Absent in pre-Group-G checkpoints → readers default to 1.
    model_archive.write("tensor_version",
                         torch::tensor(static_cast<int64_t>(config_.tensor_version)));
    {
        const auto& arch = config_.architecture;
        auto t = torch::from_blob(const_cast<char*>(arch.data()),
                                   {static_cast<long>(arch.size())},
                                   torch::kUInt8).clone();
        model_archive.write("architecture", t);
    }
    {
        const auto& act = config_.output_activation;
        auto t = torch::from_blob(const_cast<char*>(act.data()),
                                   {static_cast<long>(act.size())},
                                   torch::kUInt8).clone();
        model_archive.write("output_activation", t);
    }

    model_archive.save_to(path + ".model");

    // --- Save optimizer state separately ---
    torch::save(*optimizer_, path + ".optimizer");

    training_step_ = training_step;
}

// ---------------------------------------------------------------------------
// load_model_weights_from_archive — shared helper used by load_checkpoint and
// load_weights.  Tries current-format first, falls back to legacy MLP format
// (pre-hidden_blocks refactor: hidden_0, hidden_1, … → hidden_blocks.0, …).
// ---------------------------------------------------------------------------
static void load_model_weights_from_archive(torch::serialize::InputArchive& archive,
                                             ProYamsNet& model,
                                             const std::string& path,
                                             int ckpt_hidden_layers) {
    bool loaded = false;
    try {
        model.load(archive);
        loaded = true;
    } catch (const c10::Error& first_err) {
        // Reload a fresh archive — the original may be partially consumed.
        torch::serialize::InputArchive legacy;
        legacy.load_from(path + ".model");
        try {
            torch::NoGradGuard no_grad;
            auto params = model.named_parameters();

            torch::serialize::InputArchive out_a;
            legacy.read("output", out_a);
            torch::Tensor w, b;
            out_a.read("weight", w); out_a.read("bias", b);
            params["output.weight"].data().copy_(w);
            params["output.bias"].data().copy_(b);

            for (int i = 0; i < ckpt_hidden_layers; ++i) {
                torch::serialize::InputArchive ha;
                legacy.read("hidden_" + std::to_string(i), ha);
                torch::Tensor hw, hb;
                ha.read("weight", hw); ha.read("bias", hb);
                std::string key = "hidden_blocks." + std::to_string(i * 2);
                params[key + ".weight"].data().copy_(hw);
                params[key + ".bias"].data().copy_(hb);
            }
            loaded = true;
            std::cerr << "Note: loaded checkpoint using legacy MLP format migration.\n";
        } catch (const std::exception& leg_err) {
            std::cerr << "Legacy load also failed: " << leg_err.what() << "\n";
            throw first_err;
        }
    }
    (void)loaded;
}

void ModelTrainer::load_checkpoint(const std::string& path,
                                    int& training_step,
                                    double& temperature,
                                    double& epsilon) {
    // Discard any pending step (its loss is meaningless once we overwrite
    // the weights anyway, and we don't want a future train_step to try to
    // .item() on a tensor whose stream has been blown away).
    wait_until_step_complete();

    // --- Load model weights + metadata ---
    torch::serialize::InputArchive model_archive;
    model_archive.load_from(path + ".model");

    // Read metadata with fallback defaults (old checkpoints may lack some keys).
    auto read_int64 = [&](const std::string& key, int64_t def) -> int64_t {
        try { torch::Tensor t; model_archive.read(key, t); return t.item<int64_t>(); }
        catch (...) { return def; }
    };
    auto read_dbl = [&](const std::string& key, double def) -> double {
        try { torch::Tensor t; model_archive.read(key, t); return t.item<double>(); }
        catch (...) { return def; }
    };

    training_step = static_cast<int>(read_int64("training_step", 0));
    temperature   = read_dbl("temperature", 1.0);
    epsilon       = read_dbl("epsilon", 0.0);
    int ckpt_hidden_layers = static_cast<int>(read_int64("hidden_layers", config_.hidden_layers));
    int ckpt_hidden_width  = static_cast<int>(read_int64("hidden_width",  config_.hidden_width));
    int ckpt_game_variant  = static_cast<int>(read_int64("game_variant",  kGameVariant1v1));
    int ckpt_tensor_version = static_cast<int>(read_int64("tensor_version", 1));

    // Fail fast on game-variant mismatch — a 1v1 checkpoint silently loaded
    // into a 2v2 process (or vice versa) would crash on the first forward pass
    // with a cryptic shape error. Refuse the load instead.
    if (ckpt_game_variant != config_.game_variant) {
        throw std::runtime_error(
            "Checkpoint game_variant (" + std::to_string(ckpt_game_variant) +
            ") does not match runtime game_variant (" +
            std::to_string(config_.game_variant) +
            "). 1=Yams1v1, 2=Yams2v2.");
    }
    // Same for the tensor layout version: resuming training requires the model
    // to consume the same features it was trained on. (Append-only versions also
    // differ in input_size, so the weight load would fail anyway — this just
    // produces a clearer error.)
    if (ckpt_tensor_version != config_.tensor_version) {
        throw std::runtime_error(
            "Checkpoint tensor_version (" + std::to_string(ckpt_tensor_version) +
            ") does not match runtime tensor_version (" +
            std::to_string(config_.tensor_version) + ").");
    }

    if (ckpt_hidden_layers != config_.hidden_layers || ckpt_hidden_width != config_.hidden_width) {
        std::cerr << "Warning: checkpoint architecture (" << ckpt_hidden_layers
                  << "x" << ckpt_hidden_width << ") differs from model config ("
                  << config_.hidden_layers << "x" << config_.hidden_width
                  << ") — load may fail.\n";
    }

    load_model_weights_from_archive(model_archive, *model_, path, ckpt_hidden_layers);
    model_->to(device_);

    // --- Load optimizer state (best-effort; skip if incompatible) ---
    try {
        torch::load(*optimizer_, path + ".optimizer");
        // The serialised optimizer may carry a learning rate that drifted from
        // config.learning_rate (e.g. after LR back-off). Resync the tracked
        // value so learning_rate() / set_learning_rate() stay consistent.
        if (!optimizer_->param_groups().empty()) {
            learning_rate_ = static_cast<torch::optim::AdamOptions&>(
                optimizer_->param_groups().front().options()).lr();
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not load optimizer state (" << e.what()
                  << ") — optimizer will start fresh.\n";
    }

    training_step_ = training_step;
}

void ModelTrainer::load_weights(const std::string& path) {
    wait_until_step_complete();
    torch::serialize::InputArchive archive;
    archive.load_from(path + ".model");

    auto read_int64 = [&](const std::string& key, int64_t def) -> int64_t {
        try { torch::Tensor t; archive.read(key, t); return t.item<int64_t>(); }
        catch (...) { return def; }
    };
    int ckpt_hidden_layers = static_cast<int>(
        read_int64("hidden_layers", config_.hidden_layers));

    load_model_weights_from_archive(archive, *model_, path, ckpt_hidden_layers);
    model_->to(device_);
}

ModelConfig ModelTrainer::config_from_checkpoint(const std::string& path) {
    ModelConfig cfg;  // sensible defaults

    torch::serialize::InputArchive archive;
    archive.load_from(path + ".model");

    auto read_int64 = [&](const std::string& key, int64_t def) -> int64_t {
        try { torch::Tensor t; archive.read(key, t); return t.item<int64_t>(); }
        catch (...) { return def; }
    };

    cfg.hidden_layers = static_cast<int>(read_int64("hidden_layers", 3));
    cfg.hidden_width  = static_cast<int>(read_int64("hidden_width",  256));
    cfg.input_size    = static_cast<int>(read_int64("input_size",    cfg.input_size));
    // game_variant: older checkpoints predate the tag — default to 1v1.
    cfg.game_variant  = static_cast<int>(read_int64("game_variant",  kGameVariant1v1));
    // tensor_version: older checkpoints predate Group G — default to version 1.
    cfg.tensor_version = static_cast<int>(read_int64("tensor_version", 1));

    // Old checkpoints pre-date the resnet refactor — assume mlp.
    // New checkpoints save "architecture" explicitly.
    try {
        torch::Tensor t;
        archive.read("architecture", t);
        cfg.architecture = std::string(reinterpret_cast<const char*>(t.data_ptr()),
                                       t.numel());
    } catch (...) {
        cfg.architecture = "mlp";
    }

    // Old checkpoints pre-date saving output_activation — assume tanh (the
    // historical default).  Newer checkpoints save it explicitly.
    try {
        torch::Tensor t;
        archive.read("output_activation", t);
        cfg.output_activation = std::string(reinterpret_cast<const char*>(t.data_ptr()),
                                             t.numel());
    } catch (...) {
        cfg.output_activation = "sigmoid";
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// save_managed_checkpoint
// ---------------------------------------------------------------------------

void save_managed_checkpoint(ModelTrainer& trainer, const std::string& dir,
                               int step, double temperature, double epsilon,
                               int max_checkpoints) {
    namespace fs = std::filesystem;

    std::string stem = dir + "/checkpoint_step_" + std::to_string(step);
    trainer.save_checkpoint(stem, step, temperature, epsilon);

    // Collect existing checkpoints (by .model file)
    std::vector<int> steps;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        const std::string prefix = "checkpoint_step_";
        const std::string suffix = ".model";
        if (name.rfind(prefix, 0) == 0 &&
            name.size() > prefix.size() + suffix.size() &&
            name.substr(name.size() - suffix.size()) == suffix) {
            try {
                int s = std::stoi(name.substr(prefix.size(),
                                              name.size() - prefix.size() - suffix.size()));
                steps.push_back(s);
            } catch (...) {}
        }
    }

    // Sort ascending and, for steps beyond the limit, prune the heavyweight
    // optimizer/buffer files while keeping the .model itself permanently.
    std::sort(steps.begin(), steps.end());
    while (static_cast<int>(steps.size()) > max_checkpoints) {
        int old_step = steps.front();
        steps.erase(steps.begin());
        std::string old_stem = dir + "/checkpoint_step_" + std::to_string(old_step);
        fs::remove(old_stem + ".optimizer");
        fs::remove(old_stem + ".buffer");
    }
}
