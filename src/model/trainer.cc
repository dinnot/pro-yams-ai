#include "model/trainer.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>

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
}

double ModelTrainer::train_step(const float* states, const double* targets,
                                  int batch_size) {
    assert(batch_size > 0);
    model_->train();

    // Build input tensor — copy into an owned tensor before moving to device.
    // Using from_blob directly can cause autograd issues in some libtorch builds.
    auto state_tensor = torch::empty({batch_size, kTensorSize}, torch::kFloat32);
    std::memcpy(state_tensor.data_ptr<float>(), states,
                static_cast<size_t>(batch_size) * kTensorSize * sizeof(float));
    state_tensor = state_tensor.to(device_);

    // Build target tensor: convert doubles to float32, clamped to the valid range
    // for the chosen activation (tanh → [-1, 1], sigmoid → [0, 1]).
    const bool use_tanh = (config_.output_activation != "sigmoid");
    auto target_tensor = torch::empty({batch_size, 1}, torch::kFloat32);
    {
        float* tptr = target_tensor.data_ptr<float>();
        for (int i = 0; i < batch_size; ++i) {
            float val = static_cast<float>(targets[i]);
            if (std::isnan(val)) val = use_tanh ? 0.0f : 0.5f;
            tptr[i] = use_tanh ? std::max(-1.0f, std::min(1.0f, val))
                               : std::max( 0.0f, std::min(1.0f, val));
        }
    }
    target_tensor = target_tensor.to(device_);

    // Forward pass
    auto prediction = model_->forward(state_tensor);

    // Loss: configurable per loss_function setting.
    torch::Tensor loss;
    if (config_.loss_function == "mse") {
        loss = torch::mse_loss(prediction, target_tensor);
    } else {
        loss = torch::binary_cross_entropy(prediction, target_tensor);
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

    return loss.item<double>();
}

std::shared_ptr<ProYamsNet> ModelTrainer::clone_for_inference(
    torch::Device device) {
    return clone_model(*model_, device);
}

void ModelTrainer::save_checkpoint(const std::string& path,
                                    int training_step,
                                    double temperature,
                                    double epsilon) {
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
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not load optimizer state (" << e.what()
                  << ") — optimizer will start fresh.\n";
    }

    training_step_ = training_step;
}

void ModelTrainer::load_weights(const std::string& path) {
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

    // Sort ascending and delete oldest if over limit
    std::sort(steps.begin(), steps.end());
    while (static_cast<int>(steps.size()) > max_checkpoints) {
        int old_step = steps.front();
        steps.erase(steps.begin());
        std::string old_stem = dir + "/checkpoint_step_" + std::to_string(old_step);
        fs::remove(old_stem + ".model");
        fs::remove(old_stem + ".optimizer");
    }
}
