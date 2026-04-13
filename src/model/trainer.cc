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

    // Build target tensor: convert doubles to float32
    auto target_tensor = torch::empty({batch_size, 1}, torch::kFloat32);
    {
        float* tptr = target_tensor.data_ptr<float>();
        for (int i = 0; i < batch_size; ++i) {
            float val = static_cast<float>(targets[i]);
            if (std::isnan(val)) val = 0.5f; // Failsafe
            tptr[i] = std::max(0.0f, std::min(1.0f, val));
        }
    }
    target_tensor = target_tensor.to(device_);  // clone because from_blob doesn't own the data

    // Forward pass
    auto prediction = model_->forward(state_tensor);

    // BCE loss — sigmoid output with binary targets; avoids vanishing gradients
    // that MSE causes when the sigmoid is saturated.
    auto loss = torch::binary_cross_entropy(prediction, target_tensor);

    if (config_.debug_mode && training_step_ % 1000 == 0) {
        auto pred_cpu = prediction.to(torch::kCPU);
        auto targ_cpu = target_tensor.to(torch::kCPU);
        
        double sum_win = 0.0, sum_loss = 0.0;
        int count_win = 0, count_loss = 0;
        
        const float* p_ptr = pred_cpu.data_ptr<float>();
        const float* t_ptr = targ_cpu.data_ptr<float>();
        
        for (int i = 0; i < batch_size; ++i) {
            if (t_ptr[i] > 0.5f) { sum_win += p_ptr[i]; count_win++; }
            else                 { sum_loss += p_ptr[i]; count_loss++; }
        }
        
        double avg_win = count_win > 0 ? sum_win / count_win : 0.0;
        double avg_loss = count_loss > 0 ? sum_loss / count_loss : 0.0;

        std::stringstream ss;
        ss << "\n--- Debug Batch @ Step " << training_step_ << " ---\n"
           << "BCE Loss:        " << loss.item<double>() << "\n"
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

    model_archive.save_to(path + ".model");

    // --- Save optimizer state separately ---
    torch::save(*optimizer_, path + ".optimizer");

    training_step_ = training_step;
}

void ModelTrainer::load_checkpoint(const std::string& path,
                                    int& training_step,
                                    double& temperature,
                                    double& epsilon) {
    // --- Load model weights + metadata ---
    torch::serialize::InputArchive model_archive;
    model_archive.load_from(path + ".model");
    model_->load(model_archive);
    model_->to(device_);

    // Read metadata — use separate tensor vars to avoid dtype reuse issues
    {
        torch::Tensor t;
        model_archive.read("training_step", t);
        training_step = static_cast<int>(t.item<int64_t>());
    }
    {
        torch::Tensor t;
        model_archive.read("temperature", t);
        temperature = t.item<double>();
    }
    {
        torch::Tensor t;
        model_archive.read("epsilon", t);
        epsilon = t.item<double>();
    }

    // Verify architecture
    {
        torch::Tensor t;
        model_archive.read("hidden_layers", t);
        assert(static_cast<int>(t.item<int64_t>()) == config_.hidden_layers);
    }
    {
        torch::Tensor t;
        model_archive.read("hidden_width", t);
        assert(static_cast<int>(t.item<int64_t>()) == config_.hidden_width);
    }

    // --- Load optimizer state ---
    torch::load(*optimizer_, path + ".optimizer");

    training_step_ = training_step;
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
