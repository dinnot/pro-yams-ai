#include "training/resume.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "training/training_loop.h"

// ---------------------------------------------------------------------------
// find_latest_checkpoint_stem
// ---------------------------------------------------------------------------

std::string find_latest_checkpoint_stem(const std::string& dir) {
    namespace fs = std::filesystem;

    if (!fs::exists(dir)) return {};

    std::vector<int> steps;
    const std::string prefix = "checkpoint_step_";
    const std::string suffix = ".model";

    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) != 0) continue;
        if (name.size() <= prefix.size() + suffix.size()) continue;
        if (name.substr(name.size() - suffix.size()) != suffix) continue;
        try {
            int s = std::stoi(name.substr(prefix.size(),
                                          name.size() - prefix.size() - suffix.size()));
            steps.push_back(s);
        } catch (...) {}
    }

    if (steps.empty()) return {};

    int best = *std::max_element(steps.begin(), steps.end());
    return dir + "/checkpoint_step_" + std::to_string(best);
}

// ---------------------------------------------------------------------------
// resume_from_checkpoint
// ---------------------------------------------------------------------------

bool resume_from_checkpoint(TrainingLoop& loop, const std::string& dir) {
    std::string stem = find_latest_checkpoint_stem(dir);
    if (stem.empty()) return false;

    // Full resume requires the optimizer state and the replay buffer to
    // exist alongside the model. Without the optimizer, Adam's moving
    // averages reset to zero and the first update is an unscaled gradient
    // step that can rip a converged network out of its local minimum.
    // Without the replay buffer, the run refills ~1M samples against the
    // just-loaded network before training resumes, distorting the on-policy
    // distribution. Both have been observed to cause severe regression.
    // Refuse to proceed and tell the user how to opt out explicitly.
    namespace fs = std::filesystem;
    if (!fs::exists(stem + ".optimizer")) {
        throw std::runtime_error(
            "Resume artifact missing: " + stem + ".optimizer\n"
            "Refusing to resume — an Adam reset on a converged network can cause "
            "severe regression. Use --checkpoint <dir> instead if you intend to "
            "start a fresh optimizer from these weights.");
    }
    if (!fs::exists(stem + ".buffer")) {
        throw std::runtime_error(
            "Resume artifact missing: " + stem + ".buffer\n"
            "Refusing to resume — without the replay buffer the run will refill "
            "samples against the just-loaded network before training resumes, "
            "distorting the on-policy distribution. Use --checkpoint <dir> "
            "instead if you intend to start a fresh buffer from these weights.");
    }

    // Restore model + optimizer + scalar state.
    int    step        = 0;
    double temperature = 1.0;
    double epsilon     = 0.0;
    loop.trainer().load_checkpoint(stem, step, temperature, epsilon);
    loop.set_training_step(step);
    loop.set_temperature(temperature);
    loop.set_epsilon(epsilon);

    loop.replay_buffer().load(stem + ".buffer");

    return true;
}

// ---------------------------------------------------------------------------
// init_from_checkpoint
// ---------------------------------------------------------------------------

bool init_from_checkpoint(TrainingLoop& loop, const std::string& path) {
    namespace fs = std::filesystem;

    std::string stem;
    if (fs::is_directory(path)) {
        // Directory — find the latest checkpoint inside it.
        stem = find_latest_checkpoint_stem(path);
        if (stem.empty()) return false;
    } else {
        // Assume it's a file stem (e.g. "checkpoints/checkpoint_step_5000").
        // Verify the .model file exists.
        if (!fs::exists(path + ".model")) return false;
        stem = path;
    }

    // Load only the model weights — no optimizer state, no step/temp/epsilon.
    loop.trainer().load_weights(stem);
    return true;
}
