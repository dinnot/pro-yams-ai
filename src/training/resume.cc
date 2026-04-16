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

    // Restore model + optimizer + scalar state.
    int    step        = 0;
    double temperature = 1.0;
    double epsilon     = 0.0;
    loop.trainer().load_checkpoint(stem, step, temperature, epsilon);
    loop.set_training_step(step);
    loop.set_temperature(temperature);
    loop.set_epsilon(epsilon);

    // Restore replay buffer if the companion file exists.
    namespace fs = std::filesystem;
    std::string buf_path = stem + ".buffer";
    if (fs::exists(buf_path)) {
        loop.replay_buffer().load(buf_path);
    }

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
