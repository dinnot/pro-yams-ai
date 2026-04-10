#include "training/resume.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "training/training_loop.h"

bool resume_from_checkpoint(TrainingLoop& loop, const std::string& dir) {
    namespace fs = std::filesystem;

    if (!fs::exists(dir)) return false;

    // Find all checkpoint step numbers (identified by .model files).
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

    if (steps.empty()) return false;

    // Use the highest (most recent) step.
    int best = *std::max_element(steps.begin(), steps.end());
    std::string stem = dir + "/checkpoint_step_" + std::to_string(best);

    // Restore model + optimizer + scalar state.
    int    step        = 0;
    double temperature = 1.0;
    double epsilon     = 0.0;
    loop.trainer().load_checkpoint(stem, step, temperature, epsilon);
    loop.set_training_step(step);
    loop.set_temperature(temperature);
    loop.set_epsilon(epsilon);

    // Restore replay buffer if the companion file exists.
    std::string buf_path = stem + ".buffer";
    if (fs::exists(buf_path)) {
        loop.replay_buffer().load(buf_path);
    }

    return true;
}
