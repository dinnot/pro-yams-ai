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

template <typename Traits>
bool resume_from_checkpoint(TrainingLoopT<Traits>& loop, const std::string& dir) {
    std::string stem = find_latest_checkpoint_stem(dir);
    if (stem.empty()) return false;

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

template <typename Traits>
bool init_from_checkpoint(TrainingLoopT<Traits>& loop, const std::string& path) {
    namespace fs = std::filesystem;

    std::string stem;
    if (fs::is_directory(path)) {
        stem = find_latest_checkpoint_stem(path);
        if (stem.empty()) return false;
    } else {
        if (!fs::exists(path + ".model")) return false;
        stem = path;
    }

    loop.trainer().load_weights(stem);
    return true;
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template bool resume_from_checkpoint<Yams1v1>(TrainingLoopT<Yams1v1>&, const std::string&);
template bool resume_from_checkpoint<Yams2v2>(TrainingLoopT<Yams2v2>&, const std::string&);
template bool init_from_checkpoint<Yams1v1>(TrainingLoopT<Yams1v1>&, const std::string&);
template bool init_from_checkpoint<Yams2v2>(TrainingLoopT<Yams2v2>&, const std::string&);
