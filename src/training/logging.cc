#include "training/logging.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// log_metrics
// ---------------------------------------------------------------------------

void log_metrics(const std::string& path, const TrainingMetrics& metrics) {
    namespace fs = std::filesystem;
    
    // Ensure parent directory exists
    auto parent = fs::path(path).parent_path();
    if (!parent.empty() && !fs::exists(parent)) {
        fs::create_directories(parent);
    }

    bool write_header = !fs::exists(path) || fs::file_size(path) == 0;

    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("log_metrics: cannot open " + path);

    if (write_header) {
        f << "step,games_played,buffer_size,loss,learning_rate,temperature,epsilon,gps,total_samples\n";
    }

    f << metrics.training_step     << ','
      << metrics.games_played      << ','
      << metrics.samples_in_buffer << ','
      << metrics.loss              << ','
      << metrics.learning_rate     << ','
      << metrics.temperature       << ','
      << metrics.epsilon           << ','
      << metrics.games_per_second   << ','
      << metrics.total_samples_trained << '\n';
}

// ---------------------------------------------------------------------------
// prune_old_checkpoints
// ---------------------------------------------------------------------------

void prune_old_checkpoints(const std::string& dir, int max_keep) {
    namespace fs = std::filesystem;

    if (!fs::exists(dir)) return;

    // Collect all step numbers that have a .model file.
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

    std::sort(steps.begin(), steps.end());

    // For steps beyond max_keep, prune the heavyweight buffer/optimizer files
    // but keep the .model itself permanently so the full checkpoint history of
    // weights is preserved.
    while (static_cast<int>(steps.size()) > max_keep) {
        int old_step = steps.front();
        steps.erase(steps.begin());
        std::string stem = dir + "/checkpoint_step_" + std::to_string(old_step);
        fs::remove(stem + ".optimizer");
        fs::remove(stem + ".buffer");
    }
}
