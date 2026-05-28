#include "distil/distil_resume.h"

#include <filesystem>
#include <stdexcept>

#include "distil/distil_loop.h"
#include "engine/game_traits.h"
#include "training/resume.h"   // find_latest_checkpoint_stem (re-used)

template <typename Traits>
bool resume_distil_from_checkpoint(DistilLoopT<Traits>& loop,
                                    const std::string& dir) {
    std::string stem = find_latest_checkpoint_stem(dir);
    if (stem.empty()) return false;

    namespace fs = std::filesystem;
    if (!fs::exists(stem + ".optimizer")) {
        throw std::runtime_error(
            "Resume artifact missing: " + stem + ".optimizer\n"
            "Refusing to resume — restarting Adam on already-trained weights "
            "can spike loss enough to undo prior progress. Use --checkpoint "
            "<stem> instead if you intend to continue from these weights "
            "with a fresh optimizer.");
    }

    int    step        = 0;
    double temperature = 0.0;   // unused by distil; required by signature
    double epsilon     = 0.0;
    loop.trainer().load_checkpoint(stem, step, temperature, epsilon);
    loop.set_training_step(step);

    // Transient state (consecutive_passes_, rolling_mse_) is intentionally
    // not restored — it rebuilds from a few eval / train_step iterations
    // and doesn't justify a schema change to the checkpoint format.

    return true;
}

template <typename Traits>
bool init_distil_from_checkpoint(DistilLoopT<Traits>& loop,
                                  const std::string& path) {
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
template bool resume_distil_from_checkpoint<Yams1v1>(DistilLoopT<Yams1v1>&, const std::string&);
template bool resume_distil_from_checkpoint<Yams2v2>(DistilLoopT<Yams2v2>&, const std::string&);
template bool init_distil_from_checkpoint<Yams1v1>(DistilLoopT<Yams1v1>&, const std::string&);
template bool init_distil_from_checkpoint<Yams2v2>(DistilLoopT<Yams2v2>&, const std::string&);
