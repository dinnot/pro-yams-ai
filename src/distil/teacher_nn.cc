#include "distil/teacher_nn.h"

#include <stdexcept>
#include <string>
#include <type_traits>

#include "engine/game_traits.h"

namespace {

template <typename Traits>
constexpr int expected_variant() {
    return std::is_same_v<Traits, Yams1v1> ? kGameVariant1v1
                                           : kGameVariant2v2;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <typename Traits>
NNTeacher<Traits>::NNTeacher(const std::string& checkpoint_stem,
                             torch::Device device,
                             bool use_duel_margin_maximization)
    : device_(device) {
    // Step 1: read the teacher's ModelConfig from the checkpoint metadata.
    ModelConfig teacher_cfg = ModelTrainer::config_from_checkpoint(checkpoint_stem);

    // Step 2: validate the teacher matches the run's Traits.
    if (teacher_cfg.game_variant != expected_variant<Traits>()) {
        throw std::runtime_error(
            "NNTeacher: checkpoint game_variant (" +
            std::to_string(teacher_cfg.game_variant) +
            ") does not match run's variant (" +
            std::to_string(expected_variant<Traits>()) + "): " + checkpoint_stem);
    }
    if (teacher_cfg.input_size != Traits::kTensorSize) {
        throw std::runtime_error(
            "NNTeacher: checkpoint input_size (" +
            std::to_string(teacher_cfg.input_size) +
            ") does not match Traits::kTensorSize (" +
            std::to_string(Traits::kTensorSize) + "): " + checkpoint_stem);
    }

    // Step 3: validate output activation matches the target shape so the
    // student trains against in-range targets. tanh ↔ margin mode,
    // sigmoid ↔ win-prob mode.
    const std::string expected_activation =
        use_duel_margin_maximization ? "tanh" : "sigmoid";
    if (teacher_cfg.output_activation != expected_activation) {
        throw std::runtime_error(
            "NNTeacher: checkpoint output_activation \"" +
            teacher_cfg.output_activation + "\" does not match the run's "
            "use_duel_margin_maximization setting (expected \"" +
            expected_activation + "\"): " + checkpoint_stem);
    }

    // Step 4: instantiate a trainer with the teacher's config, load weights
    // (without optimizer state — inference doesn't need it and the .optimizer
    // file may not be present).
    loader_ = std::make_unique<ModelTrainer>(teacher_cfg, device_);
    loader_->load_weights(checkpoint_stem);

    // Step 5: clone for inference and wrap in a thread-safe engine.
    teacher_model_ = loader_->clone_for_inference(device_);
    teacher_model_->eval();
    inference_ = std::make_unique<InferenceEngine>(teacher_model_, device_);
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

template <typename Traits>
void NNTeacher<Traits>::evaluate(
    const BoardStateT<Traits>& /*board*/,
    const GameContextT<Traits>& /*ctx*/,
    const AfterstateRequest* /*requests*/, int n,
    const float* tensors,
    double* targets,
    double* solver_evs) {
    if (n <= 0) return;
    // InferenceEngine::batch_inference is thread-safe (mutex inside);
    // multiple workers serialise here. Per-call batch ~30..150 — small
    // for GPU throughput; a coordinator-batched variant is a follow-up.
    //
    // For an NN teacher the prediction IS the action value: solver_resolve
    // selects actions to maximise expected predicted value, which is
    // self-consistent. So solver_evs and targets get the same numbers.
    double* dst = targets ? targets : solver_evs;
    inference_->batch_inference(tensors, n, dst);
    if (targets && solver_evs && targets != solver_evs) {
        for (int i = 0; i < n; ++i) solver_evs[i] = targets[i];
    } else if (!targets && solver_evs) {
        // dst == solver_evs already.
    } else if (targets && !solver_evs) {
        // Caller only wanted training targets; nothing to copy.
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template class NNTeacher<Yams1v1>;
template class NNTeacher<Yams2v2>;
