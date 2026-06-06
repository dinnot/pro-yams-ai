#include "distil/teacher_nn.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "engine/game_traits.h"
#include "engine/tensor.h"   // kTensorVersionLatest, tensor_size_for_version

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
    // Append-only versioning: the teacher may consume an OLDER tensor layout
    // than the student (its features are a byte-exact prefix of the student's).
    // Require (a) the checkpoint is self-consistent and (b) its layout is an
    // ancestor of this build's latest, so a prefix read is well-defined.
    input_size_     = teacher_cfg.input_size;
    tensor_version_ = teacher_cfg.tensor_version;
    const int expected_for_version = tensor_size_for_version<Traits>(tensor_version_);
    if (input_size_ != expected_for_version) {
        throw std::runtime_error(
            "NNTeacher: checkpoint input_size (" + std::to_string(input_size_) +
            ") is inconsistent with its tensor_version (" +
            std::to_string(tensor_version_) + ", expected " +
            std::to_string(expected_for_version) + "): " + checkpoint_stem);
    }
    if (tensor_version_ > kTensorVersionLatest ||
        input_size_ > Traits::kTensorSize) {
        throw std::runtime_error(
            "NNTeacher: checkpoint tensor_version (" +
            std::to_string(tensor_version_) +
            ") is newer than this build's latest (" +
            std::to_string(kTensorVersionLatest) +
            ") — cannot prefix-read a future layout: " + checkpoint_stem);
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
    const float* tensors, int tensor_stride,
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

    // The worker hands us the student's (latest) tensor at `tensor_stride`
    // floats/row. When the teacher consumes an older, narrower layout we copy
    // each row's leading `input_size_` columns into a contiguous packed buffer
    // (append-only versioning makes that prefix exactly the teacher's layout).
    // The buffer is thread_local because multiple workers call evaluate()
    // concurrently before serialising inside batch_inference.
    if (tensor_stride == input_size_) {
        inference_->batch_inference(tensors, n, dst);
    } else {
        thread_local std::vector<float> packed;
        const size_t need = static_cast<size_t>(n) * input_size_;
        if (packed.size() < need) packed.resize(need);
        for (int i = 0; i < n; ++i) {
            std::memcpy(packed.data() + static_cast<size_t>(i) * input_size_,
                        tensors + static_cast<size_t>(i) * tensor_stride,
                        static_cast<size_t>(input_size_) * sizeof(float));
        }
        inference_->batch_inference(packed.data(), n, dst);
    }
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
