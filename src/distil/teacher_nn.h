#pragma once

#include <memory>
#include <string>

#include <torch/torch.h>

#include "distil/teacher.h"
#include "model/inference.h"
#include "model/pro_yams_net.h"
#include "model/trainer.h"

// ---------------------------------------------------------------------------
// NNTeacher<Traits> — distillation teacher backed by a frozen ProYamsNet
// loaded from a checkpoint.
//
// The teacher's architecture (hidden_layers, hidden_width, output_activation)
// is read from the checkpoint and may differ from the student's — that's the
// whole point of NN distillation (big ↔ small, deep ↔ wide, etc.).
// Constraints validated at construction:
//
//   - teacher's `game_variant` == Traits' variant
//   - teacher's `input_size`   == Traits::kTensorSize
//   - teacher's `output_activation` matches the run's target shape
//     (use_duel_margin_maximization=true ⇒ "tanh"; false ⇒ "sigmoid")
//
// evaluate() calls InferenceEngine::batch_inference synchronously. With N
// workers the calls serialise on the engine's internal mutex; this trades
// peak GPU batch size for implementation simplicity. A coordinator-thread
// variant (one big batch across all workers) is a profile-driven follow-up.
// ---------------------------------------------------------------------------
template <typename Traits>
class NNTeacher : public Teacher<Traits> {
public:
    NNTeacher(const std::string& checkpoint_stem,
              torch::Device device,
              bool use_duel_margin_maximization);

    void evaluate(const BoardStateT<Traits>& board,
                  const GameContextT<Traits>& ctx,
                  const AfterstateRequest* requests, int n,
                  const float* tensors,
                  double* targets,
                  double* solver_evs) override;

    bool needs_tensor_input() const override { return true; }

    // --- Accessors used by the eval helpers (run_evaluation_vs etc.). ---
    ProYamsNet&   model()  { return *teacher_model_; }
    torch::Device device() const { return device_; }

private:
    torch::Device                    device_;
    std::unique_ptr<ModelTrainer>    loader_;        // owns parameter storage
    std::shared_ptr<ProYamsNet>      teacher_model_;
    std::unique_ptr<InferenceEngine> inference_;
};
