#pragma once

#include "engine/game_context.h"
#include "engine/game_state.h"
#include "engine/game_traits.h"
#include "solver/solver.h"   // AfterstateRequest

// ---------------------------------------------------------------------------
// Teacher<Traits> — abstract interface for distillation oracles.
//
// In distill mode the teacher plays every seat. After solver_get_requests
// the distill worker calls evaluate() to score every afterstate the teacher
// could choose between. Those scores serve two purposes simultaneously:
//
//   1. They drive the teacher's action (passed into solver_resolve as evs).
//   2. They become the student's training targets — one (state, target)
//      sample per visited afterstate.
//
// Concrete implementations:
//   - HeuristicTeacher<Traits>   (synchronous CPU, no GPU)
//   - NNTeacher<Traits>           (batched GPU inference via InferenceEngine)
// ---------------------------------------------------------------------------
template <typename Traits>
class Teacher {
public:
    virtual ~Teacher() = default;

    /// Score every afterstate in `requests[0..n)` and write the result into
    /// `evs[0..n)`. The exact value semantics (win-prob vs. margin, scale)
    /// depend on the concrete teacher; they must match the student's
    /// `output_activation` so the supervised targets are in-range for the
    /// student's loss.
    ///
    /// @param tensors May be nullptr when needs_tensor_input() == false
    ///                (heuristic teachers do not consume tensors).
    virtual void evaluate(const BoardStateT<Traits>& board,
                          const GameContextT<Traits>& ctx,
                          const AfterstateRequest* requests, int n,
                          const float* tensors,
                          double* evs) = 0;

    /// True when the teacher requires the worker to materialise the
    /// per-request state tensors before calling evaluate(). NN teachers
    /// return true (the tensors are the model's input); heuristic teachers
    /// return false (the worker still generates tensors so it can emit
    /// distill samples — see distil_worker — but the teacher doesn't read them).
    virtual bool needs_tensor_input() const = 0;
};
