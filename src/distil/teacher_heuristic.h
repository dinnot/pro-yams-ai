#pragma once

#include "distil/teacher.h"
#include "heuristic/heuristic_bot.h"
#include "solver/precomputed_tables.h"

// ---------------------------------------------------------------------------
// HeuristicTeacher<Traits> — distillation teacher backed by one of the
// V1..V17 heuristic evaluators.
//
// Reproduces worker.cc's heuristic-evaluation + normalisation pipeline, but
// without the NN blending step: the raw heuristic value is squashed once
// into the target shape (margin tanh or win-prob) and returned directly.
//
// The (use_margin, margin_scale) pair must match the student's
// `output_activation` / `duel_margin_maximization_scale` so the produced
// targets land in-range for the student's loss.
// ---------------------------------------------------------------------------
template <typename Traits>
class HeuristicTeacher : public Teacher<Traits> {
public:
    HeuristicTeacher(HeuristicVersion version,
                     const PrecomputedTables& tables,
                     bool   use_duel_margin_maximization,
                     double duel_margin_maximization_scale = 4000.0);

    void evaluate(const BoardStateT<Traits>& board,
                  const GameContextT<Traits>& ctx,
                  const AfterstateRequest* requests, int n,
                  const float* tensors,
                  double* evs) override;

    bool needs_tensor_input() const override { return false; }

    HeuristicVersion version() const { return version_; }

private:
    HeuristicVersion         version_;
    const PrecomputedTables& tables_;
    bool                     use_margin_;
    double                   margin_scale_;
};
