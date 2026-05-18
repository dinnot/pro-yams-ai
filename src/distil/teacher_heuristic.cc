#include "distil/teacher_heuristic.h"

#include <algorithm>
#include <cmath>

namespace {

// Dispatch by HeuristicVersion onto the right heuristic_evaluate_* call.
// Mirrors worker.cc's branching (kept in sync — see notes in distil_worker
// when that lands). `out_is_odds_bot` reports whether the chosen evaluator
// produces win-odds (V4..V17 with output_win_odds set) so the caller can
// pick the right normalisation branch.
template <typename Traits>
void evaluate_raw(HeuristicVersion v,
                  const BoardStateT<Traits>& board,
                  const GameContextT<Traits>& ctx,
                  const AfterstateRequest* requests, int n,
                  const PrecomputedTables& tables,
                  double* out,
                  bool& out_is_odds_bot) {
    out_is_odds_bot = false;
    if (static_cast<int>(v) >= static_cast<int>(HeuristicVersion::V4)) {
        const ResearchConfig& cfg = get_research_config_for(v);
        out_is_odds_bot = cfg.output_win_odds;
        heuristic_evaluate_research<Traits>(board, ctx, requests, n,
                                            out, tables, cfg);
    } else if (v == HeuristicVersion::V3) {
        heuristic_evaluate_v3<Traits>(board, ctx, requests, n, out, tables);
    } else if (v == HeuristicVersion::V2) {
        heuristic_evaluate_v2<Traits>(board, ctx, requests, n, out, tables);
    } else {
        heuristic_evaluate<Traits>(board, ctx, requests, n, out);
    }
}

}  // namespace

template <typename Traits>
HeuristicTeacher<Traits>::HeuristicTeacher(
    HeuristicVersion version,
    const PrecomputedTables& tables,
    bool   use_duel_margin_maximization,
    double duel_margin_maximization_scale)
    : version_(version),
      tables_(tables),
      use_margin_(use_duel_margin_maximization),
      margin_scale_(duel_margin_maximization_scale) {}

template <typename Traits>
void HeuristicTeacher<Traits>::evaluate(
    const BoardStateT<Traits>& board,
    const GameContextT<Traits>& ctx,
    const AfterstateRequest* requests, int n,
    const float* /*tensors*/,
    double* targets,
    double* solver_evs) {
    bool is_odds_bot = false;
    // Write the RAW heuristic output directly into solver_evs. solver_resolve
    // averages these via expectimax (E[V0]) — averaging raw values is unbiased,
    // averaging tanh-squashed values penalises variance and cripples policy.
    evaluate_raw<Traits>(version_, board, ctx, requests, n, tables_,
                         solver_evs, is_odds_bot);

    if (targets == nullptr) return;

    // Now squash the raw evs into the student's target shape. The four
    // normalisation cases match worker.cc's blending block exactly.
    const bool is_margin_style =
        (version_ != HeuristicVersion::V1 && !is_odds_bot);

    for (int i = 0; i < n; ++i) {
        const double raw = solver_evs[i];
        double t;
        if (is_odds_bot) {
            double p = std::max(0.0, std::min(1.0, raw));
            t = use_margin_ ? (p * 2.0 - 1.0) : p;
        } else if (is_margin_style) {
            double squashed = std::tanh(raw / margin_scale_);
            t = use_margin_ ? squashed : (squashed + 1.0) / 2.0;
        } else if (use_margin_) {
            // V1, margin mode: tanh-squash the unbounded raw value.
            t = std::tanh(raw / margin_scale_);
        } else {
            // V1, win-prob mode: legacy /1800 clamp from worker.cc.
            t = std::max(0.0, std::min(1.0, raw / 1800.0));
        }
        targets[i] = t;
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template class HeuristicTeacher<Yams1v1>;
template class HeuristicTeacher<Yams2v2>;
