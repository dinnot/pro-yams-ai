#pragma once

#include <cmath>

// ---------------------------------------------------------------------------
// Pure convergence helpers for the distillation loop.
//
// Extracted as free inline functions so the logic can be unit-tested without
// constructing a full DistilLoopT (which requires a precomputed-tables build
// and libtorch).
//
// The three primitives compose to drive the loop's stop decision:
//
//   bool pass        = within_win_rate_delta(student_wr, teacher_wr, delta);
//   patience         = next_patience(patience, pass);
//   bool stop_now    = convergence_satisfied(step, min_steps,
//                                            patience, patience_target);
// ---------------------------------------------------------------------------

namespace distil {

/// True iff |student_wr - teacher_wr| < delta. A non-positive `delta` makes
/// the predicate trivially false, which is the sentinel for "disable this
/// criterion": `delta = -1.0` means the loop runs to `max_steps`.
inline bool within_win_rate_delta(double student_wr,
                                  double teacher_wr,
                                  double delta) {
    if (delta <= 0.0) return false;
    return std::abs(student_wr - teacher_wr) < delta;
}

/// Increment patience on a pass, reset to 0 on a fail. The contract is "N
/// consecutive eval checks within the delta" — a single noisy bad reading
/// resets the count.
inline int next_patience(int current_patience, bool current_check_passed) {
    return current_check_passed ? current_patience + 1 : 0;
}

/// Final stop predicate. Two gates: (1) we've trained at least `min_steps`
/// to avoid early lucky reads; (2) we've had `patience_target` consecutive
/// passing eval checks.
///
/// Setting `patience_target <= 0` would make the second gate trivially
/// pass — callers should validate it's positive (the config validator does).
inline bool convergence_satisfied(int training_step,
                                  int min_steps,
                                  int consecutive_passes,
                                  int patience_target) {
    if (training_step < min_steps) return false;
    return consecutive_passes >= patience_target;
}

// ---------------------------------------------------------------------------
// Optional secondary criterion: rolling match-MSE (EMA of training loss).
//
// The student's per-batch training loss IS the MSE between its outputs and
// the teacher's targets on that batch (each sample is one (state, teacher_ev)
// pair). So a low rolling EMA of train_loss means "the student is matching
// the teacher's outputs closely on the chunks it has seen" — a useful early
// signal complementing the win-rate-vs-reference criterion.
//
// The wr criterion catches "student plays as well as teacher". The mse
// criterion catches "student reproduces teacher's value function". Either
// passing for the configured patience triggers convergence.
// ---------------------------------------------------------------------------

struct EmaState {
    double value       = 0.0;
    bool   initialized = false;
};

/// Lazily-initialised EMA. First sample seeds the EMA; later samples are
/// blended with weight `alpha` on the new observation. alpha ~ 0.05 gives
/// a ~20-batch half-life, smoothing out per-batch noise without lagging
/// too far behind real changes.
inline void update_ema(EmaState& s, double new_value, double alpha) {
    if (!s.initialized) {
        s.value       = new_value;
        s.initialized = true;
        return;
    }
    s.value = alpha * new_value + (1.0 - alpha) * s.value;
}

/// True iff the EMA has been seeded AND its smoothed value is below the
/// threshold. Non-positive `threshold` is the "disabled" sentinel — matches
/// the convention used by `within_win_rate_delta`.
inline bool within_match_mse(const EmaState& s, double threshold) {
    if (threshold <= 0.0) return false;
    if (!s.initialized)   return false;
    return s.value < threshold;
}

}  // namespace distil
