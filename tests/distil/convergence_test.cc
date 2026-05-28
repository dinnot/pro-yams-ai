#include <gtest/gtest.h>

#include "distil/convergence.h"

using distil::convergence_satisfied;
using distil::EmaState;
using distil::next_patience;
using distil::update_ema;
using distil::within_match_mse;
using distil::within_win_rate_delta;

// ---------------------------------------------------------------------------
// within_win_rate_delta
// ---------------------------------------------------------------------------

TEST(WithinWinRateDeltaTest, PositiveDelta_TruthTable) {
    // Gap = 0.01, delta = 0.02 → within
    EXPECT_TRUE (within_win_rate_delta(0.50, 0.51, 0.02));
    // Gap = 0.02, delta = 0.02 → strict less-than, NOT within
    EXPECT_FALSE(within_win_rate_delta(0.50, 0.52, 0.02));
    // Gap = 0.03, delta = 0.02 → outside
    EXPECT_FALSE(within_win_rate_delta(0.50, 0.53, 0.02));
}

TEST(WithinWinRateDeltaTest, SignDoesNotMatter) {
    // |student - teacher| is symmetric; whether student leads or trails
    // doesn't change the pass/fail.
    EXPECT_TRUE (within_win_rate_delta(0.50, 0.51, 0.02));
    EXPECT_TRUE (within_win_rate_delta(0.51, 0.50, 0.02));
    EXPECT_FALSE(within_win_rate_delta(0.50, 0.55, 0.02));
    EXPECT_FALSE(within_win_rate_delta(0.55, 0.50, 0.02));
}

TEST(WithinWinRateDeltaTest, ZeroDeltaIsDisabledSentinel) {
    // delta = 0 disables — any gap (even exactly zero) returns false.
    EXPECT_FALSE(within_win_rate_delta(0.50, 0.50, 0.0));
    EXPECT_FALSE(within_win_rate_delta(0.50, 0.55, 0.0));
}

TEST(WithinWinRateDeltaTest, NegativeDeltaIsDisabledSentinel) {
    // Plan-prescribed sentinel for "run to max_steps regardless of gap".
    EXPECT_FALSE(within_win_rate_delta(0.50, 0.50, -1.0));
    EXPECT_FALSE(within_win_rate_delta(0.50, 0.51, -0.01));
}

// ---------------------------------------------------------------------------
// next_patience
// ---------------------------------------------------------------------------

TEST(NextPatienceTest, IncrementsOnPass) {
    EXPECT_EQ(next_patience(0, true), 1);
    EXPECT_EQ(next_patience(1, true), 2);
    EXPECT_EQ(next_patience(99, true), 100);
}

TEST(NextPatienceTest, ResetsOnFail) {
    EXPECT_EQ(next_patience(0, false),  0);
    EXPECT_EQ(next_patience(1, false),  0);
    EXPECT_EQ(next_patience(99, false), 0);
}

TEST(NextPatienceTest, SequenceModelsConsecutiveStreak) {
    // Pass, pass, fail, pass, pass, pass — final patience should be 3.
    int p = 0;
    p = next_patience(p, true);   // 1
    p = next_patience(p, true);   // 2
    p = next_patience(p, false);  // 0
    p = next_patience(p, true);   // 1
    p = next_patience(p, true);   // 2
    p = next_patience(p, true);   // 3
    EXPECT_EQ(p, 3);
}

// ---------------------------------------------------------------------------
// convergence_satisfied
// ---------------------------------------------------------------------------

TEST(ConvergenceSatisfiedTest, MinStepsGate_BlocksEarlyConvergence) {
    // Even with patience overflowing, min_steps not reached → don't stop.
    EXPECT_FALSE(convergence_satisfied(/*step=*/100,
                                       /*min_steps=*/1000,
                                       /*passes=*/10,
                                       /*target=*/3));
}

TEST(ConvergenceSatisfiedTest, MinStepsGate_BoundaryInclusive) {
    // step == min_steps is enough (the loop has trained min_steps batches).
    EXPECT_TRUE(convergence_satisfied(1000, 1000, /*passes=*/3, /*target=*/3));
}

TEST(ConvergenceSatisfiedTest, PatienceShortByOne_NotConverged) {
    EXPECT_FALSE(convergence_satisfied(/*step=*/5000, /*min_steps=*/1000,
                                       /*passes=*/2, /*target=*/3));
}

TEST(ConvergenceSatisfiedTest, PatienceMet_Converged) {
    EXPECT_TRUE(convergence_satisfied(/*step=*/5000, /*min_steps=*/1000,
                                      /*passes=*/3, /*target=*/3));
    // Excess patience is also converged.
    EXPECT_TRUE(convergence_satisfied(/*step=*/5000, /*min_steps=*/1000,
                                      /*passes=*/10, /*target=*/3));
}

TEST(ConvergenceSatisfiedTest, ZeroPatienceTarget_ShouldNotBeReachable) {
    // The validator rejects patience_target <= 0, so this is a defensive
    // check only — but if somehow target=0, ANY post-min-steps state would
    // count as converged. Document the actual behaviour so it doesn't
    // silently change.
    EXPECT_TRUE(convergence_satisfied(/*step=*/1000, /*min_steps=*/1000,
                                      /*passes=*/0, /*target=*/0));
}

// ---------------------------------------------------------------------------
// End-to-end composition: simulate the loop's update sequence.
// ---------------------------------------------------------------------------

TEST(ConvergenceCompositionTest, SimulatedRun_ReachesConvergenceAfterPatience) {
    const int    min_steps  = 1000;
    const int    target     = 3;
    const double delta      = 0.02;
    const double teacher_wr = 0.50;

    int patience = 0;
    int step     = 0;

    // Pretend we eval every 500 steps with a sequence of student wr readings.
    const double readings[] = {
        0.40, 0.42, 0.48,                 // gaps 0.10, 0.08, 0.02 — last is at the boundary
        0.51, 0.49, 0.495, 0.503          // four close hits in a row → should stop
    };

    bool stopped = false;
    int  stopped_step = -1;
    for (double wr : readings) {
        step += 500;
        bool pass = within_win_rate_delta(wr, teacher_wr, delta);
        patience = next_patience(patience, pass);
        if (convergence_satisfied(step, min_steps, patience, target)) {
            stopped = true;
            stopped_step = step;
            break;
        }
    }

    ASSERT_TRUE(stopped) << "should have converged by the end of the sequence";
    // After "0.40, 0.42, 0.48" (gap 0.10, 0.08, 0.02 — 0.02 is at delta boundary,
    // so NOT within < 0.02). Patience resets. Then 0.51 (gap 0.01) → patience=1,
    // 0.49 (gap 0.01) → patience=2, 0.495 → patience=3 — converged at step 3000.
    EXPECT_EQ(stopped_step, 3000);
}

// ---------------------------------------------------------------------------
// update_ema
// ---------------------------------------------------------------------------

TEST(UpdateEmaTest, FirstCallSeedsValue) {
    EmaState s;
    EXPECT_FALSE(s.initialized);
    update_ema(s, 0.42, /*alpha=*/0.05);
    EXPECT_TRUE(s.initialized);
    EXPECT_DOUBLE_EQ(s.value, 0.42);  // first call seeds directly
}

TEST(UpdateEmaTest, SubsequentCallsBlend) {
    EmaState s;
    update_ema(s, 1.0, 0.1);  // seeded at 1.0
    update_ema(s, 0.0, 0.1);  // 0.1*0 + 0.9*1.0 = 0.9
    EXPECT_NEAR(s.value, 0.9, 1e-12);
    update_ema(s, 0.0, 0.1);  // 0.1*0 + 0.9*0.9 = 0.81
    EXPECT_NEAR(s.value, 0.81, 1e-12);
}

TEST(UpdateEmaTest, ConvergesToConstantInput) {
    EmaState s;
    for (int i = 0; i < 200; ++i) update_ema(s, 0.5, 0.05);
    // ~20-step half-life × 10 = well-converged after 200 steps.
    EXPECT_NEAR(s.value, 0.5, 1e-3);
}

// ---------------------------------------------------------------------------
// within_match_mse
// ---------------------------------------------------------------------------

TEST(WithinMatchMseTest, UninitializedNeverPasses) {
    EmaState s;  // not initialized
    EXPECT_FALSE(within_match_mse(s, 1.0));   // any positive threshold
    EXPECT_FALSE(within_match_mse(s, 1e-9));  // even very loose
}

TEST(WithinMatchMseTest, NonPositiveThresholdIsDisabled) {
    EmaState s;
    update_ema(s, 0.0, 0.5);  // tightly converged
    EXPECT_FALSE(within_match_mse(s, 0.0));   // disabled
    EXPECT_FALSE(within_match_mse(s, -1.0));  // disabled sentinel
}

TEST(WithinMatchMseTest, PassesWhenBelowThreshold) {
    EmaState s;
    update_ema(s, 0.001, 0.5);
    EXPECT_TRUE (within_match_mse(s, 0.01));   // 0.001 < 0.01
    EXPECT_FALSE(within_match_mse(s, 0.001));  // strict less-than, 0.001 !< 0.001
    EXPECT_FALSE(within_match_mse(s, 0.0001)); // 0.001 > 0.0001
}

// ---------------------------------------------------------------------------
// Composition: mse criterion can carry convergence when wr criterion fails.
// ---------------------------------------------------------------------------

TEST(ConvergenceCompositionTest, MseCriterionCanCarryConvergence) {
    const int    min_steps    = 100;
    const int    target       = 2;
    const double wr_delta     = 0.01;   // tight — wr always misses
    const double mse_target   = 0.05;
    const double teacher_wr   = 0.50;

    int patience = 0;
    EmaState mse;

    // Big wr gap (0.20) but mse converges fast.
    for (int step = 100; step <= 500; step += 100) {
        update_ema(mse, 0.01, 0.5);  // seeded low, stays low

        bool wr  = within_win_rate_delta(/*student=*/0.30, teacher_wr, wr_delta);
        bool mse_pass = within_match_mse(mse, mse_target);
        patience = next_patience(patience, wr || mse_pass);

        if (convergence_satisfied(step, min_steps, patience, target)) {
            EXPECT_GE(step, min_steps);
            EXPECT_TRUE(mse_pass);   // mse is what carried us
            EXPECT_FALSE(wr);
            SUCCEED() << "mse criterion fired at step " << step;
            return;
        }
    }
    FAIL() << "should have converged via mse criterion";
}

TEST(ConvergenceCompositionTest, NegativeDelta_NeverConverges) {
    const int    min_steps  = 100;
    const int    target     = 2;
    const double delta      = -1.0;      // disabled sentinel
    const double teacher_wr = 0.50;

    int patience = 0;
    for (int step = 100; step <= 10000; step += 100) {
        bool pass = within_win_rate_delta(/*student_wr=*/0.50,
                                          teacher_wr, delta);
        patience = next_patience(patience, pass);
        ASSERT_FALSE(convergence_satisfied(step, min_steps, patience, target))
            << "negative delta must never fire convergence (step=" << step << ")";
    }
    EXPECT_EQ(patience, 0);  // patience never advances either
}
