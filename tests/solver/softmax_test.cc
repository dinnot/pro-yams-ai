#include <gtest/gtest.h>
#include "solver/solver.h"
#include "engine/rng.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static double logit(double v) {
    constexpr double kClampMin = 1e-6;
    constexpr double kClampMax = 1.0 - 1e-6;
    double vc = std::max(kClampMin, std::min(kClampMax, v));
    return std::log(vc / (1.0 - vc));
}

// ---------------------------------------------------------------------------
// Logit transformation verification
// ---------------------------------------------------------------------------
TEST(Softmax, LogitTransform_HalfIsZero) {
    EXPECT_NEAR(logit(0.5), 0.0, 1e-10);
}
TEST(Softmax, LogitTransform_095) {
    EXPECT_NEAR(logit(0.95), 2.9444389792, 1e-6);
}
TEST(Softmax, LogitTransform_005) {
    EXPECT_NEAR(logit(0.05), -2.9444389792, 1e-6);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------
TEST(Softmax, Deterministic_SameSeedSameResult) {
    double vals[3] = {0.9, 0.5, 0.1};
    RNG rng1(42), rng2(42);
    EXPECT_EQ(softmax_sample(vals, 3, 1.0, rng1),
              softmax_sample(vals, 3, 1.0, rng2));
}

// ---------------------------------------------------------------------------
// Single value → always returns 0
// ---------------------------------------------------------------------------
TEST(Softmax, SingleValue_AlwaysZero) {
    RNG rng(1);
    double vals[1] = {0.7};
    for (int i = 0; i < 100; ++i)
        EXPECT_EQ(softmax_sample(vals, 1, 1.0, rng), 0);
}

// ---------------------------------------------------------------------------
// Boundary clamping — no NaN/inf
// ---------------------------------------------------------------------------
TEST(Softmax, BoundaryClamping_NoBadValues) {
    RNG rng(2);
    double vals[2] = {1.0, 0.0};  // Will be clamped.
    for (int i = 0; i < 100; ++i) {
        int result = softmax_sample(vals, 2, 1.0, rng);
        EXPECT_GE(result, 0);
        EXPECT_LT(result, 2);
    }
}

TEST(Softmax, IdenticalValues_RoughlyUniform) {
    // Values {0.5, 0.5, 0.5} — with logit all 0 → softmax uniform.
    RNG rng(3);
    double vals[3] = {0.5, 0.5, 0.5};
    int counts[3] = {};
    int N = 9000;
    for (int i = 0; i < N; ++i) counts[softmax_sample(vals, 3, 1.0, rng)]++;
    for (int k = 0; k < 3; ++k)
        EXPECT_NEAR(counts[k] / (double)N, 1.0/3.0, 0.05)
            << "Uniform values should yield roughly uniform sampling";
}

// ---------------------------------------------------------------------------
// Temperature = 0.001 (near greedy) → almost always picks highest value
// ---------------------------------------------------------------------------
TEST(Softmax, LowTemperature_NearlyGreedy) {
    RNG rng(4);
    double vals[3] = {0.9, 0.5, 0.1};
    int count_best = 0;
    int N = 1000;
    for (int i = 0; i < N; ++i)
        if (softmax_sample(vals, 3, 0.001, rng) == 0) ++count_best;
    EXPECT_GT(count_best, 990) << "Low temperature should pick argmax >99% of the time";
}

// ---------------------------------------------------------------------------
// High temperature → roughly uniform among options
// ---------------------------------------------------------------------------
TEST(Softmax, HighTemperature_NearlyUniform) {
    RNG rng(5);
    double vals[3] = {0.9, 0.5, 0.1};
    int counts[3] = {};
    int N = 12000;
    for (int i = 0; i < N; ++i) counts[softmax_sample(vals, 3, 100.0, rng)]++;
    for (int k = 0; k < 3; ++k)
        EXPECT_NEAR(counts[k] / (double)N, 1.0/3.0, 0.04)
            << "High temperature should be roughly uniform";
}

// ---------------------------------------------------------------------------
// Decisive values with T=1: {0.95, 0.05} — high wins overwhelmingly
// ---------------------------------------------------------------------------
TEST(Softmax, DecisiveValues_HighWinsVast) {
    RNG rng(6);
    double vals[2] = {0.95, 0.05};
    int count_high = 0;
    int N = 1000;
    for (int i = 0; i < N; ++i)
        if (softmax_sample(vals, 2, 1.0, rng) == 0) ++count_high;
    EXPECT_GT(count_high, 990)
        << "With logit(0.95) >> logit(0.05) at T=1, high value wins >99%";
}

// ---------------------------------------------------------------------------
// Close values: {0.6, 0.5, 0.4} — all chosen with some frequency
// ---------------------------------------------------------------------------
TEST(Softmax, CloseValues_AllChosen) {
    RNG rng(7);
    double vals[3] = {0.6, 0.5, 0.4};
    int counts[3] = {};
    int N = 6000;
    for (int i = 0; i < N; ++i) counts[softmax_sample(vals, 3, 1.0, rng)]++;
    // All three should be chosen with non-trivial frequency.
    for (int k = 0; k < 3; ++k)
        EXPECT_GT(counts[k], 100)
            << "All close values should be chosen at least occasionally";
    // Highest value should be chosen most often.
    EXPECT_GT(counts[0], counts[2]) << "Highest value (0.6) should be chosen more than lowest";
}

// ---------------------------------------------------------------------------
// Numerical stability: very close values
// ---------------------------------------------------------------------------
TEST(Softmax, NumericalStability_VeryCloseValues) {
    RNG rng(8);
    double vals[3] = {0.501, 0.500, 0.499};
    for (int i = 0; i < 1000; ++i) {
        int result = softmax_sample(vals, 3, 1.0, rng);
        EXPECT_GE(result, 0);
        EXPECT_LT(result, 3);
    }
}
