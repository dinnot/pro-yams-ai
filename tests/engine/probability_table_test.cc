#include <gtest/gtest.h>
#include "solver/precomputed_tables.h"
#include "engine/probability_tables.h"
#include "engine/constants.h"

// Shared fixture: initialise tables once for all tests.
class ProbabilityTableTest : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) { init_precomputed_tables(tables); initialised = true; }
    }
};
PrecomputedTables ProbabilityTableTest::tables;
bool ProbabilityTableTest::initialised = false;

// ---------------------------------------------------------------------------
// Boundary: threshold=0 always yields 1.0 (every score >= 0)
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, Threshold0_AlwaysOne) {
    for (int row = 0; row < kNumRows; ++row) {
        EXPECT_NEAR(tables.prob_tables.prob_3rolls[row][0], 1.0, 1e-9)
            << "prob_3rolls[" << row << "][0] should be 1.0";
        EXPECT_NEAR(tables.prob_tables.prob_2rolls[row][0], 1.0, 1e-9)
            << "prob_2rolls[" << row << "][0] should be 1.0";
    }
}

// ---------------------------------------------------------------------------
// Threshold=1: probability of any non-zero score (non-scratch)
// This should be > 0 for all rows and close to 1.0 for number rows.
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, Threshold1_PositiveForAllRows) {
    for (int row = 0; row < kNumRows; ++row) {
        EXPECT_GT(tables.prob_tables.prob_3rolls[row][1], 0.0)
            << "prob_3rolls[" << row << "][1] should be > 0";
        EXPECT_GT(tables.prob_tables.prob_2rolls[row][1], 0.0)
            << "prob_2rolls[" << row << "][1] should be > 0";
    }
}

// ---------------------------------------------------------------------------
// Number rows: very high probability of non-zero score
// (With 3 rolls, probability of getting at least one of a face value > 99%)
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, NumberRows_HighProbability) {
    // P(at least one die showing face X in 3 full rolls) ≈ 1 - (5/6)^15 ≈ 0.935
    for (int row = 0; row <= kRow6s; ++row) {
        EXPECT_GT(tables.prob_tables.prob_3rolls[row][1], 0.90)
            << "P(non-zero score for number row " << row << ") should be > 90% with 3 rolls";
    }
}

// ---------------------------------------------------------------------------
// Threshold above row maximum: probability = 0.0
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, ThresholdAboveMax_Zero) {
    // Max scores per row from kMaxScorePerRow: {5,10,15,20,25,30,29,30,50,54,50,75,100}
    // Check threshold = max+1 (where <= 100)
    constexpr int kMax[kNumRows] = {5, 10, 15, 20, 25, 30, 29, 30, 50, 54, 50, 75, 100};
    for (int row = 0; row < kNumRows; ++row) {
        int above = kMax[row] + 1;
        if (above > 100) continue;
        EXPECT_NEAR(tables.prob_tables.prob_3rolls[row][above], 0.0, 1e-9)
            << "prob_3rolls[" << row << "][" << above << "] should be 0 (above max)";
        EXPECT_NEAR(tables.prob_tables.prob_2rolls[row][above], 0.0, 1e-9)
            << "prob_2rolls[" << row << "][" << above << "] should be 0";
    }
}

// ---------------------------------------------------------------------------
// Threshold at max score: probability > 0 (achievable but rare)
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, ThresholdAtMax_Positive) {
    constexpr int kMax[kNumRows] = {5, 10, 15, 20, 25, 30, 29, 30, 50, 54, 50, 75, 100};
    for (int row = 0; row < kNumRows; ++row) {
        int max_score = kMax[row];
        EXPECT_GT(tables.prob_tables.prob_3rolls[row][max_score], 0.0)
            << "prob_3rolls[" << row << "][" << max_score << "] should be > 0";
    }
}

// ---------------------------------------------------------------------------
// Monotone in threshold: prob(row, t) >= prob(row, t+1)
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, Monotone_HigherThresholdLowerProb) {
    for (int row = 0; row < kNumRows; ++row) {
        for (int t = 0; t < 100; ++t) {
            EXPECT_GE(tables.prob_tables.prob_3rolls[row][t],
                      tables.prob_tables.prob_3rolls[row][t + 1] - 1e-12)
                << "prob_3rolls[" << row << "] not monotone at threshold " << t;
            EXPECT_GE(tables.prob_tables.prob_2rolls[row][t],
                      tables.prob_tables.prob_2rolls[row][t + 1] - 1e-12)
                << "prob_2rolls[" << row << "] not monotone at threshold " << t;
        }
    }
}

// ---------------------------------------------------------------------------
// Turbo (2-roll) always <= Normal (3-roll): more rolls = higher probability
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, TurboLeNormal) {
    for (int row = 0; row < kNumRows; ++row) {
        for (int t = 0; t <= 100; ++t) {
            EXPECT_LE(tables.prob_tables.prob_2rolls[row][t],
                      tables.prob_tables.prob_3rolls[row][t] + 1e-12)
                << "prob_2rolls[" << row << "][" << t << "] > prob_3rolls (should be <=)";
        }
    }
}

// ---------------------------------------------------------------------------
// Probability values in [0, 1]
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, ValuesInUnitRange) {
    for (int row = 0; row < kNumRows; ++row) {
        for (int t = 0; t <= 100; ++t) {
            EXPECT_GE(tables.prob_tables.prob_3rolls[row][t], 0.0 - 1e-12);
            EXPECT_LE(tables.prob_tables.prob_3rolls[row][t], 1.0 + 1e-12);
            EXPECT_GE(tables.prob_tables.prob_2rolls[row][t], 0.0 - 1e-12);
            EXPECT_LE(tables.prob_tables.prob_2rolls[row][t], 1.0 + 1e-12);
        }
    }
}

// ---------------------------------------------------------------------------
// Yams probability sanity check (~4.6% chance of rolling Yams in 3 rolls)
// For Y row, any Yams qualifies (threshold=75 gives prob of any Yams).
// Exact value: 1 - (1 - 6/6^5)^... difficult, use soft bound.
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, Yams_3Rolls_ReasonableProbability) {
    double p = tables.prob_tables.prob_3rolls[kRowY][75];  // any Yams
    // Not astronomical, but meaningful — between 1% and 25%
    EXPECT_GT(p, 0.01) << "P(any Yams in 3 rolls) should be > 1%";
    EXPECT_LT(p, 0.25) << "P(any Yams in 3 rolls) should be < 25%";
}

// ---------------------------------------------------------------------------
// U8 (Under Eight) sanity check
// Score 60 = sum==8: threshold 60. Should be achievable with good probability.
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, U8_Threshold60_Reasonable) {
    double p = tables.prob_tables.prob_3rolls[kRowU8][60];  // sum <= 8 needed
    EXPECT_GT(p, 0.05) << "P(U8 score >= 60 in 3 rolls) should be > 5%";
}

// ---------------------------------------------------------------------------
// STR: only two values (45, 50). Threshold=45 should give P > threshold=50.
// ---------------------------------------------------------------------------
TEST_F(ProbabilityTableTest, STR_Threshold45_GtThreshold50) {
    double p45 = tables.prob_tables.prob_3rolls[kRowSTR][45];
    double p50 = tables.prob_tables.prob_3rolls[kRowSTR][50];
    EXPECT_GE(p45, p50) << "P(STR>=45) should be >= P(STR>=50)";
    EXPECT_GT(p45, 0.0) << "P(STR>=45) should be > 0 with 3 rolls";
}
