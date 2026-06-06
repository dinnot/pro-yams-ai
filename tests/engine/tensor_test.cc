#include <gtest/gtest.h>
#include <cmath>
#include "engine/tensor.h"
#include "engine/board_init.h"
#include "engine/game_flow.h"
#include "engine/placement.h"
#include "engine/scoring.h"
#include "solver/precomputed_tables.h"

// V2.1 layout offsets (986 features)
//   Group A: [0,        312)
//   Group B: [312,      420)
//   Group C: [420,      576)
//   Group D: [576,      590)
//   Group E: [590,      806)
//   Group F: [806,      986)
constexpr int kGroupAStart = 0;
constexpr int kGroupASize  = 312;
constexpr int kGroupBStart = kGroupAStart + kGroupASize;
constexpr int kGroupBSize  = 108;
constexpr int kGroupCStart = kGroupBStart + kGroupBSize;
constexpr int kGroupCSize  = 156;
constexpr int kGroupDStart = kGroupCStart + kGroupCSize;
constexpr int kGroupDSize  = 14;
constexpr int kGroupEStart = kGroupDStart + kGroupDSize;
constexpr int kGroupESize  = 216;
constexpr int kGroupFStart = kGroupEStart + kGroupESize;
constexpr int kGroupFSize  = 180;

// Shared fixture
class TensorTest : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) { init_precomputed_tables(tables); initialised = true; }
    }
    void SetUp() override {
        RNG rng(seed_++);
        init_game(gs, ctx, rng);
    }
    GameState gs;
    GameContext ctx;
    static int seed_;
};
PrecomputedTables TensorTest::tables;
bool TensorTest::initialised = false;
int TensorTest::seed_ = 5000;

// ---------------------------------------------------------------------------
// Layout sanity
// ---------------------------------------------------------------------------
TEST_F(TensorTest, V1LayoutOffsetsSumToV1Size) {
    // The A..F groups below are the frozen V1 layout (986). They are now the
    // PREFIX of the latest tensor; Group G is appended after them.
    EXPECT_EQ(Yams1v1::kTensorSizeV1, 986);
    EXPECT_EQ(kGroupAStart + kGroupASize + kGroupBSize + kGroupCSize
                + kGroupDSize + kGroupESize + kGroupFSize,
              Yams1v1::kTensorSizeV1);
    // Latest adds Group G.
    EXPECT_EQ(kTensorSize, Yams1v1::kTensorSizeV1 + Yams1v1::kGroupGSize);
}

// ---------------------------------------------------------------------------
// Empty board: Group A is_filled and score features = 0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, EmptyBoard_GroupAZero) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    // Group A: 2 features per cell (is_filled, score/cell_max)
    for (int i = 0; i < kGroupASize; ++i) {
        EXPECT_EQ(out[kGroupAStart + i], 0.0f)
            << "Group A feature " << i << " should be 0 on empty board";
    }
}

// ---------------------------------------------------------------------------
// Empty board: Group C 1-turn probabilities all > 0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, EmptyBoard_GroupCProbabilitiesPositive) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    for (int i = 0; i < kGroupCSize; ++i) {
        EXPECT_GT(out[kGroupCStart + i], 0.0f)
            << "Group C probability at " << i << " should be > 0 on empty board";
    }
}

// ---------------------------------------------------------------------------
// Game progress on fresh board = 0.0 (Group D[6])
// ---------------------------------------------------------------------------
TEST_F(TensorTest, EmptyBoard_GameProgressZero) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    int progress_idx = kGroupDStart + 6;  // 6 coefficients then progress
    EXPECT_NEAR(out[progress_idx], 0.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// All features in [-1, 1] (signed tanh / crush-diff features may go negative)
// ---------------------------------------------------------------------------
TEST_F(TensorTest, AllFeatures_InSignedUnitRange) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    for (int i = 0; i < kTensorSize; ++i) {
        EXPECT_GE(out[i], -1.0f - 1e-5f) << "Feature " << i << " below -1";
        EXPECT_LE(out[i],  1.0f + 1e-5f) << "Feature " << i << " above 1";
        EXPECT_FALSE(std::isnan(out[i])) << "Feature " << i << " is NaN";
    }
}

// ---------------------------------------------------------------------------
// Single placement: filled cell features in Group A correct
// ---------------------------------------------------------------------------
TEST_F(TensorTest, SinglePlacement_CellFeaturesCorrect) {
    apply_placement(0, kColFree, kRow6s, 18, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, /*player=*/0, tables, out);

    // Group A: player 0 (pi=0) → base 0; col=Free=1, row=6s=5
    // index = (col * 13 + row) * 2
    int cell_feat_idx = (kColFree * kNumRows + kRow6s) * 2;
    EXPECT_NEAR(out[cell_feat_idx + 0], 1.0f, 1e-6f) << "is_filled should be 1";
    EXPECT_NEAR(out[cell_feat_idx + 1], 18.0f / 30.0f, 1e-6f) << "score / cell_max";
}

// ---------------------------------------------------------------------------
// Single placement: game progress = 1/156
// ---------------------------------------------------------------------------
TEST_F(TensorTest, SinglePlacement_ProgressCorrect) {
    apply_placement(0, kColFree, kRow6s, 18, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    int progress_idx = kGroupDStart + 6;
    EXPECT_NEAR(out[progress_idx], 1.0f / 156.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Filled cell: probability feature in Group C = 1.0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, FilledCell_GroupCProbabilityIsOne) {
    apply_placement(0, kColFree, kRow6s, 18, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    int prob_idx = kGroupCStart + (kColFree * kNumRows + kRow6s);
    EXPECT_NEAR(out[prob_idx], 1.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// SS forced scratch under a filled LS: with LS=20 and no SS golden, SS must be
// < 20 but its natural floor is 20 → no legal SS sum → Group C p_one = 0.
// (Guards the floor=20 edge the old `golden_min >= ls_val` check missed.)
// ---------------------------------------------------------------------------
TEST_F(TensorTest, SS_ForcedScratchUnderLS20_GroupCZero) {
    apply_placement(0, kColFree, kRowLS, 20, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    int ss_idx = kGroupCStart + (kColFree * kNumRows + kRowSS);
    EXPECT_NEAR(out[ss_idx], 0.0f, 1e-6f) << "SS must be a forced scratch under LS=20";
}

// ---------------------------------------------------------------------------
// SS band still open under a filled LS: with LS=25 and no SS golden, SS can
// still legally land in [20,24] → Group C p_one stays > 0 (no over-zeroing).
// ---------------------------------------------------------------------------
TEST_F(TensorTest, SS_BandOpenUnderLS25_GroupCPositive) {
    apply_placement(0, kColFree, kRowLS, 25, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    int ss_idx = kGroupCStart + (kColFree * kNumRows + kRowSS);
    EXPECT_GT(out[ss_idx], 0.0f) << "SS band [20,24] is still legal under LS=25";
}

// ---------------------------------------------------------------------------
// Perspective flipping: Group A halves swap
// ---------------------------------------------------------------------------
TEST_F(TensorTest, PerspectiveFlip_GroupASymmetric) {
    apply_placement(0, kColFree, kRow6s, 18, gs.board, ctx);
    apply_placement(1, kColFree, kRow3s, 9,  gs.board, ctx);

    float tensor0[kTensorSize] = {};
    float tensor1[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, tensor0);
    generate_tensor(gs.board, ctx, 1, tables, tensor1);

    int half_A = kNumColumns * kNumRows * 2;  // 156 features per player
    for (int i = 0; i < half_A; ++i) {
        EXPECT_NEAR(tensor0[i], tensor1[half_A + i], 1e-6f)
            << "Group A: tensor0 P0 != tensor1 P1 at feature " << i;
        EXPECT_NEAR(tensor0[half_A + i], tensor1[i], 1e-6f)
            << "Group A: tensor0 P1 != tensor1 P0 at feature " << i;
    }
}

// ---------------------------------------------------------------------------
// SS forced scratch: LS scratched → SS Group C probability = 0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, SS_LsScratched_ProbabilityZero) {
    int p = 0;
    int col = kColFree;
    ctx.ls_scratched[p][col] = true;
    apply_placement(p, col, kRowLS, 0, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, p, tables, out);

    int prob_idx = kGroupCStart + (kColFree * kNumRows + kRowSS);
    EXPECT_NEAR(out[prob_idx], 0.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// LS forced scratch: SS scratched → LS Group C probability = 0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, LS_SsScratched_ProbabilityZero) {
    int p = 0;
    int col = kColFree;
    ctx.ss_scratched[p][col] = true;
    apply_placement(p, col, kRowSS, 0, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, p, tables, out);

    int prob_idx = kGroupCStart + (kColFree * kNumRows + kRowLS);
    EXPECT_NEAR(out[prob_idx], 0.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Mid-game: tensor generation does not crash; values stay in [-1, 1]
// ---------------------------------------------------------------------------
TEST_F(TensorTest, MidGame_NoCrash) {
    RNG rng(9999);
    for (int i = 0; i < 20 && !is_terminal(gs.board); ++i) {
        const auto& legal = get_legal_placements(gs, ctx);
        if (legal.count > 0) {
            auto pl = legal.placements[0];
            (void)calculate_score(pl.row, gs.dice,
                                  gs.board.current_player, pl.column,
                                  gs.board, ctx);
            perform_placement(gs, ctx, pl.column, pl.row, rng);
        }
    }
    float out[kTensorSize] = {};
    ASSERT_NO_THROW(generate_tensor(gs.board, ctx, gs.board.current_player, tables, out));

    for (int i = 0; i < kTensorSize; ++i) {
        EXPECT_GE(out[i], -1.0f - 1e-4f) << "Feature " << i << " below -1 in mid-game";
        EXPECT_LE(out[i],  1.0f + 1e-4f) << "Feature " << i << " above 1 in mid-game";
        EXPECT_FALSE(std::isnan(out[i])) << "Feature " << i << " NaN";
    }
}

// ---------------------------------------------------------------------------
// Coefficient normalisation in Group D (first 6 features)
// ---------------------------------------------------------------------------
TEST_F(TensorTest, Coefficients_Normalised) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    for (int col = 0; col < kNumColumns; ++col) {
        float expected = static_cast<float>(gs.board.coefficients[col]) / 18.0f;
        EXPECT_NEAR(out[kGroupDStart + col], expected, 1e-6f)
            << "Coefficient for col " << col << " incorrect";
    }
}

// ---------------------------------------------------------------------------
// Phase flags: empty board → all phase flags 0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, EmptyBoard_PhaseFlagsZero) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    // Group D layout: 6 coeffs, 1 progress, 1 dn_signed, 1 dE_signed,
    //                 1 won, 1 lost, 3 phase flags
    int phase_start = kGroupDStart + 6 + 1 + 1 + 1 + 1 + 1;
    EXPECT_NEAR(out[phase_start + 0], 0.0f, 1e-6f);  // is_midgame
    EXPECT_NEAR(out[phase_start + 1], 0.0f, 1e-6f);  // is_endgame
    EXPECT_NEAR(out[phase_start + 2], 0.0f, 1e-6f);  // is_final
}

// ---------------------------------------------------------------------------
// expected_duel_margin: clean-column bonus folded into the EV margin,
// weighted by P_clean so the margin trends toward the clean-completed value
// as P_clean goes 0 -> 1 (instead of stepping when is_clean flips).
// ---------------------------------------------------------------------------
TEST(ExpectedDuelMargin, CleanBonusWeightedByPClean) {
    // Neither side can clean -> pure expected-raw margin.
    EXPECT_DOUBLE_EQ(expected_duel_margin(300, 300, 0.0f, 0.0f, 10, 1), 0.0);
    EXPECT_DOUBLE_EQ(expected_duel_margin(320, 300, 0.0f, 0.0f, 10, 1), 200.0);

    // Opp certain to complete a clean column: margin drops by 200*coeff.
    EXPECT_DOUBLE_EQ(expected_duel_margin(300, 300, 0.0f, 1.0f, 10, 1), -2000.0);
    // Me certain to clean: symmetric.
    EXPECT_DOUBLE_EQ(expected_duel_margin(300, 300, 1.0f, 0.0f, 10, 1), +2000.0);

    // Monotone toward the clean-completed value as opp P_clean rises 0->.5->1.
    double m0 = expected_duel_margin(300, 300, 0.0f, 0.0f, 10, 1);
    double m1 = expected_duel_margin(300, 300, 0.0f, 0.5f, 10, 1);
    double m2 = expected_duel_margin(300, 300, 0.0f, 1.0f, 10, 1);
    EXPECT_GT(m0, m1);
    EXPECT_GT(m1, m2);

    // Under crush (x2) the clean bonus is 100, and the margin is doubled:
    // me_adj=300, opp_adj=300+100 -> (300-400)*10*2 = -2000.
    EXPECT_DOUBLE_EQ(expected_duel_margin(300, 300, 0.0f, 1.0f, 10, 2), -2000.0);
}
