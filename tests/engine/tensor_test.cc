#include <gtest/gtest.h>
#include "engine/tensor.h"
#include "engine/board_init.h"
#include "engine/game_flow.h"
#include "engine/placement.h"
#include "engine/scoring.h"
#include "solver/precomputed_tables.h"

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
// Output buffer size matches kTensorSize
// ---------------------------------------------------------------------------
TEST_F(TensorTest, TensorSize_IsExactly809) {
    EXPECT_EQ(kTensorSize, 809);
}

// ---------------------------------------------------------------------------
// Empty board: all is_filled and score features = 0; probabilities > 0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, EmptyBoard_FilledFeaturesZero) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    // Group A: features at positions 0, 3, 6, ... (is_filled) and 1, 4, 7 ... (score/max)
    // should all be 0 since board is freshly initialised (all cells empty)
    for (int i = 0; i < 2 * kNumColumns * kNumRows; ++i) {
        int cell_idx = i;
        // Check the is_filled feature for each cell: every 3rd position starting from 0
        EXPECT_EQ(out[cell_idx * 3 + 0], 0.0f)
            << "is_filled at cell " << i << " should be 0";
        EXPECT_EQ(out[cell_idx * 3 + 1], 0.0f)
            << "score/max at cell " << i << " should be 0";
        EXPECT_EQ(out[cell_idx * 3 + 2], 0.0f)
            << "score/section_max at cell " << i << " should be 0";
    }
}

// ---------------------------------------------------------------------------
// Empty board: all probability features > 0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, EmptyBoard_ProbabilitiesPositive) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    // Group C starts at offset: 468 (Group A) + 96 (Group B) = 564
    int group_c_start = 468 + 96;
    for (int i = group_c_start; i < group_c_start + 156; ++i) {
        EXPECT_GT(out[i], 0.0f)
            << "Probability feature at position " << i << " should be > 0 on empty board";
    }
}

// ---------------------------------------------------------------------------
// Game progress on fresh board = 0.0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, EmptyBoard_GameProgressZero) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    // Group D: game progress is at offset 468 + 96 + 156 + 6 = 726
    int progress_offset = 468 + 96 + 156 + 6;
    EXPECT_NEAR(out[progress_offset], 0.0f, 1e-6f)
        << "Game progress should be 0 on fresh board";
}

// ---------------------------------------------------------------------------
// All features in [0, 1]
// ---------------------------------------------------------------------------
TEST_F(TensorTest, AllFeatures_InUnitRange) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    for (int i = 0; i < kTensorSize; ++i) {
        EXPECT_GE(out[i], 0.0f - 1e-6f) << "Feature " << i << " below 0";
        EXPECT_LE(out[i], 1.0f + 1e-6f) << "Feature " << i << " above 1";
    }
}

// ---------------------------------------------------------------------------
// Single placement: filled cell features are correct
// ---------------------------------------------------------------------------
TEST_F(TensorTest, SinglePlacement_CellFeaturesCorrect) {
    // Place score 18 in player 0's Free column (kColFree=1), 6s row (kRow6s=5)
    apply_placement(0, kColFree, kRow6s, 18, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, /*player=*/0, tables, out);

    // Group A: player 0 (pi=0) → base offset 0
    // col=kColFree=1, row=kRow6s=5 → cell index = 1*13 + 5 = 18
    // features at: 18*3 = 54
    int cell_feat_idx = (kColFree * kNumRows + kRow6s) * 3;
    EXPECT_NEAR(out[cell_feat_idx + 0], 1.0f, 1e-6f) << "is_filled should be 1";
    EXPECT_NEAR(out[cell_feat_idx + 1], 18.0f / 30.0f, 1e-6f) << "score/cell_max";
    EXPECT_NEAR(out[cell_feat_idx + 2], 18.0f / 30.0f, 1e-6f) << "score/section_max (upper)";
}

// ---------------------------------------------------------------------------
// Single placement: game progress = 1/156
// ---------------------------------------------------------------------------
TEST_F(TensorTest, SinglePlacement_ProgressCorrect) {
    apply_placement(0, kColFree, kRow6s, 18, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    int progress_offset = 468 + 96 + 156 + 6;
    EXPECT_NEAR(out[progress_offset], 1.0f / 156.0f, 1e-6f)
        << "Game progress should be 1/156 after one placement";
}

// ---------------------------------------------------------------------------
// Filled cell: probability feature = 1.0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, FilledCell_ProbabilityIsOne) {
    apply_placement(0, kColFree, kRow6s, 18, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    // Group C: player 0 (pi=0), col=kColFree=1, row=kRow6s=5
    // Offset = 468 + 96 + (0*6 + kColFree)*13 + kRow6s
    int group_c_start = 468 + 96;
    int prob_idx = group_c_start + (kColFree * kNumRows + kRow6s);
    EXPECT_NEAR(out[prob_idx], 1.0f, 1e-6f)
        << "Probability for filled cell should be 1.0";
}

// ---------------------------------------------------------------------------
// Perspective flipping: player 0's tensor has player 1's data as "opponent"
// and vice versa.
// ---------------------------------------------------------------------------
TEST_F(TensorTest, PerspectiveFlip_Symmetric) {
    // Place one cell for each player to make them distinguishable
    apply_placement(0, kColFree, kRow6s, 18, gs.board, ctx);
    apply_placement(1, kColFree, kRow3s, 9,  gs.board, ctx);

    float tensor0[kTensorSize] = {};
    float tensor1[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, tensor0);
    generate_tensor(gs.board, ctx, 1, tables, tensor1);

    // Group A: player 0's perspective
    //   tensor0[pi=0]: player 0's cells
    //   tensor0[pi=1]: player 1's cells
    //   tensor1[pi=0]: player 1's cells
    //   tensor1[pi=1]: player 0's cells
    // => tensor0[0..234] should equal tensor1[234..468] (up to 78 cells * 3 features)

    int half_A = kNumColumns * kNumRows * 3;  // 234 features per player in Group A
    for (int i = 0; i < half_A; ++i) {
        EXPECT_NEAR(tensor0[i], tensor1[half_A + i], 1e-6f)
            << "Group A: tensor0 player0 != tensor1 player1 at feature " << i;
        EXPECT_NEAR(tensor0[half_A + i], tensor1[i], 1e-6f)
            << "Group A: tensor0 player1 != tensor1 player0 at feature " << i;
    }
}

// ---------------------------------------------------------------------------
// SS forced scratch: LS scratched → SS probability = 0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, SS_LsScratched_ProbabilityZero) {
    int p = 0;
    int col = kColFree;
    ctx.ls_scratched[p][col] = true;
    apply_placement(p, col, kRowLS, 0, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, p, tables, out);

    // Group C for player 0 (pi=0), col=kColFree, row=kRowSS
    int group_c_start = 468 + 96;
    int prob_idx = group_c_start + (kColFree * kNumRows + kRowSS);
    EXPECT_NEAR(out[prob_idx], 0.0f, 1e-6f)
        << "SS probability should be 0 when LS is scratched";
}

// ---------------------------------------------------------------------------
// LS forced scratch: SS scratched → LS probability = 0
// ---------------------------------------------------------------------------
TEST_F(TensorTest, LS_SsScratched_ProbabilityZero) {
    int p = 0;
    int col = kColFree;
    ctx.ss_scratched[p][col] = true;
    apply_placement(p, col, kRowSS, 0, gs.board, ctx);

    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, p, tables, out);

    int group_c_start = 468 + 96;
    int prob_idx = group_c_start + (kColFree * kNumRows + kRowLS);
    EXPECT_NEAR(out[prob_idx], 0.0f, 1e-6f)
        << "LS probability should be 0 when SS is scratched";
}

// ---------------------------------------------------------------------------
// Run generate_tensor on a mid-game board without crashing
// ---------------------------------------------------------------------------
TEST_F(TensorTest, MidGame_NoCrash) {
    // Play 20 turns heuristically to get a mid-game state
    RNG rng(9999);
    for (int i = 0; i < 20 && !is_terminal(gs.board); ++i) {
        // Advance with a simple policy: pick first legal placement
        const auto& legal = get_legal_placements(gs, ctx);
        if (legal.count > 0) {
            auto pl = legal.placements[0];
            int sc = calculate_score(pl.row, gs.dice,
                                     gs.board.current_player, pl.column,
                                     gs.board, ctx);
            perform_placement(gs, ctx, pl.column, pl.row, rng);
        }
    }
    float out[kTensorSize] = {};
    ASSERT_NO_THROW(generate_tensor(gs.board, ctx, gs.board.current_player, tables, out));

    // All features in range
    for (int i = 0; i < kTensorSize; ++i) {
        EXPECT_GE(out[i], 0.0f - 1e-4f) << "Feature " << i << " below 0 in mid-game";
        EXPECT_LE(out[i], 1.0f + 1e-4f) << "Feature " << i << " above 1 in mid-game";
    }
}

// ---------------------------------------------------------------------------
// Perspective flipping in Group B: duel advantages swap correctly
// ---------------------------------------------------------------------------
TEST_F(TensorTest, PerspectiveFlip_GroupB_Symmetric) {
    // Place different scores so players are distinguishable in duel features
    apply_placement(0, kColFree, kRow6s, 24, gs.board, ctx);
    apply_placement(1, kColFree, kRow6s, 12, gs.board, ctx);

    float tensor0[kTensorSize] = {};
    float tensor1[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, tensor0);
    generate_tensor(gs.board, ctx, 1, tables, tensor1);

    // Group B: 2 players × 6 columns × 8 features = 96 features, starting at offset 468
    // Features 3 (advantage) and 4 (disadvantage) should swap between perspectives.
    int group_b_start = 468;
    int feats_per_col = 8;
    int half_B = kNumColumns * feats_per_col;  // 48 features per player

    for (int col = 0; col < kNumColumns; ++col) {
        int p0_base = group_b_start + col * feats_per_col;          // tensor0 pi=0
        int p1_base = group_b_start + half_B + col * feats_per_col; // tensor0 pi=1
        int q0_base = group_b_start + col * feats_per_col;          // tensor1 pi=0
        int q1_base = group_b_start + half_B + col * feats_per_col; // tensor1 pi=1

        // tensor0's "my advantage" (pi=0) should equal tensor1's "my advantage" (pi=1)
        // because tensor0 pi=0 is player 0, tensor1 pi=1 is also player 0.
        EXPECT_NEAR(tensor0[p0_base + 3], tensor1[q1_base + 3], 1e-5f)
            << "Group B advantage mismatch at col " << col;
        EXPECT_NEAR(tensor0[p0_base + 4], tensor1[q1_base + 4], 1e-5f)
            << "Group B disadvantage mismatch at col " << col;

        // And vice versa
        EXPECT_NEAR(tensor0[p1_base + 3], tensor1[q0_base + 3], 1e-5f)
            << "Group B opp advantage mismatch at col " << col;
        EXPECT_NEAR(tensor0[p1_base + 4], tensor1[q0_base + 4], 1e-5f)
            << "Group B opp disadvantage mismatch at col " << col;
    }
}

// ---------------------------------------------------------------------------
// Coefficient normalisation: coefficients[col] / 18 in Group D
// ---------------------------------------------------------------------------
TEST_F(TensorTest, Coefficients_Normalised) {
    float out[kTensorSize] = {};
    generate_tensor(gs.board, ctx, 0, tables, out);

    // Group D: coefficients start at offset 468 + 96 + 156 = 720
    int coeff_start = 468 + 96 + 156;
    for (int col = 0; col < kNumColumns; ++col) {
        float expected = static_cast<float>(gs.board.coefficients[col]) / 18.0f;
        EXPECT_NEAR(out[coeff_start + col], expected, 1e-6f)
            << "Coefficient for col " << col << " incorrect";
    }
}
