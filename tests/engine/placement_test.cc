#include <gtest/gtest.h>
#include "engine/placement.h"
#include "engine/board_init.h"
#include "engine/rng.h"

static void make_fresh(BoardState& board, GameContext& ctx) {
    RNG rng(99);
    init_board(board, rng);
    init_context(ctx, board);
}

TEST(Placement, BasicPlacement) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    apply_placement(0, kColFree, kRow5s, 15, board, ctx);
    EXPECT_EQ(board.cells[0][kColFree][kRow5s], 15);
    EXPECT_EQ(board.cells_filled, 1);
    EXPECT_EQ(ctx.golden_max[kColFree][kRow5s], 15);
    EXPECT_FALSE(ctx.legal_all[0].is_legal[kColFree][kRow5s]);
}

TEST(Placement, GoldenMaxUpdated) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    apply_placement(0, kColFree, kRow6s, 18, board, ctx);
    EXPECT_EQ(ctx.golden_max[kColFree][kRow6s], 18);
    apply_placement(1, kColFree, kRow6s, 24, board, ctx);
    EXPECT_EQ(ctx.golden_max[kColFree][kRow6s], 24);
}

TEST(Placement, UpperSumTracking) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    apply_placement(0, kColFree, kRow1s, 3, board, ctx);
    apply_placement(0, kColFree, kRow2s, 8, board, ctx);
    EXPECT_EQ(ctx.upper_sum[0][kColFree], 11);
}

TEST(Placement, LowerScratchTracking) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    apply_placement(0, kColFree, kRowFH, 0, board, ctx);  // scratch lower
    EXPECT_TRUE(ctx.lower_has_scratch[0][kColFree]);
    EXPECT_FALSE(ctx.lower_has_scratch[1][kColFree]);
}

TEST(Placement, LowerNonzeroNoScratch) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    apply_placement(0, kColFree, kRowFH, 33, board, ctx);
    EXPECT_FALSE(ctx.lower_has_scratch[0][kColFree]);
}

TEST(Placement, SS_ScratchSetsSsFlag) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    apply_placement(0, kColFree, kRowSS, 0, board, ctx);
    EXPECT_TRUE(ctx.ss_scratched[0][kColFree]);
}

TEST(Placement, ScratchSS_Then_FilledLS_MutualDestruction) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    // Fill LS with 25
    board.cells[0][kColFree][kRowLS] = 25;
    ctx.golden_max[kColFree][kRowLS] = 25;
    // Now scratch SS → LS should be destroyed
    apply_placement(0, kColFree, kRowSS, 0, board, ctx);
    EXPECT_EQ(board.cells[0][kColFree][kRowLS], 0);
    EXPECT_TRUE(ctx.ls_scratched[0][kColFree]);
    EXPECT_TRUE(ctx.lower_has_scratch[0][kColFree]);
    // golden_max for LS should be recalculated (player 1 has no LS → max=0)
    EXPECT_EQ(ctx.golden_max[kColFree][kRowLS], 0);
}

TEST(Placement, ScratchLS_Then_FilledSS_MutualDestruction) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    // Fill SS with 22
    board.cells[0][kColFree][kRowSS] = 22;
    ctx.golden_max[kColFree][kRowSS] = 22;
    // Now scratch LS → SS should be destroyed
    apply_placement(0, kColFree, kRowLS, 0, board, ctx);
    EXPECT_EQ(board.cells[0][kColFree][kRowSS], 0);
    EXPECT_TRUE(ctx.ss_scratched[0][kColFree]);
    EXPECT_TRUE(ctx.lower_has_scratch[0][kColFree]);
}

TEST(Placement, NonTurboCellsRemaining_Decremented) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    EXPECT_EQ(ctx.non_turbo_cells_remaining[0], 65);

    apply_placement(0, kColFree, kRow1s, 3, board, ctx);
    EXPECT_EQ(ctx.non_turbo_cells_remaining[0], 64);
    apply_placement(0, kColTurbo, kRow2s, 4, board, ctx);
    EXPECT_EQ(ctx.non_turbo_cells_remaining[0], 64); // Unchanged
}

TEST(Placement, ScratchSS_WhenLS_Empty_NoDestruction) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    // LS not filled yet
    apply_placement(0, kColFree, kRowSS, 0, board, ctx);
    EXPECT_TRUE(ctx.ss_scratched[0][kColFree]);
    // LS still empty, not auto-scratched yet (but will be forced if attempted)
    EXPECT_EQ(board.cells[0][kColFree][kRowLS], kCellEmpty);
    EXPECT_FALSE(ctx.ls_scratched[0][kColFree]);
}
