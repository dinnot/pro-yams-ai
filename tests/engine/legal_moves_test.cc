#include <gtest/gtest.h>
#include "engine/legal_moves.h"
#include "engine/board_init.h"
#include "engine/rng.h"
#include "engine/placement.h"

static void make_fresh(BoardState& board, GameContext& ctx) {
    RNG rng(42);
    init_board(board, rng);
    init_context(ctx, board);
}

static bool has_placement(const GameContext& ctx, int player, int col, int row) {
    return ctx.legal_all[player].is_legal[col][row];
}

// ---------------------------------------------------------------------------
// Down column
// ---------------------------------------------------------------------------
TEST(LegalMoves, Down_FreshOnlyRow0) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    EXPECT_TRUE(has_placement(ctx, p, kColDown, 0));
    for (int r = 1; r < kNumRows; ++r)
        EXPECT_FALSE(has_placement(ctx, p, kColDown, r));
}
TEST(LegalMoves, Down_AfterRow0FilledOnlyRow1) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    apply_placement(p, kColDown, 0, 5, board, ctx);
    EXPECT_FALSE(has_placement(ctx, p, kColDown, 0));
    EXPECT_TRUE(has_placement(ctx, p, kColDown, 1));
    for (int r = 2; r < kNumRows; ++r)
        EXPECT_FALSE(has_placement(ctx, p, kColDown, r));
}
TEST(LegalMoves, Down_AfterAllFilled_None) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    for (int r = 0; r < kNumRows; ++r)
        apply_placement(p, kColDown, r, 0, board, ctx);
    for (int r = 0; r < kNumRows; ++r)
        EXPECT_FALSE(has_placement(ctx, p, kColDown, r));
}

// ---------------------------------------------------------------------------
// Up column
// ---------------------------------------------------------------------------
TEST(LegalMoves, Up_FreshOnlyRow12) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    EXPECT_TRUE(has_placement(ctx, p, kColUp, 12));
    for (int r = 0; r < 12; ++r)
        EXPECT_FALSE(has_placement(ctx, p, kColUp, r));
}
TEST(LegalMoves, Up_AfterRow12FilledOnlyRow11) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    apply_placement(p, kColUp, 12, 0, board, ctx);
    EXPECT_FALSE(has_placement(ctx, p, kColUp, 12));
    EXPECT_TRUE(has_placement(ctx, p, kColUp, 11));
}

// ---------------------------------------------------------------------------
// Free column
// ---------------------------------------------------------------------------
TEST(LegalMoves, Free_FreshAllRows) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    for (int r = 0; r < kNumRows; ++r)
        EXPECT_TRUE(has_placement(ctx, p, kColFree, r));
    EXPECT_EQ(ctx.legal_all[p].count, 43);  // Down:1 + Free:13 + Up:1 + Mid:2 + Turbo:13 + UpDown:13
}
TEST(LegalMoves, Free_After1Fill_12Rows) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    apply_placement(p, kColFree, 5, 0, board, ctx);
    EXPECT_FALSE(has_placement(ctx, p, kColFree, 5));
    for (int r = 0; r < kNumRows; ++r) {
        if (r == 5) continue;
        EXPECT_TRUE(has_placement(ctx, p, kColFree, r));
    }
}

// ---------------------------------------------------------------------------
// Mid column
// ---------------------------------------------------------------------------
TEST(LegalMoves, Mid_FreshRows5And6) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    EXPECT_TRUE(has_placement(ctx, p, kColMid, 5));
    EXPECT_TRUE(has_placement(ctx, p, kColMid, 6));
    for (int r = 0; r < kNumRows; ++r) {
        if (r == 5 || r == 6) continue;
        EXPECT_FALSE(has_placement(ctx, p, kColMid, r));
    }
}
TEST(LegalMoves, Mid_AfterRow5_Rows4And6Legal) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    apply_placement(p, kColMid, 5, 0, board, ctx);
    EXPECT_TRUE(has_placement(ctx, p, kColMid, 4));
    EXPECT_TRUE(has_placement(ctx, p, kColMid, 6));
    EXPECT_FALSE(has_placement(ctx, p, kColMid, 5));
    EXPECT_FALSE(has_placement(ctx, p, kColMid, 3));
}
TEST(LegalMoves, Mid_WrapAround) {
    // Fill rows 5,6,7,8,9,10,11,12 — row 0 should then be legal (wraps from 12)
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    // Start from row 5, fill outward one side (toward row 12) and toward row 0
    apply_placement(p, kColMid, 5, 0, board, ctx);
    apply_placement(p, kColMid, 6, 0, board, ctx);
    apply_placement(p, kColMid, 7, 0, board, ctx);
    apply_placement(p, kColMid, 8, 0, board, ctx);
    apply_placement(p, kColMid, 9, 0, board, ctx);
    apply_placement(p, kColMid, 10, 0, board, ctx);
    apply_placement(p, kColMid, 11, 0, board, ctx);
    apply_placement(p, kColMid, 12, 0, board, ctx);
    // Row 12 is filled; row 0 is empty and wraps to row 12 (filled) → row 0 legal
    EXPECT_TRUE(has_placement(ctx, p, kColMid, 0));
    EXPECT_TRUE(has_placement(ctx, p, kColMid, 4));  // neighbour of 5
}

// ---------------------------------------------------------------------------
// UpDown column
// ---------------------------------------------------------------------------
TEST(LegalMoves, UpDown_FreshAllRows) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    for (int r = 0; r < kNumRows; ++r)
        EXPECT_TRUE(has_placement(ctx, p, kColUpDown, r));
}
TEST(LegalMoves, UpDown_AfterRow7_OnlyNeighbours) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    apply_placement(p, kColUpDown, 7, 0, board, ctx);
    EXPECT_TRUE(has_placement(ctx, p, kColUpDown, 6));
    EXPECT_TRUE(has_placement(ctx, p, kColUpDown, 8));
    for (int r = 0; r < kNumRows; ++r) {
        if (r == 6 || r == 8) continue;
        EXPECT_FALSE(has_placement(ctx, p, kColUpDown, r));
    }
}
TEST(LegalMoves, UpDown_WrapRow0) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    apply_placement(p, kColUpDown, 0, 0, board, ctx);
    EXPECT_TRUE(has_placement(ctx, p, kColUpDown, 1));
    EXPECT_TRUE(has_placement(ctx, p, kColUpDown, 12));  // wraps
}
TEST(LegalMoves, UpDown_WrapRow12) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    apply_placement(p, kColUpDown, 12, 0, board, ctx);
    EXPECT_TRUE(has_placement(ctx, p, kColUpDown, 11));
    EXPECT_TRUE(has_placement(ctx, p, kColUpDown, 0));  // wraps
}

// ---------------------------------------------------------------------------
// Turbo — behaves like Free for legal moves
// ---------------------------------------------------------------------------
TEST(LegalMoves, Turbo_FreshAllRows) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    int p = board.current_player;
    for (int r = 0; r < kNumRows; ++r)
        EXPECT_TRUE(has_placement(ctx, p, kColTurbo, r));
}
