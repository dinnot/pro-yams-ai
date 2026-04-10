#include <gtest/gtest.h>
#include "engine/scoring.h"
#include "engine/board_state.h"
#include "engine/game_context.h"
#include "engine/board_init.h"
#include "engine/rng.h"
#include "engine/solver_tables.h"
#include "engine/placement.h"

// Helper: create a clean board+context pair
static void make_fresh(BoardState& board, GameContext& ctx) {
    RNG rng(42);
    init_board(board, rng);
    init_context(ctx, board);
}

// Helper: dice array
static std::array<int8_t,5> d(int a,int b,int c,int d_,int e) {
    return {(int8_t)a,(int8_t)b,(int8_t)c,(int8_t)d_,(int8_t)e};
}

// ---------------------------------------------------------------------------
// Number rows 0-5
// ---------------------------------------------------------------------------
TEST(Scoring, NumberRow_ThreeThrees) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(3,3,3,1,2);
    EXPECT_EQ(calculate_score(kRow3s, dice.data(), 0, kColFree, board, ctx), 9);
}
TEST(Scoring, NumberRow_FiveSixes) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(6,6,6,6,6);
    EXPECT_EQ(calculate_score(kRow6s, dice.data(), 0, kColFree, board, ctx), 30);
}
TEST(Scoring, NumberRow_NoSixes) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(1,2,3,4,5);
    EXPECT_EQ(calculate_score(kRow6s, dice.data(), 0, kColFree, board, ctx), 0);
}
TEST(Scoring, NumberRow_FiveOnes) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(1,1,1,1,1);
    EXPECT_EQ(calculate_score(kRow1s, dice.data(), 0, kColFree, board, ctx), 5);
}

// ---------------------------------------------------------------------------
// SS (row 6)
// ---------------------------------------------------------------------------
TEST(Scoring, SS_Sum20) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(4,4,4,4,4);  // sum=20
    EXPECT_EQ(calculate_score(kRowSS, dice.data(), 0, kColFree, board, ctx), 20);
}
TEST(Scoring, SS_Sum15_Below20) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(3,3,3,3,3);  // sum=15
    EXPECT_EQ(calculate_score(kRowSS, dice.data(), 0, kColFree, board, ctx), 0);
}
TEST(Scoring, SS_Sum30_Capped29) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(6,6,6,6,6);  // sum=30, SS capped at 29
    EXPECT_EQ(calculate_score(kRowSS, dice.data(), 0, kColFree, board, ctx), 29);
}
TEST(Scoring, SS_WhenLSScratched) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    ctx.ls_scratched[0][kColFree] = true;
    auto dice = d(4,5,5,5,5);  // sum=24
    EXPECT_EQ(calculate_score(kRowSS, dice.data(), 0, kColFree, board, ctx), 0);
}
TEST(Scoring, SS_BelowOwnLS) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    // LS already filled at 25
    board.cells[0][kColFree][kRowLS] = 25;
    // SS with raw=24 is valid (24 < 25)
    auto dice = d(4,4,4,6,6);  // sum=24
    EXPECT_EQ(calculate_score(kRowSS, dice.data(), 0, kColFree, board, ctx), 24);
    // SS with raw=25 is invalid (25 >= 25)
    auto dice2 = d(5,5,5,5,5);  // sum=25
    EXPECT_EQ(calculate_score(kRowSS, dice2.data(), 0, kColFree, board, ctx), 0);
}

// ---------------------------------------------------------------------------
// LS (row 7)
// ---------------------------------------------------------------------------
TEST(Scoring, LS_Sum30) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(6,6,6,6,6);
    EXPECT_EQ(calculate_score(kRowLS, dice.data(), 0, kColFree, board, ctx), 30);
}
TEST(Scoring, LS_Sum15_Below20) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(3,3,3,3,3);
    EXPECT_EQ(calculate_score(kRowLS, dice.data(), 0, kColFree, board, ctx), 0);
}
TEST(Scoring, LS_WhenSSScratched) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    ctx.ss_scratched[0][kColFree] = true;
    auto dice = d(5,5,5,5,5);
    EXPECT_EQ(calculate_score(kRowLS, dice.data(), 0, kColFree, board, ctx), 0);
}
TEST(Scoring, LS_AboveOwnSS) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    // SS filled at 24
    board.cells[0][kColFree][kRowSS] = 24;
    ctx.golden_max[kColFree][kRowSS] = 24;
    // LS=25 valid (25 > 24)
    auto dice = d(5,5,5,5,5);  // sum=25
    EXPECT_EQ(calculate_score(kRowLS, dice.data(), 0, kColFree, board, ctx), 25);
    // LS=24 invalid (24 <= 24)
    auto dice2 = d(4,4,6,6,4);  // sum=24
    EXPECT_EQ(calculate_score(kRowLS, dice2.data(), 0, kColFree, board, ctx), 0);
}

TEST(Scoring, LS_AboveOpponentSS) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    // Opponent (player 1) SS filled at 26
    board.cells[1][kColFree][kRowSS] = 26;
    ctx.golden_max[kColFree][kRowSS] = 26;
    // LS=27 valid (27 > 26)
    auto dice_valid = d(5,5,5,6,6);
    EXPECT_EQ(calculate_score(kRowLS, dice_valid.data(), 0, kColFree, board, ctx), 27);
    // LS=26 invalid (26 <= 26)
    auto dice_invalid = d(4,4,6,6,6);
    EXPECT_EQ(calculate_score(kRowLS, dice_invalid.data(), 0, kColFree, board, ctx), 0);
}

// ---------------------------------------------------------------------------
// FH (row 8)
// ---------------------------------------------------------------------------
TEST(Scoring, FH_Valid) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(3,3,3,2,2);  // sum=13, score=33
    EXPECT_EQ(calculate_score(kRowFH, dice.data(), 0, kColFree, board, ctx), 33);
}
TEST(Scoring, FH_YamsQualifies) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(5,5,5,5,5);  // sum=25, score=45
    EXPECT_EQ(calculate_score(kRowFH, dice.data(), 0, kColFree, board, ctx), 45);
}
TEST(Scoring, FH_Invalid) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(1,2,3,4,5);
    EXPECT_EQ(calculate_score(kRowFH, dice.data(), 0, kColFree, board, ctx), 0);
}

// ---------------------------------------------------------------------------
// K (row 9)
// ---------------------------------------------------------------------------
TEST(Scoring, K_FourFours) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(4,4,4,4,1);  // score = 30+16=46
    EXPECT_EQ(calculate_score(kRowK, dice.data(), 0, kColFree, board, ctx), 46);
}
TEST(Scoring, K_YamsQualifies) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(6,6,6,6,6);  // score = 30+24=54
    EXPECT_EQ(calculate_score(kRowK, dice.data(), 0, kColFree, board, ctx), 54);
}
TEST(Scoring, K_FHInvalid) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(3,3,3,2,2);
    EXPECT_EQ(calculate_score(kRowK, dice.data(), 0, kColFree, board, ctx), 0);
}

// ---------------------------------------------------------------------------
// STR (row 10)
// ---------------------------------------------------------------------------
TEST(Scoring, STR_Small) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(1,2,3,4,5);
    EXPECT_EQ(calculate_score(kRowSTR, dice.data(), 0, kColFree, board, ctx), 45);
}
TEST(Scoring, STR_Large) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(2,3,4,5,6);
    EXPECT_EQ(calculate_score(kRowSTR, dice.data(), 0, kColFree, board, ctx), 50);
}
TEST(Scoring, STR_Invalid) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(1,2,3,4,6);
    EXPECT_EQ(calculate_score(kRowSTR, dice.data(), 0, kColFree, board, ctx), 0);
}

// ---------------------------------------------------------------------------
// U8 (row 11)
// ---------------------------------------------------------------------------
TEST(Scoring, U8_Sum5) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(1,1,1,1,1);  // sum=5, score=60+15=75
    EXPECT_EQ(calculate_score(kRowU8, dice.data(), 0, kColFree, board, ctx), 75);
}
TEST(Scoring, U8_Sum8) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(1,1,1,1,4);  // sum=8, score=60
    EXPECT_EQ(calculate_score(kRowU8, dice.data(), 0, kColFree, board, ctx), 60);
}
TEST(Scoring, U8_Sum9_Invalid) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(1,1,1,2,4);  // sum=9
    EXPECT_EQ(calculate_score(kRowU8, dice.data(), 0, kColFree, board, ctx), 0);
}

// ---------------------------------------------------------------------------
// Y (row 12)
// ---------------------------------------------------------------------------
TEST(Scoring, Y_Sixes) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(6,6,6,6,6);  // score=75+25=100
    EXPECT_EQ(calculate_score(kRowY, dice.data(), 0, kColFree, board, ctx), 100);
}
TEST(Scoring, Y_Ones) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(1,1,1,1,1);  // score=75
    EXPECT_EQ(calculate_score(kRowY, dice.data(), 0, kColFree, board, ctx), 75);
}
TEST(Scoring, Y_Invalid) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    auto dice = d(3,3,3,3,2);
    EXPECT_EQ(calculate_score(kRowY, dice.data(), 0, kColFree, board, ctx), 0);
}

// ---------------------------------------------------------------------------
// Golden Rule
// ---------------------------------------------------------------------------
TEST(Scoring, GoldenRule_BelowThreshold) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    ctx.golden_max[kColFree][kRow3s] = 12;
    auto dice = d(3,3,3,1,1);  // score for 3s = 9
    EXPECT_EQ(calculate_score(kRow3s, dice.data(), 0, kColFree, board, ctx), 0);
}
TEST(Scoring, GoldenRule_MeetsThreshold) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    ctx.golden_max[kColFree][kRow3s] = 9;
    auto dice = d(3,3,3,1,1);  // score=9 >= 9
    EXPECT_EQ(calculate_score(kRow3s, dice.data(), 0, kColFree, board, ctx), 9);
}
