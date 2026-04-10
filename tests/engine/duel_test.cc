#include <gtest/gtest.h>
#include <cstring>
#include "engine/duel.h"
#include "engine/board_init.h"
#include "engine/rng.h"

// Helper: build a complete board with known cell values for testing
static void fill_all_cells(BoardState& board, GameContext& ctx, int player, int col,
                            int cell_total, int upper_sum_val) {
    // Clear and set a specific column: put cell_total in upper and lower rows
    // to achieve a specific raw score and upper_sum
    ctx.upper_sum[player][col] = (int16_t)upper_sum_val;
    int remaining = cell_total - upper_sum_val;  // cell sum excluding upper bonus
    // Distribute remaining across lower rows
    board.cells[player][col][kRowFH] = (int8_t)remaining;
    // Fill all empty cells with 0 to mark them filled
    for (int r = 0; r < kNumRows; ++r) {
        if (board.cells[player][col][r] == kCellEmpty) {
            board.cells[player][col][r] = 0;
            board.cells_filled++;
        }
    }
}

TEST(Duel, UpperSectionBonus) {
    EXPECT_EQ(upper_section_bonus(59), 0);
    EXPECT_EQ(upper_section_bonus(60), 30);
    EXPECT_EQ(upper_section_bonus(69), 30);
    EXPECT_EQ(upper_section_bonus(70), 50);
    EXPECT_EQ(upper_section_bonus(79), 50);
    EXPECT_EQ(upper_section_bonus(80), 100);
    EXPECT_EQ(upper_section_bonus(89), 100);
    EXPECT_EQ(upper_section_bonus(90), 200);
    EXPECT_EQ(upper_section_bonus(99), 200);
    EXPECT_EQ(upper_section_bonus(100), 500);
    EXPECT_EQ(upper_section_bonus(150), 500);
}

TEST(Duel, CrushMultiplier) {
    EXPECT_EQ(crush_multiplier(100, 0), 5);    // opponent = 0
    EXPECT_EQ(crush_multiplier(500, 100), 5);  // 5x
    EXPECT_EQ(crush_multiplier(400, 100), 4);  // 4x
    EXPECT_EQ(crush_multiplier(300, 100), 3);  // 3x
    EXPECT_EQ(crush_multiplier(200, 100), 2);  // 2x
    EXPECT_EQ(crush_multiplier(150, 100), 1);  // no crush
    EXPECT_EQ(crush_multiplier(0, 0), 1);      // both 0
}

TEST(Duel, SimpleDuel_NoCrush) {
    // Player 0: raw=210, Player 1: raw=110, col coeff=10, no crush, no clean bonus
    // 210 < 2*110=220 → no crush. diff=100. points=100*1*10=1000
    BoardState board;
    GameContext ctx;
    std::memset(&board, 0, sizeof(board));
    std::memset(&ctx, 0, sizeof(ctx));
    board.coefficients[0] = 10;
    for (int c = 1; c < kNumColumns; ++c) board.coefficients[c] = 8;
    board.cells_filled = 156;

    board.cells[0][0][kRowY]  = 100;
    board.cells[0][0][kRowFH] = 110;  // raw0 = 210
    board.cells[1][0][kRowY]  = 110;  // raw1 = 110

    int result = compute_duel(board, ctx);
    EXPECT_EQ(result, 1000);
}

TEST(Duel, CrushMultiplier_AppliedInDuel) {
    // Player 0: raw=250, Player 1: raw=100, coeff=10, crush=2x
    // Diff=150, points=150*2*10=3000
    BoardState board;
    GameContext ctx;
    std::memset(&board, 0, sizeof(board));
    std::memset(&ctx, 0, sizeof(ctx));
    board.coefficients[0] = 10;
    for (int c = 1; c < kNumColumns; ++c) board.coefficients[c] = 8;
    board.cells_filled = 156;

    board.cells[0][0][kRowY]  = 100;
    board.cells[0][0][kRowFH] = 100;  // 200 so far
    board.cells[0][0][kRowK]  = 50;   // 250 total
    board.cells[1][0][kRowY]  = 100;  // 100 total

    // 250 >= 2*100 → crush=2
    int result = compute_duel(board, ctx);
    EXPECT_EQ(result, (250-100)*2*10);
}

TEST(Duel, CleanColumnBonus_NocrushAdds200) {
    // Player 0 has clean column, no crush → +200
    BoardState board;
    GameContext ctx;
    std::memset(&board, 0, sizeof(board));
    std::memset(&ctx, 0, sizeof(ctx));
    board.coefficients[0] = 10;
    for (int c = 1; c < kNumColumns; ++c) board.coefficients[c] = 8;
    board.cells_filled = 156;

    board.cells[0][0][kRowY] = 100;
    board.cells[1][0][kRowY] = 90;  // 90 < 2*100, no crush on either side

    ctx.upper_sum[0][0] = 60;   // meets clean threshold
    ctx.lower_has_scratch[0][0] = false;  // no lower scratch → clean
    // Player 0 raw = 100 + upper_bonus(60)=100+30=130. Player 1 raw=90.
    // No crush (130 < 2*90). Clean bonus = 200. Adjusted[0]=130+200=330.
    // Diff = 330-90=240. Points=240*1*10=2400.
    int result = compute_duel(board, ctx);
    EXPECT_EQ(result, (130 + 200 - 90) * 1 * 10);
}

TEST(Duel, CleanColumnBonus_WithCrushAdds100) {
    BoardState board;
    GameContext ctx;
    std::memset(&board, 0, sizeof(board));
    std::memset(&ctx, 0, sizeof(ctx));
    board.coefficients[0] = 10;
    for (int c = 1; c < kNumColumns; ++c) board.coefficients[c] = 8;
    board.cells_filled = 156;

    board.cells[0][0][kRowY] = 100;
    board.cells[0][0][kRowFH]= 100;  // player 0 raw = 200
    board.cells[1][0][kRowY] = 90;   // player 1 raw = 90

    ctx.upper_sum[0][0] = 60;        // clean eligible
    ctx.lower_has_scratch[0][0] = false;
    // raw[0]=200+upper_bonus(60)=230. raw[1]=90.
    // 230 >= 2*90=180 → crush=2. clean bonus=100.
    // adjusted[0]=230+100=330. adjusted[1]=90. diff=240. points=240*2*10=4800.
    int result = compute_duel(board, ctx);
    EXPECT_EQ(result, (200 + 30 + 100 - 90) * 2 * 10);
}

TEST(Duel, Player1Wins_NegativeResult) {
    BoardState board;
    GameContext ctx;
    std::memset(&board, 0, sizeof(board));
    std::memset(&ctx, 0, sizeof(ctx));
    board.coefficients[0] = 10;
    for (int c = 1; c < kNumColumns; ++c) board.coefficients[c] = 8;
    board.cells_filled = 156;

    board.cells[0][0][kRowY] = 50;
    board.cells[1][0][kRowY] = 100;
    // raw[1]=100 >= 2*raw[0]=50 → crush=2 for player 1. active_crush=2.
    // diff = 50-100 = -50. points = -50*2*10 = -1000.
    int result = compute_duel(board, ctx);
    EXPECT_EQ(result, (50-100)*2*10);  // -1000
    EXPECT_LT(result, 0);
}
