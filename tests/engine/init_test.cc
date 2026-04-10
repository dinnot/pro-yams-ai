#include <gtest/gtest.h>
#include "engine/board_init.h"
#include "engine/rng.h"
#include "engine/constants.h"

#include <algorithm>
#include <array>

TEST(Init, AllCellsEmpty) {
    RNG rng(1);
    BoardState board; GameContext ctx;
    init_board(board, rng);
    init_context(ctx, board);
    for (int p = 0; p < kNumPlayers; ++p)
        for (int c = 0; c < kNumColumns; ++c)
            for (int r = 0; r < kNumRows; ++r)
                EXPECT_EQ(board.cells[p][c][r], kCellEmpty);
}

TEST(Init, CoefficientPermutation) {
    RNG rng(2);
    BoardState board; GameContext ctx;
    init_board(board, rng);
    init_context(ctx, board);
    std::array<int8_t,6> coeffs;
    for (int c = 0; c < kNumColumns; ++c) coeffs[c] = board.coefficients[c];
    std::sort(coeffs.begin(), coeffs.end());
    std::array<int8_t,6> expected = {8,10,12,14,16,18};
    EXPECT_EQ(coeffs, expected);
}

TEST(Init, CurrentPlayerValid) {
    for (int seed = 0; seed < 20; ++seed) {
        RNG rng(seed);
        BoardState board; GameContext ctx;
        init_board(board, rng);
        init_context(ctx, board);
        EXPECT_TRUE(board.current_player == 0 || board.current_player == 1);
    }
}

TEST(Init, CellsFilledZero) {
    RNG rng(3);
    BoardState board; GameContext ctx;
    init_board(board, rng);
    init_context(ctx, board);
    EXPECT_EQ(board.cells_filled, 0);
}

TEST(Init, GoldenMaxAllZero) {
    RNG rng(4);
    BoardState board; GameContext ctx;
    init_board(board, rng);
    init_context(ctx, board);
    for (int c = 0; c < kNumColumns; ++c)
        for (int r = 0; r < kNumRows; ++r)
            EXPECT_EQ(ctx.golden_max[c][r], 0);
}

TEST(Init, UpperSumAllZero) {
    RNG rng(5);
    BoardState board; GameContext ctx;
    init_board(board, rng);
    init_context(ctx, board);
    for (int p = 0; p < kNumPlayers; ++p)
        for (int c = 0; c < kNumColumns; ++c)
            EXPECT_EQ(ctx.upper_sum[p][c], 0);
}

TEST(Init, InitialLegalPlacements) {
    RNG rng(6);
    BoardState board; GameContext ctx;
    init_board(board, rng);
    init_context(ctx, board);
    for (int p = 0; p < kNumPlayers; ++p) {
        // Down: only row 0
        EXPECT_TRUE(ctx.legal_all[p].is_legal[kColDown][0]);
        for (int r = 1; r < kNumRows; ++r)
            EXPECT_FALSE(ctx.legal_all[p].is_legal[kColDown][r]);
        // Up: only row 12
        EXPECT_TRUE(ctx.legal_all[p].is_legal[kColUp][12]);
        for (int r = 0; r < 12; ++r)
            EXPECT_FALSE(ctx.legal_all[p].is_legal[kColUp][r]);
        // Free: all rows
        for (int r = 0; r < kNumRows; ++r)
            EXPECT_TRUE(ctx.legal_all[p].is_legal[kColFree][r]);
        // Mid: rows 5 and 6 only
        EXPECT_TRUE(ctx.legal_all[p].is_legal[kColMid][5]);
        EXPECT_TRUE(ctx.legal_all[p].is_legal[kColMid][6]);
        for (int r = 0; r < kNumRows; ++r) {
            if (r == 5 || r == 6) continue;
            EXPECT_FALSE(ctx.legal_all[p].is_legal[kColMid][r]);
        }
        // Turbo: all rows
        for (int r = 0; r < kNumRows; ++r)
            EXPECT_TRUE(ctx.legal_all[p].is_legal[kColTurbo][r]);
        // UpDown: all rows
        for (int r = 0; r < kNumRows; ++r)
            EXPECT_TRUE(ctx.legal_all[p].is_legal[kColUpDown][r]);
    }
}
