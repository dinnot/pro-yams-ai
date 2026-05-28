#include "engine/board_init.h"
#include "engine/constants.h"
#include "engine/game_traits.h"
#include "engine/placement.h"
#include "engine/rng.h"
#include "engine/scoring.h"

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// 2v2-only regression tests covering the Golden Rule (Sub-task 2.4).
//
// In 2v2 the Golden Rule applies *globally* — including the teammate. These
// tests pin that behavior so a future refactor that narrows golden_max to only
// loop over opponents will fail loudly.
//
// Seating (clockwise): P0 → P1 → P2 → P3. Teams: {P0, P2} and {P1, P3}.
// ---------------------------------------------------------------------------

namespace {

// Helper: build a fresh 2v2 board + context.
void make_empty_2v2(BoardState2v2& board, GameContext2v2& ctx) {
    RNG rng(42);
    init_board<Yams2v2>(board, rng);
    init_context<Yams2v2>(ctx, board);
}

}  // namespace

TEST(GoldenRuleTeammate2v2, TeammateScoreRaisesOwnThreshold) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_empty_2v2(board, ctx);

    // Teammate (P2) places four 6s (score 24) in the Free column's "6s" row.
    apply_placement<Yams2v2>(/*player=*/2, kColFree, kRow6s, /*score=*/24, board, ctx);
    ASSERT_EQ(ctx.golden_max[kColFree][kRow6s], 24);

    // P0 (Team 0, same team as P2) rolls only three 6s + two 1s. Raw score = 18.
    // Golden Rule must block this: 18 < 24.
    int8_t dice_three_sixes[5] = {6, 6, 6, 1, 1};
    int score_p0 = calculate_score<Yams2v2>(kRow6s, dice_three_sixes,
                                            /*player=*/0, kColFree, board, ctx);
    EXPECT_EQ(score_p0, 0)
        << "P0 must scratch — teammate's 24 raised the bar above P0's 18";

    // Symmetric check: P1 (Team 1, opposing team) is also blocked by the same
    // threshold — the Golden Rule does not distinguish team membership.
    int score_p1 = calculate_score<Yams2v2>(kRow6s, dice_three_sixes,
                                            /*player=*/1, kColFree, board, ctx);
    EXPECT_EQ(score_p1, 0);

    // P3 (Team 1) with four 6s themselves *can* match the threshold.
    int8_t dice_four_sixes[5] = {6, 6, 6, 6, 1};
    int score_p3 = calculate_score<Yams2v2>(kRow6s, dice_four_sixes,
                                            /*player=*/3, kColFree, board, ctx);
    EXPECT_EQ(score_p3, 24);
}

TEST(GoldenRuleTeammate2v2, OpponentScoreAlsoRaisesThreshold) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_empty_2v2(board, ctx);

    // Sanity baseline: an opponent (P1) sets the bar — the existing 1v1
    // behavior must still hold after templatization.
    apply_placement<Yams2v2>(/*player=*/1, kColFree, kRow6s, /*score=*/24, board, ctx);

    int8_t dice_three_sixes[5] = {6, 6, 6, 1, 1};
    int score_p0 = calculate_score<Yams2v2>(kRow6s, dice_three_sixes,
                                            /*player=*/0, kColFree, board, ctx);
    EXPECT_EQ(score_p0, 0);
}

TEST(GoldenRuleTeammate2v2, LsInterlock_AnyoneIncludesTeammateSS) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_empty_2v2(board, ctx);

    // Teammate (P2) places SS = 25 in the Free column. (Five 5s sums to 25.)
    apply_placement<Yams2v2>(/*player=*/2, kColFree, kRowSS, /*score=*/25, board, ctx);
    ASSERT_EQ(ctx.golden_max[kColFree][kRowSS], 25);

    // P0 tries to place LS = 25 in the same column. LS must be STRICTLY higher
    // than the highest SS recorded by *anyone* — including the teammate.
    int8_t dice_five_fives[5] = {5, 5, 5, 5, 5};
    int score_p0_ls_tie = calculate_score<Yams2v2>(kRowLS, dice_five_fives,
                                                   /*player=*/0, kColFree, board, ctx);
    EXPECT_EQ(score_p0_ls_tie, 0)
        << "LS must be strictly greater than teammate's SS — tie at 25 is invalid";

    // LS = 26 (six 4s? — no, can't roll 6 dice. Try 6+6+6+4+4 = 26.) clears the bar.
    int8_t dice_sum_26[5] = {6, 6, 6, 4, 4};
    int score_p0_ls_pass = calculate_score<Yams2v2>(kRowLS, dice_sum_26,
                                                    /*player=*/0, kColFree, board, ctx);
    EXPECT_EQ(score_p0_ls_pass, 26);
}

TEST(GoldenRuleTeammate2v2, GoldenMaxScansAllFourPlayers) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_empty_2v2(board, ctx);

    // All three other players (1, 2, 3) score increasing amounts in the same
    // cell. golden_max must reflect the global max after each placement.
    // (Using Down column: each placement targets a different row to avoid
    // the "cell already filled" assert. We use Free for the threshold test.)
    apply_placement<Yams2v2>(/*player=*/1, kColFree, kRow5s, /*score=*/5,  board, ctx);
    EXPECT_EQ(ctx.golden_max[kColFree][kRow5s], 5);

    apply_placement<Yams2v2>(/*player=*/2, kColDown, kRow5s, /*score=*/15, board, ctx);
    // golden_max is per (col, row) — kColFree[kRow5s] unchanged, kColDown[kRow5s] updated.
    EXPECT_EQ(ctx.golden_max[kColFree][kRow5s], 5);
    EXPECT_EQ(ctx.golden_max[kColDown][kRow5s], 15);

    // Now have P3 (cross-team opponent of P0) score higher than P1 in Free[5s].
    // The cell isn't filled by P0 or P2 yet so this is valid.
    apply_placement<Yams2v2>(/*player=*/3, kColFree, kRow5s, /*score=*/20, board, ctx);
    EXPECT_EQ(ctx.golden_max[kColFree][kRow5s], 20);
}
