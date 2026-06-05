#include <gtest/gtest.h>

#include <array>
#include <cstring>

#include "engine/board_init.h"
#include "engine/board_state.h"
#include "engine/constants.h"
#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_rules.h"
#include "engine/game_state.h"
#include "engine/placement.h"
#include "engine/rng.h"
#include "engine/scoring.h"

// ---------------------------------------------------------------------------
// "Lucky Yams" first-roll bonus.
// ---------------------------------------------------------------------------

namespace {

void make_fresh(BoardState& board, GameContext& ctx) {
    RNG rng(42);
    init_board(board, rng);
    init_context(ctx, board);
}

std::array<int8_t, 5> d(int a, int b, int c, int e, int f) {
    return {(int8_t)a, (int8_t)b, (int8_t)c, (int8_t)e, (int8_t)f};
}

// RAII helper: restore the default (bonus ON) rules after a test mutates them.
struct RulesGuard {
    ~RulesGuard() { set_game_rules(GameRules{}); }
};

}  // namespace

// --- is_five_of_a_kind -----------------------------------------------------

TEST(LuckyYams, FiveOfAKind) {
    auto yes = d(4, 4, 4, 4, 4);
    auto no  = d(4, 4, 4, 4, 3);
    EXPECT_TRUE(is_five_of_a_kind(yes.data()));
    EXPECT_FALSE(is_five_of_a_kind(no.data()));
}

// --- yams_bonus_active gating ----------------------------------------------

TEST(LuckyYams, BonusActiveGating) {
    RulesGuard guard;
    set_game_rules(GameRules{/*yams_first_roll_bonus=*/true});

    GameState gs{};
    auto yams = d(3, 3, 3, 3, 3);
    std::memcpy(gs.dice, yams.data(), sizeof(gs.dice));

    gs.rolls_left = 2;
    EXPECT_TRUE(yams_bonus_active(gs));            // first roll + yams + enabled

    gs.rolls_left = 1;
    EXPECT_FALSE(yams_bonus_active(gs));           // after a reroll → forfeited
    gs.rolls_left = 0;
    EXPECT_FALSE(yams_bonus_active(gs));

    gs.rolls_left = 2;
    auto mixed = d(3, 3, 3, 3, 2);
    std::memcpy(gs.dice, mixed.data(), sizeof(gs.dice));
    EXPECT_FALSE(yams_bonus_active(gs));           // not a yams

    std::memcpy(gs.dice, yams.data(), sizeof(gs.dice));
    set_game_rules(GameRules{/*yams_first_roll_bonus=*/false});
    EXPECT_FALSE(yams_bonus_active(gs));           // rule disabled
}

// --- wildcard maxima per row -----------------------------------------------

TEST(LuckyYams, WildcardRowMaxima) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    const int p = 0, c = kColFree;

    EXPECT_EQ(calculate_yams_bonus_score(kRow1s, p, c, board, ctx), 5);
    EXPECT_EQ(calculate_yams_bonus_score(kRow6s, p, c, board, ctx), 30);
    EXPECT_EQ(calculate_yams_bonus_score(kRowSS, p, c, board, ctx), 29);
    EXPECT_EQ(calculate_yams_bonus_score(kRowLS, p, c, board, ctx), 30);
    EXPECT_EQ(calculate_yams_bonus_score(kRowFH, p, c, board, ctx), 50);
    EXPECT_EQ(calculate_yams_bonus_score(kRowK,  p, c, board, ctx), 54);
    EXPECT_EQ(calculate_yams_bonus_score(kRowSTR, p, c, board, ctx), 50);
    EXPECT_EQ(calculate_yams_bonus_score(kRowU8, p, c, board, ctx), 75);
    EXPECT_EQ(calculate_yams_bonus_score(kRowY,  p, c, board, ctx), 100);
}

// --- SS/LS interlock + Golden Rule under the wildcard ----------------------

TEST(LuckyYams, SsCappedBelowFilledLs) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    const int p = 0, c = kColFree;

    // LS already 25 in this column → SS wildcard must be the largest legal
    // value strictly below 25, i.e. 24.
    board.cells[p][c][kRowLS] = 25;
    ctx.golden_max[c][kRowLS] = 25;

    EXPECT_EQ(calculate_yams_bonus_score(kRowSS, p, c, board, ctx), 24);
}

TEST(LuckyYams, LsScratchedWhenSsScratched) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    const int p = 0, c = kColFree;

    ctx.ss_scratched[p][c] = true;   // SS scratched forces LS to scratch
    EXPECT_EQ(calculate_yams_bonus_score(kRowLS, p, c, board, ctx), 0);

    GameContext ctx2; BoardState b2; make_fresh(b2, ctx2);
    ctx2.ls_scratched[p][c] = true;  // LS scratched forces SS to scratch
    EXPECT_EQ(calculate_yams_bonus_score(kRowSS, p, c, b2, ctx2), 0);
}

TEST(LuckyYams, GoldenRuleStillApplies) {
    BoardState board; GameContext ctx; make_fresh(board, ctx);
    const int p = 0, c = kColFree;

    // A higher SS recorded elsewhere raises the LS floor: LS must be strictly
    // greater than the global max SS. Max SS = 29 → LS wildcard stays 30.
    ctx.golden_max[c][kRowSS] = 29;
    EXPECT_EQ(calculate_yams_bonus_score(kRowLS, p, c, board, ctx), 30);

    // Combined: LS filled at 22 and global max SS at 21 squeezes SS to 21..21
    // failing the < LS test only at 22; the surviving max legal SS is 21.
    BoardState b2; GameContext ctx2; make_fresh(b2, ctx2);
    b2.cells[p][c][kRowLS] = 22; ctx2.golden_max[c][kRowLS] = 22;
    ctx2.golden_max[c][kRowSS] = 21;  // SS must be >= 21 (golden) and < 22
    EXPECT_EQ(calculate_yams_bonus_score(kRowSS, p, c, b2, ctx2), 21);
}

// --- end-to-end placement records the wildcard score -----------------------

TEST(LuckyYams, PerformPlacementWritesWildcard) {
    RulesGuard guard;
    set_game_rules(GameRules{/*yams_first_roll_bonus=*/true});

    GameState gs{}; GameContext ctx;
    RNG rng(7);
    init_game(gs, ctx, rng);

    const int p = gs.board.current_player;
    auto yams = d(3, 3, 3, 3, 3);          // five 3s
    std::memcpy(gs.dice, yams.data(), sizeof(gs.dice));
    gs.rolls_left = 2;                      // first roll

    // Place in the 6s box: normally five 3s score 0 there, but the wildcard
    // writes the row max (30).
    int score = perform_placement(gs, ctx, kColFree, kRow6s, rng);
    EXPECT_EQ(score, 30);
    EXPECT_EQ(gs.board.cells[p][kColFree][kRow6s], 30);
}

TEST(LuckyYams, PerformPlacementBonusOffScoresNormally) {
    RulesGuard guard;
    set_game_rules(GameRules{/*yams_first_roll_bonus=*/false});

    GameState gs{}; GameContext ctx;
    RNG rng(7);
    init_game(gs, ctx, rng);

    const int p = gs.board.current_player;
    auto yams = d(3, 3, 3, 3, 3);
    std::memcpy(gs.dice, yams.data(), sizeof(gs.dice));
    gs.rolls_left = 2;

    // With the bonus disabled, five 3s in the 6s box is a scratch (0).
    int score = perform_placement(gs, ctx, kColFree, kRow6s, rng);
    EXPECT_EQ(score, 0);
    EXPECT_EQ(gs.board.cells[p][kColFree][kRow6s], 0);
}
