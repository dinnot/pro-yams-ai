#include <gtest/gtest.h>
#include "engine/game_flow.h"
#include "engine/board_init.h"
#include "engine/scoring.h"

// ---------------------------------------------------------------------------
// Turbo column rules:
//   - Turbo is limited to 2 rolls max (available when rolls_left > 0)
//   - get_legal_placements returns legal_all (includes Turbo) when rolls_left > 0
//   - get_legal_placements returns legal_no_turbo when rolls_left == 0
//   - legal_no_turbo never contains kColTurbo
//   - legal_all always contains kColTurbo on a fresh board
// ---------------------------------------------------------------------------

static void make_fresh(GameState& gs, GameContext& ctx, RNG& rng) {
    init_game(gs, ctx, rng);
}

TEST(Turbo, LegalAll_ContainsTurbo_FreshBoard) {
    RNG rng(100);
    GameState gs; GameContext ctx;
    make_fresh(gs, ctx, rng);
    int p = gs.board.current_player;
    bool any_turbo = false;
    for (int r = 0; r < kNumRows; ++r)
        if (ctx.legal_all[p].is_legal[kColTurbo][r]) { any_turbo = true; break; }
    EXPECT_TRUE(any_turbo);
}

TEST(Turbo, LegalNoTurbo_NeverContainsTurbo_FreshBoard) {
    RNG rng(101);
    GameState gs; GameContext ctx;
    make_fresh(gs, ctx, rng);
    int p = gs.board.current_player;
    for (int r = 0; r < kNumRows; ++r)
        EXPECT_FALSE(ctx.legal_no_turbo[p].is_legal[kColTurbo][r]);
}

TEST(Turbo, GetLegalPlacements_RollsLeft2_TurboIncluded) {
    RNG rng(102);
    GameState gs; GameContext ctx;
    make_fresh(gs, ctx, rng);
    ASSERT_EQ(gs.rolls_left, 2);
    const auto& cache = get_legal_placements(gs, ctx);
    bool any_turbo = false;
    for (int r = 0; r < kNumRows; ++r)
        if (cache.is_legal[kColTurbo][r]) { any_turbo = true; break; }
    EXPECT_TRUE(any_turbo);
}

TEST(Turbo, GetLegalPlacements_RollsLeft1_TurboIncluded) {
    RNG rng(103);
    GameState gs; GameContext ctx;
    make_fresh(gs, ctx, rng);
    perform_reroll(gs, 0, rng);
    ASSERT_EQ(gs.rolls_left, 1);
    const auto& cache = get_legal_placements(gs, ctx);
    bool any_turbo = false;
    for (int r = 0; r < kNumRows; ++r)
        if (cache.is_legal[kColTurbo][r]) { any_turbo = true; break; }
    EXPECT_TRUE(any_turbo);
}

TEST(Turbo, GetLegalPlacements_RollsLeft0_NoTurbo) {
    RNG rng(104);
    GameState gs; GameContext ctx;
    make_fresh(gs, ctx, rng);
    perform_reroll(gs, 0, rng);  // rolls_left=1
    perform_reroll(gs, 0, rng);  // rolls_left=0
    ASSERT_EQ(gs.rolls_left, 0);
    const auto& cache = get_legal_placements(gs, ctx);
    for (int r = 0; r < kNumRows; ++r)
        EXPECT_FALSE(cache.is_legal[kColTurbo][r]);
}

TEST(Turbo, TurboPlacement_ValidEarly) {
    RNG rng(105);
    GameState gs; GameContext ctx;
    make_fresh(gs, ctx, rng);
    int p = gs.board.current_player;
    perform_reroll(gs, 0, rng);
    ASSERT_EQ(gs.rolls_left, 1);
    int sc = calculate_score(0, gs.dice, p, kColTurbo, gs.board, ctx);
    perform_placement(gs, ctx, kColTurbo, 0, rng);
    EXPECT_NE(gs.board.cells[p][kColTurbo][0], kCellEmpty);
}

TEST(Turbo, CanReroll_TurboOnlyEdgeCase) {
    RNG rng(108);
    GameState gs; GameContext ctx;
    make_fresh(gs, ctx, rng);
    int p = gs.board.current_player;
    ctx.non_turbo_cells_remaining[p] = 0;

    gs.rolls_left = 2;
    EXPECT_TRUE(can_reroll(gs, ctx));

    gs.rolls_left = 1;
    EXPECT_FALSE(can_reroll(gs, ctx));
}

TEST(Turbo, TurboColumn_NotIn_LegalNoTurbo_AfterPlacements) {
    // Fill a few non-Turbo cells and verify legal_no_turbo still excludes Turbo
    RNG rng(106);
    GameState gs; GameContext ctx;
    make_fresh(gs, ctx, rng);
    for (int i = 0; i < 5; ++i) {
        int p = gs.board.current_player;
        int sc = calculate_score(i, gs.dice, gs.board.current_player,
                                 kColFree, gs.board, ctx);
        perform_placement(gs, ctx, kColFree, i, rng);
        (void)p;
    }
    for (int pl = 0; pl < kNumPlayers; ++pl)
        for (int r = 0; r < kNumRows; ++r)
            EXPECT_FALSE(ctx.legal_no_turbo[pl].is_legal[kColTurbo][r]);
}

TEST(Turbo, LegalAll_TurboCount_FreshBoard) {
    // On a fresh board, all 13 Turbo rows should be in legal_all
    RNG rng(107);
    GameState gs; GameContext ctx;
    make_fresh(gs, ctx, rng);
    int p = gs.board.current_player;
    int turbo_count = 0;
    for (int r = 0; r < kNumRows; ++r)
        if (ctx.legal_all[p].is_legal[kColTurbo][r]) ++turbo_count;
    EXPECT_EQ(turbo_count, kNumRows);
}
