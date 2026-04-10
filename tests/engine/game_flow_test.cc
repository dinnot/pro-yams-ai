#include <gtest/gtest.h>
#include "engine/game_flow.h"
#include "engine/board_init.h"
#include "engine/scoring.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static bool dice_valid(const int8_t dice[kNumDice]) {
    for (int i = 0; i < kNumDice; ++i)
        if (dice[i] < 1 || dice[i] > kNumDieSides) return false;
    return true;
}

// ---------------------------------------------------------------------------
// init_game
// ---------------------------------------------------------------------------
TEST(GameFlow, InitGame_DiceValid) {
    RNG rng(1);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    EXPECT_TRUE(dice_valid(gs.dice));
}

TEST(GameFlow, InitGame_RollsLeft2) {
    RNG rng(2);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    EXPECT_EQ(gs.rolls_left, 2);
}

TEST(GameFlow, InitGame_BoardEmpty) {
    RNG rng(3);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    EXPECT_EQ(gs.board.cells_filled, 0);
}

// ---------------------------------------------------------------------------
// start_turn
// ---------------------------------------------------------------------------
TEST(GameFlow, StartTurn_DiceValid) {
    RNG rng(10);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    start_turn(gs, rng);
    EXPECT_TRUE(dice_valid(gs.dice));
}

TEST(GameFlow, StartTurn_SetsRollsLeft2) {
    RNG rng(11);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    gs.rolls_left = 0;
    start_turn(gs, rng);
    EXPECT_EQ(gs.rolls_left, 2);
}

// ---------------------------------------------------------------------------
// can_reroll
// ---------------------------------------------------------------------------
TEST(GameFlow, CanReroll_TrueWhenRollsLeft2) {
    RNG rng(20);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    EXPECT_TRUE(can_reroll(gs, ctx));
}

TEST(GameFlow, CanReroll_TrueWhenRollsLeft1) {
    RNG rng(21);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    gs.rolls_left = 1;
    EXPECT_TRUE(can_reroll(gs, ctx));
}

TEST(GameFlow, CanReroll_FalseWhenRollsLeft0) {
    RNG rng(22);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    gs.rolls_left = 0;
    EXPECT_FALSE(can_reroll(gs, ctx));
}

// ---------------------------------------------------------------------------
// perform_reroll
// ---------------------------------------------------------------------------
TEST(GameFlow, PerformReroll_DecrementsRollsLeft) {
    RNG rng(30);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    EXPECT_EQ(gs.rolls_left, 2);
    perform_reroll(gs, 0, rng);
    EXPECT_EQ(gs.rolls_left, 1);
    perform_reroll(gs, 0, rng);
    EXPECT_EQ(gs.rolls_left, 0);
}

TEST(GameFlow, PerformReroll_NoOpAtZero) {
    RNG rng(31);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    gs.rolls_left = 0;
    int8_t before[kNumDice];
    for (int i = 0; i < kNumDice; ++i) before[i] = gs.dice[i];
    perform_reroll(gs, 0, rng);
    EXPECT_EQ(gs.rolls_left, 0);
    for (int i = 0; i < kNumDice; ++i) EXPECT_EQ(gs.dice[i], before[i]);
}

TEST(GameFlow, PerformReroll_HeldDiceUnchanged) {
    // Hold all dice (mask = 0b11111 = 31) — no dice should change
    RNG rng(32);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    int8_t before[kNumDice];
    for (int i = 0; i < kNumDice; ++i) before[i] = gs.dice[i];
    perform_reroll(gs, 0x1F, rng);  // keep all 5 dice
    for (int i = 0; i < kNumDice; ++i) EXPECT_EQ(gs.dice[i], before[i]);
}

TEST(GameFlow, PerformReroll_UnheldDiceRerolled_RangeValid) {
    // Hold dice 0 only (mask=1), reroll dice 1-4
    RNG rng(33);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    int8_t held = gs.dice[0];
    perform_reroll(gs, 0x01, rng);  // keep dice[0]
    EXPECT_EQ(gs.dice[0], held);
    EXPECT_TRUE(dice_valid(gs.dice));
}

TEST(GameFlow, PerformReroll_DiceStillValid) {
    RNG rng(34);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    perform_reroll(gs, 0, rng);
    EXPECT_TRUE(dice_valid(gs.dice));
    perform_reroll(gs, 0, rng);
    EXPECT_TRUE(dice_valid(gs.dice));
}

// ---------------------------------------------------------------------------
// get_legal_placements — roll-aware Turbo gating
// ---------------------------------------------------------------------------
TEST(GameFlow, GetLegalPlacements_RollsLeft2_TurboIncluded) {
    RNG rng(40);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    ASSERT_EQ(gs.rolls_left, 2);
    const auto& cache = get_legal_placements(gs, ctx);
    int p = gs.board.current_player;
    bool any_turbo = false;
    for (int r = 0; r < kNumRows; ++r)
        if (cache.is_legal[kColTurbo][r]) { any_turbo = true; break; }
    EXPECT_TRUE(any_turbo);
    EXPECT_EQ(cache.count, ctx.legal_all[p].count);
}

TEST(GameFlow, GetLegalPlacements_RollsLeft0_NoTurbo) {
    RNG rng(41);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    gs.rolls_left = 0;
    int p = gs.board.current_player;
    const auto& cache = get_legal_placements(gs, ctx);
    for (int r = 0; r < kNumRows; ++r)
        EXPECT_FALSE(cache.is_legal[kColTurbo][r]);
    EXPECT_EQ(cache.count, ctx.legal_no_turbo[p].count);
}

// ---------------------------------------------------------------------------
// perform_placement
// ---------------------------------------------------------------------------
TEST(GameFlow, PerformPlacement_SwitchesPlayer) {
    RNG rng(50);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    int first_player = gs.board.current_player;
    int sc = calculate_score(0, gs.dice, first_player, kColDown, gs.board, ctx);
    // Place in Down column, row 0 (always legal on fresh board)
    perform_placement(gs, ctx, kColDown, 0, rng);
    EXPECT_NE(gs.board.current_player, first_player);
}

TEST(GameFlow, PerformPlacement_CellFilled) {
    RNG rng(51);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    int p = gs.board.current_player;
    int sc = calculate_score(0, gs.dice, p, kColDown, gs.board, ctx);
    perform_placement(gs, ctx, kColDown, 0, rng);
    EXPECT_NE(gs.board.cells[p][kColDown][0], kCellEmpty);
}

TEST(GameFlow, PerformPlacement_CellsFilledIncremented) {
    RNG rng(52);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    EXPECT_EQ(gs.board.cells_filled, 0);
    int sc = calculate_score(0, gs.dice, gs.board.current_player, kColDown, gs.board, ctx);
    perform_placement(gs, ctx, kColDown, 0, rng);
    EXPECT_EQ(gs.board.cells_filled, 1);
}

TEST(GameFlow, PerformPlacement_NewDiceRolled) {
    RNG rng(53);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    int sc = calculate_score(0, gs.dice, gs.board.current_player, kColDown, gs.board, ctx);
    perform_placement(gs, ctx, kColDown, 0, rng);
    // After placement, new player's dice should be valid and rolls_left==2
    EXPECT_TRUE(dice_valid(gs.dice));
    EXPECT_EQ(gs.rolls_left, 2);
}

TEST(GameFlow, PerformPlacement_ReturnsScore) {
    // Score returned must be >= 0 (0 = scratch or golden rule blocked)
    RNG rng(54);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    int sc = calculate_score(0, gs.dice, gs.board.current_player, kColDown, gs.board, ctx);
    int score = perform_placement(gs, ctx, kColDown, 0, rng);
    EXPECT_GE(score, 0);
}
