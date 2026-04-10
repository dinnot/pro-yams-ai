#include <gtest/gtest.h>
#include "engine/game_flow.h"
#include "engine/scoring.h"
#include "engine/duel.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Play a complete game to terminal using random placements.
/// At each turn: use all rerolls (hold nothing), then place in first legal cell.
static void play_full_game(GameState& gs, GameContext& ctx, RNG& rng) {
    while (!is_terminal(gs.board)) {
        // Use all rerolls
        while (can_reroll(gs, ctx))
            perform_reroll(gs, 0, rng);
        // Place in first legal cell
        const auto& cache = get_legal_placements(gs, ctx);
        if (cache.count == 0) break;  // safety — should never happen in valid game
        auto pl = cache.placements[0];
        int score = calculate_score(pl.row, gs.dice,
                                    gs.board.current_player, pl.column,
                                    gs.board, ctx);
        perform_placement(gs, ctx, pl.column, pl.row, score, rng);
    }
}

// ---------------------------------------------------------------------------
// Full game integration tests
// ---------------------------------------------------------------------------

TEST(FullGame, CompletesToTerminal) {
    RNG rng(200);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    play_full_game(gs, ctx, rng);
    EXPECT_TRUE(is_terminal(gs.board));
}

TEST(FullGame, CellsFilledCount) {
    RNG rng(201);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    play_full_game(gs, ctx, rng);
    EXPECT_EQ(gs.board.cells_filled, kTotalCells);
}

TEST(FullGame, AllCellsFilled_NoEmpty) {
    RNG rng(202);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    play_full_game(gs, ctx, rng);
    for (int p = 0; p < kNumPlayers; ++p)
        for (int c = 0; c < kNumColumns; ++c)
            for (int r = 0; r < kNumRows; ++r)
                EXPECT_NE(gs.board.cells[p][c][r], kCellEmpty)
                    << "cell[" << p << "][" << c << "][" << r << "] still empty";
}

TEST(FullGame, DuelComputesWithoutCrash) {
    RNG rng(203);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    play_full_game(gs, ctx, rng);
    // compute_duel returns an int; just verify it doesn't crash and is finite
    int result = get_game_result(gs, ctx);
    (void)result;  // no specific value asserted
    SUCCEED();
}

TEST(FullGame, LegalCachesEmpty_AtTerminal) {
    RNG rng(204);
    GameState gs; GameContext ctx;
    init_game(gs, ctx, rng);
    play_full_game(gs, ctx, rng);
    // After all cells filled, both players' caches should be empty
    for (int p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(ctx.legal_all[p].count, 0);
        EXPECT_EQ(ctx.legal_no_turbo[p].count, 0);
    }
}

TEST(FullGame, Deterministic_SameSeedSameResult) {
    auto play = [](uint64_t seed) -> int {
        RNG rng(seed);
        GameState gs; GameContext ctx;
        init_game(gs, ctx, rng);
        while (!is_terminal(gs.board)) {
            while (can_reroll(gs, ctx))
                perform_reroll(gs, 0, rng);
            const auto& cache = get_legal_placements(gs, ctx);
            if (cache.count == 0) break;
            auto pl = cache.placements[0];
            int sc = calculate_score(pl.row, gs.dice,
                                     gs.board.current_player, pl.column,
                                     gs.board, ctx);
            perform_placement(gs, ctx, pl.column, pl.row, sc, rng);
        }
        return get_game_result(gs, ctx);
    };

    int r1 = play(42);
    int r2 = play(42);
    EXPECT_EQ(r1, r2);
}

TEST(FullGame, DifferentSeeds_TypicallyDifferentResult) {
    auto play = [](uint64_t seed) -> int {
        RNG rng(seed);
        GameState gs; GameContext ctx;
        init_game(gs, ctx, rng);
        while (!is_terminal(gs.board)) {
            while (can_reroll(gs, ctx))
                perform_reroll(gs, 0, rng);
            const auto& cache = get_legal_placements(gs, ctx);
            if (cache.count == 0) break;
            auto pl = cache.placements[0];
            int sc = calculate_score(pl.row, gs.dice,
                                     gs.board.current_player, pl.column,
                                     gs.board, ctx);
            perform_placement(gs, ctx, pl.column, pl.row, sc, rng);
        }
        return get_game_result(gs, ctx);
    };

    // Not guaranteed, but different seeds almost always yield different games
    int count_diff = 0;
    for (int s = 0; s < 10; ++s)
        if (play(s) != play(s + 100)) ++count_diff;
    EXPECT_GT(count_diff, 0);
}
