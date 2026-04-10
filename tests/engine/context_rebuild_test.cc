#include <gtest/gtest.h>

#include <cstring>

#include "engine/board_init.h"
#include "engine/context_rebuild.h"
#include "engine/game_flow.h"
#include "engine/rng.h"
#include "engine/scoring.h"
#include "heuristic/heuristic_bot.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

class ContextRebuildTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_precomputed_tables(tables_);
    }

    static PrecomputedTables tables_;
};

PrecomputedTables ContextRebuildTest::tables_;

// ---------------------------------------------------------------------------
// Round-trip: play a game with incremental updates, then rebuild context from
// the board and verify all fields match.
// ---------------------------------------------------------------------------

TEST_F(ContextRebuildTest, MatchesIncrementalAfter10Turns) {
    RNG rng(42);
    GameState state;
    GameContext ctx;
    init_game(state, ctx, rng);
    SolverBuffers buffers{};

    for (int t = 0; t < 10; ++t) {
        heuristic_play_turn(state, ctx, tables_, buffers, rng);
    }

    // Rebuild context from scratch.
    GameContext rebuilt;
    rebuild_context_from_board(state.board, rebuilt);

    // Compare golden_max.
    for (int c = 0; c < kNumColumns; ++c)
        for (int r = 0; r < kNumRows; ++r)
            EXPECT_EQ(rebuilt.golden_max[c][r], ctx.golden_max[c][r])
                << "golden_max mismatch at c=" << c << " r=" << r;

    // Compare upper_sum.
    for (int p = 0; p < kNumPlayers; ++p)
        for (int c = 0; c < kNumColumns; ++c)
            EXPECT_EQ(rebuilt.upper_sum[p][c], ctx.upper_sum[p][c])
                << "upper_sum mismatch at p=" << p << " c=" << c;

    // Compare SS/LS scratch.
    for (int p = 0; p < kNumPlayers; ++p) {
        for (int c = 0; c < kNumColumns; ++c) {
            EXPECT_EQ(rebuilt.ss_scratched[p][c], ctx.ss_scratched[p][c])
                << "ss_scratched mismatch at p=" << p << " c=" << c;
            EXPECT_EQ(rebuilt.ls_scratched[p][c], ctx.ls_scratched[p][c])
                << "ls_scratched mismatch at p=" << p << " c=" << c;
        }
    }

    // Compare lower_has_scratch.
    for (int p = 0; p < kNumPlayers; ++p)
        for (int c = 0; c < kNumColumns; ++c)
            EXPECT_EQ(rebuilt.lower_has_scratch[p][c], ctx.lower_has_scratch[p][c])
                << "lower_has_scratch mismatch at p=" << p << " c=" << c;

    for (int p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(rebuilt.non_turbo_cells_remaining[p], ctx.non_turbo_cells_remaining[p])
            << "non_turbo_cells_remaining mismatch at p=" << p;
    }

    // Compare legal placements.
    for (int p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(rebuilt.legal_all[p].count, ctx.legal_all[p].count)
            << "legal_all count mismatch at p=" << p;
        EXPECT_EQ(rebuilt.legal_no_turbo[p].count, ctx.legal_no_turbo[p].count)
            << "legal_no_turbo count mismatch at p=" << p;

        for (int c = 0; c < kNumColumns; ++c)
            for (int r = 0; r < kNumRows; ++r) {
                EXPECT_EQ(rebuilt.legal_all[p].is_legal[c][r],
                          ctx.legal_all[p].is_legal[c][r])
                    << "legal_all mismatch at p=" << p << " c=" << c << " r=" << r;
                EXPECT_EQ(rebuilt.legal_no_turbo[p].is_legal[c][r],
                          ctx.legal_no_turbo[p].is_legal[c][r])
                    << "legal_no_turbo mismatch at p=" << p << " c=" << c << " r=" << r;
            }
    }
}

TEST_F(ContextRebuildTest, MatchesIncrementalMidGame) {
    RNG rng(123);
    GameState state;
    GameContext ctx;
    init_game(state, ctx, rng);
    SolverBuffers buffers{};

    // Play to roughly half the game.
    for (int t = 0; t < 78; ++t) {
        heuristic_play_turn(state, ctx, tables_, buffers, rng);
    }

    GameContext rebuilt;
    rebuild_context_from_board(state.board, rebuilt);

    // Spot-check key fields.
    for (int c = 0; c < kNumColumns; ++c)
        for (int r = 0; r < kNumRows; ++r)
            EXPECT_EQ(rebuilt.golden_max[c][r], ctx.golden_max[c][r]);

    for (int p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(rebuilt.non_turbo_cells_remaining[p], ctx.non_turbo_cells_remaining[p]);
        EXPECT_EQ(rebuilt.legal_all[p].count, ctx.legal_all[p].count);
    }
}

TEST_F(ContextRebuildTest, MatchesIncrementalEndGame) {
    RNG rng(456);
    GameState state;
    GameContext ctx;
    init_game(state, ctx, rng);
    SolverBuffers buffers{};

    // Play to completion.
    while (!is_terminal(state.board))
        heuristic_play_turn(state, ctx, tables_, buffers, rng);

    GameContext rebuilt;
    rebuild_context_from_board(state.board, rebuilt);

    for (int p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(rebuilt.non_turbo_cells_remaining[p], 0);
        EXPECT_EQ(rebuilt.legal_all[p].count, 0);
    }

    for (int c = 0; c < kNumColumns; ++c)
        for (int r = 0; r < kNumRows; ++r)
            EXPECT_EQ(rebuilt.golden_max[c][r], ctx.golden_max[c][r]);
}

TEST_F(ContextRebuildTest, EmptyBoardMatchesInitContext) {
    RNG rng(789);
    BoardState board;
    init_board(board, rng);

    GameContext via_init;
    init_context(via_init, board);

    GameContext via_rebuild;
    rebuild_context_from_board(board, via_rebuild);

    EXPECT_EQ(via_rebuild.non_turbo_cells_remaining[0],
              via_init.non_turbo_cells_remaining[0]);
    EXPECT_EQ(via_rebuild.non_turbo_cells_remaining[1],
              via_init.non_turbo_cells_remaining[1]);
    EXPECT_EQ(via_rebuild.legal_all[0].count, via_init.legal_all[0].count);
    EXPECT_EQ(via_rebuild.legal_all[1].count, via_init.legal_all[1].count);
}
