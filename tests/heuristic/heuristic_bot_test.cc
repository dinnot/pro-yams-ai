#include <gtest/gtest.h>
#include "heuristic/heuristic_bot.h"
#include "engine/board_init.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"

// ---------------------------------------------------------------------------
// Shared fixture: initialise tables once.
// ---------------------------------------------------------------------------
class HeuristicBotTest : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) { init_precomputed_tables(tables); initialised = true; }
    }
    void SetUp() override {
        RNG rng(seed_++);
        init_game(gs, ctx, rng);
    }
    GameState gs;
    GameContext ctx;
    SolverBuffers buffers;
    static int seed_;
};
PrecomputedTables HeuristicBotTest::tables;
bool HeuristicBotTest::initialised = false;
int HeuristicBotTest::seed_ = 3000;

// ---------------------------------------------------------------------------
// Heuristic evaluation
// ---------------------------------------------------------------------------
TEST_F(HeuristicBotTest, Evaluation_Score30_Coeff18_Gives540) {
    // Build one request: score=30, column with coefficient 18.
    // Find the column with coefficient 18.
    int col18 = -1;
    for (int c = 0; c < kNumColumns; ++c)
        if (gs.board.coefficients[c] == 18) { col18 = c; break; }
    ASSERT_GE(col18, 0) << "No column with coefficient 18 found";

    AfterstateRequest req{{(int8_t)col18, 0}, 30};
    double ev = 0.0;
    heuristic_evaluate(gs.board, ctx, &req, 1, &ev);
    EXPECT_NEAR(ev, 540.0, 1e-9);
}

TEST_F(HeuristicBotTest, Evaluation_Scratch_GivesZero) {
    AfterstateRequest req{{0, 0}, 0};
    double ev = 999.0;
    heuristic_evaluate(gs.board, ctx, &req, 1, &ev);
    EXPECT_NEAR(ev, 0.0, 1e-9);
}

TEST_F(HeuristicBotTest, Evaluation_BatchCorrect) {
    // Two requests: score=10/col0, score=20/col1.
    AfterstateRequest reqs[2] = {
        {{0, 0}, 10},
        {{1, 0}, 20}
    };
    double evs[2] = {};
    heuristic_evaluate(gs.board, ctx, reqs, 2, evs);
    EXPECT_NEAR(evs[0], 10.0 * gs.board.coefficients[0], 1e-9);
    EXPECT_NEAR(evs[1], 20.0 * gs.board.coefficients[1], 1e-9);
}

// ---------------------------------------------------------------------------
// Play complete games
// ---------------------------------------------------------------------------
TEST_F(HeuristicBotTest, PlayTurn_CellFilled) {
    int cells_before = gs.board.cells_filled;
    RNG rng(42);
    heuristic_play_turn(gs, ctx, tables, buffers, rng);
    EXPECT_EQ(gs.board.cells_filled, cells_before + 1);
}

TEST_F(HeuristicBotTest, CompleteGame_100Games_NoCrash) {
    int p0_wins = 0, p1_wins = 0, draws = 0;
    for (int seed = 0; seed < 100; ++seed) {
        RNG rng(seed + 10000);
        int result = play_heuristic_game(rng, tables);
        if (result > 0) ++p0_wins;
        else if (result < 0) ++p1_wins;
        else ++draws;
    }
    // Just verify all games completed without crashing.
    EXPECT_EQ(p0_wins + p1_wins + draws, 100);
    // Heuristic is symmetric → roughly balanced (not strictly required).
    EXPECT_GT(p0_wins + p1_wins, 0);  // At least one decisive game.
}

TEST_F(HeuristicBotTest, CompleteGame_AllCellsFilled) {
    RNG rng(1234);
    GameState state;
    GameContext ctx2;
    SolverBuffers buf;
    init_game(state, ctx2, rng);
    while (!is_terminal(state.board))
        heuristic_play_turn(state, ctx2, tables, buf, rng);
    EXPECT_EQ(state.board.cells_filled, kTotalCells);
}

// ---------------------------------------------------------------------------
// Heuristic beats random (>90% win rate over 100 games)
// ---------------------------------------------------------------------------
TEST_F(HeuristicBotTest, HeuristicBeatsRandom_90Percent) {
    // Random bot: always rerolls once, then places in the first legal cell.
    int heuristic_wins = 0;

    for (int seed = 0; seed < 100; ++seed) {
        RNG rng(seed + 20000);
        GameState state;
        GameContext ctx2;
        init_game(state, ctx2, rng);

        SolverBuffers hbuf;

        while (!is_terminal(state.board)) {
            int p = state.board.current_player;
            if (p == 0) {
                // Heuristic player.
                heuristic_play_turn(state, ctx2, tables, hbuf, rng);
            } else {
                // Random player: use all rerolls, then place in first legal cell.
                while (can_reroll(state, ctx2))
                    perform_reroll(state, 0, rng);
                const auto& legal = get_legal_placements(state, ctx2);
                if (legal.count > 0) {
                    auto pl = legal.placements[0];
                    int sc = calculate_score(pl.row, state.dice,
                                             state.board.current_player, pl.column,
                                             state.board, ctx2);
                    perform_placement(state, ctx2, pl.column, pl.row, rng);
                }
            }
        }

        int result = get_game_result(state, ctx2);
        if (result > 0) ++heuristic_wins;
    }

    EXPECT_GT(heuristic_wins, 90)
        << "Heuristic should win >90% vs random (got " << heuristic_wins << "/100)";
}
