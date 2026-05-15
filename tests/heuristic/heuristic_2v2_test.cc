#include <gtest/gtest.h>
#include <cmath>

#include "engine/board_init.h"
#include "engine/game_traits.h"
#include "engine/placement.h"
#include "engine/scoring.h"
#include "heuristic/heuristic_bot.h"
#include "solver/precomputed_tables.h"

// ---------------------------------------------------------------------------
// 2v2 heuristic smoke + structural tests (Task 4 validation).
//
// These verify the templated evaluators compile and produce sensible 2v2 EVs:
//   - V1 (no opp logic) — straightforward port.
//   - V2 — team margin returned from active player's perspective. Sign flips
//          based on which team the active player is on.
//   - Research evaluator (V17 config) runs without crashing on a 2v2 board.
//
// Numerical EV values are NOT pinned (the DP tables make exact values
// brittle); we instead check structural invariants: ordering, signs, and
// 1v1-perspective consistency.
// ---------------------------------------------------------------------------

class Heuristic2v2Test : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) {
            init_precomputed_tables(tables);
            initialised = true;
        }
    }
    void SetUp() override {
        RNG rng(seed_++);
        init_board<Yams2v2>(board, rng);
        init_context<Yams2v2>(ctx, board);
    }
    BoardState2v2 board;
    GameContext2v2 ctx;
    static int seed_;
};
PrecomputedTables Heuristic2v2Test::tables;
bool Heuristic2v2Test::initialised = false;
int Heuristic2v2Test::seed_ = 5000;

// ---------------------------------------------------------------------------
// V1 — score × coefficient. No opp logic, so 2v2 result mirrors 1v1.
// ---------------------------------------------------------------------------
TEST_F(Heuristic2v2Test, V1_ScoreCoefficientProduct) {
    int col18 = -1;
    for (int c = 0; c < kNumColumns; ++c)
        if (board.coefficients[c] == 18) { col18 = c; break; }
    ASSERT_GE(col18, 0);

    AfterstateRequest req{{(int8_t)col18, 0}, 30};
    double ev = 0.0;
    heuristic_evaluate<Yams2v2>(board, ctx, &req, 1, &ev);
    EXPECT_NEAR(ev, 540.0, 1e-9);
}

// ---------------------------------------------------------------------------
// V2 — runs without crashing on a 2v2 board.
// ---------------------------------------------------------------------------
TEST_F(Heuristic2v2Test, V2_RunsOnEmptyBoard) {
    // Build a handful of request candidates spanning multiple columns.
    AfterstateRequest reqs[4] = {
        {{0, 0}, 5},
        {{1, 5}, 30},
        {{3, 8}, 50},
        {{5, 12}, 100},
    };
    double evs[4] = {};
    heuristic_evaluate_v2<Yams2v2>(board, ctx, reqs, 4, evs, tables);

    // Sanity: EVs must be finite (no NaN/inf).
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(evs[i])) << "evs[" << i << "] = " << evs[i];
    }
}

// ---------------------------------------------------------------------------
// V2 perspective: when the active player is P0 (Team 0) vs P1 (Team 1), the
// EVs for the SAME column position should have opposite signs (the position
// is symmetric, only the perspective changes).
// ---------------------------------------------------------------------------
TEST_F(Heuristic2v2Test, V2_PerspectiveFlipsBetweenTeams) {
    // Pre-fill the board so the position is non-trivial: teammate P2 already
    // scored well in the Free column's 6s row.
    apply_placement<Yams2v2>(/*p=*/2, kColFree, kRow6s, /*score=*/24, board, ctx);

    AfterstateRequest req{{(int8_t)kColFree, (int8_t)kRow1s}, 5};

    // Evaluate as P0 (Team 0, same team as P2 who just scored well).
    board.current_player = 0;
    double ev_p0 = 0.0;
    heuristic_evaluate_v2<Yams2v2>(board, ctx, &req, 1, &ev_p0, tables);

    // Evaluate as P1 (Team 1, opposing team).
    board.current_player = 1;
    double ev_p1 = 0.0;
    heuristic_evaluate_v2<Yams2v2>(board, ctx, &req, 1, &ev_p1, tables);

    EXPECT_TRUE(std::isfinite(ev_p0));
    EXPECT_TRUE(std::isfinite(ev_p1));
    // P0 is on Team 0 (same team as P2 who already scored 24). The position
    // is favorable for Team 0, so P0's EV should be positive. P1 is on the
    // opposing Team 1, so the same position is unfavorable for them.
    // (The two evaluations are NOT exact opposites because the candidate
    //  placement is applied by p_me — different placements yield different
    //  afterstates. Sign relationship is the load-bearing claim.)
    EXPECT_GT(ev_p0, 0.0) << "P0 EV should be positive — P0's team is ahead";
    EXPECT_LT(ev_p1, 0.0) << "P1 EV should be negative — P1's team is behind";
}

// ---------------------------------------------------------------------------
// V3 — runs without crashing on a 2v2 board (exercises Rule 1's all-opps
// loop and Rules 3/4 which are pure self-evaluation).
// ---------------------------------------------------------------------------
TEST_F(Heuristic2v2Test, V3_RunsOnEmptyBoard) {
    AfterstateRequest req{{0, 0}, 5};
    double ev = 0.0;
    heuristic_evaluate_v3<Yams2v2>(board, ctx, &req, 1, &ev, tables);
    EXPECT_TRUE(std::isfinite(ev));
}

// ---------------------------------------------------------------------------
// Research evaluator (V17 config — most complex: V13 base + upper_bonus
// penalty + win_odds). Runs without crashing on a 2v2 board.
// ---------------------------------------------------------------------------
TEST_F(Heuristic2v2Test, ResearchV17_RunsOnEmptyBoard) {
    const ResearchConfig& cfg = get_research_config_for(HeuristicVersion::V17);
    AfterstateRequest req{{0, 0}, 5};
    double ev = 0.0;
    heuristic_evaluate_research<Yams2v2>(board, ctx, &req, 1, &ev, tables, cfg);
    EXPECT_TRUE(std::isfinite(ev));
    // V17 has output_win_odds=true → EV should be a probability in [0, 1].
    EXPECT_GE(ev, 0.0);
    EXPECT_LE(ev, 1.0);
}
