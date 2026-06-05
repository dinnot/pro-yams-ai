#include <gtest/gtest.h>
#include "solver/precomputed_tables.h"
#include "solver/solver.h"
#include "engine/board_init.h"
#include "engine/game_flow.h"
#include "engine/game_rules.h"

// Shared fixture
class SolverResolveTest : public ::testing::Test {
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

    // Helper: fill buffers with constant EVs and resolve.
    SolverResult resolve_with_evs_greedy(double ev_value) {
        solver_get_requests(gs, ctx, tables, buffers);
        for (int i = 0; i < buffers.request_count; ++i)
            buffers.evs[i] = ev_value;
        return solver_resolve_greedy(gs, ctx, tables, buffers);
    }
};
PrecomputedTables SolverResolveTest::tables;
bool SolverResolveTest::initialised = false;
int SolverResolveTest::seed_ = 2000;

// ---------------------------------------------------------------------------
// Trivial case: rolls_left = 0 → must place immediately
// ---------------------------------------------------------------------------
TEST_F(SolverResolveTest, RollsLeft0_MustPlace) {
    gs.rolls_left = 0;
    solver_get_requests(gs, ctx, tables, buffers);
    for (int i = 0; i < buffers.request_count; ++i) buffers.evs[i] = (double)i;
    SolverResult result = solver_resolve_greedy(gs, ctx, tables, buffers);
    EXPECT_TRUE(result.should_place);
}

TEST_F(SolverResolveTest, RollsLeft0_ReturnsHighestEVForCurrentDice) {
    gs.rolls_left = 0;
    solver_get_requests(gs, ctx, tables, buffers);
    // Assign EV = index so the last achievable request has highest EV.
    for (int i = 0; i < buffers.request_count; ++i) buffers.evs[i] = (double)i;
    SolverResult result = solver_resolve_greedy(gs, ctx, tables, buffers);
    EXPECT_TRUE(result.should_place);
    EXPECT_GE(result.expected_value, 0.0);
}

// ---------------------------------------------------------------------------
// Hold vs stop decision
// ---------------------------------------------------------------------------
TEST_F(SolverResolveTest, AllEVsZero_SolverReturnsValidResult) {
    // With all EVs = 0, solver should still return a valid action.
    solver_get_requests(gs, ctx, tables, buffers);
    for (int i = 0; i < buffers.request_count; ++i) buffers.evs[i] = 0.0;
    SolverResult result = solver_resolve_greedy(gs, ctx, tables, buffers);
    // Result must be either place or hold (not an invalid state).
    if (result.should_place) {
        EXPECT_GE(result.placement.column, 0);
        EXPECT_LT(result.placement.column, kNumColumns);
        EXPECT_GE(result.placement.row, 0);
        EXPECT_LT(result.placement.row, kNumRows);
    } else {
        EXPECT_GE(result.hold_mask, 0);
        EXPECT_LT(result.hold_mask, kNumHoldMasks);
    }
}

// ---------------------------------------------------------------------------
// Layer consistency: V0 <= V1 <= V2 (more rolls = higher or equal EV)
// ---------------------------------------------------------------------------
TEST_F(SolverResolveTest, LayerConsistency_MoreRollsHigherEV) {
    // Generate requests and assign plausible EVs.
    solver_get_requests(gs, ctx, tables, buffers);
    int cnt = buffers.request_count;
    for (int i = 0; i < cnt; ++i) buffers.evs[i] = (double)(i % 100);

    // Resolve at different rolls_left levels and collect the EV.
    gs.rolls_left = 0;
    SolverResult r0 = solver_resolve_greedy(gs, ctx, tables, buffers);

    gs.rolls_left = 1;
    SolverResult r1 = solver_resolve_greedy(gs, ctx, tables, buffers);

    gs.rolls_left = 2;
    SolverResult r2 = solver_resolve_greedy(gs, ctx, tables, buffers);

    EXPECT_LE(r0.expected_value, r1.expected_value + 1e-9)
        << "V0 should be <= V1 (more rolls = more opportunity)";
    EXPECT_LE(r1.expected_value, r2.expected_value + 1e-9)
        << "V1 should be <= V2";
}

// ---------------------------------------------------------------------------
// rolls_left=0 always places
// ---------------------------------------------------------------------------
TEST_F(SolverResolveTest, RollsLeft0_AlwaysPlaces) {
    // When rolls_left==0, the player must place — no choice.
    gs.rolls_left = 0;
    solver_get_requests(gs, ctx, tables, buffers);
    ASSERT_GT(buffers.request_count, 0);
    for (int i = 0; i < buffers.request_count; ++i) buffers.evs[i] = (double)(i % 100);
    SolverResult result = solver_resolve_greedy(gs, ctx, tables, buffers);
    EXPECT_TRUE(result.should_place);
    EXPECT_GE(result.placement.column, 0);
    EXPECT_LT(result.placement.column, kNumColumns);
}

// ---------------------------------------------------------------------------
// Exploration: greedy → always same result; high temp → sometimes different
// ---------------------------------------------------------------------------
TEST_F(SolverResolveTest, Greedy_Deterministic) {
    solver_get_requests(gs, ctx, tables, buffers);
    for (int i = 0; i < buffers.request_count; ++i) buffers.evs[i] = (double)i;

    SolverResult r1 = solver_resolve_greedy(gs, ctx, tables, buffers);
    SolverResult r2 = solver_resolve_greedy(gs, ctx, tables, buffers);

    EXPECT_EQ(r1.should_place, r2.should_place);
    if (r1.should_place) {
        EXPECT_EQ(r1.placement.column, r2.placement.column);
        EXPECT_EQ(r1.placement.row, r2.placement.row);
        EXPECT_EQ(r1.score, r2.score);
    } else {
        EXPECT_EQ(r1.hold_mask, r2.hold_mask);
    }
}

TEST_F(SolverResolveTest, HighTemperature_SometimesNonOptimal) {
    // With very high placement temperature and rolls_left=0, the solver must place
    // but should sometimes pick non-greedy placements.
    gs.rolls_left = 0;
    solver_get_requests(gs, ctx, tables, buffers);
    ASSERT_GE(buffers.request_count, 2);

    // Assign distinct EVs so there's a clear "best" and alternatives.
    for (int i = 0; i < buffers.request_count; ++i)
        buffers.evs[i] = static_cast<double>(i + 1) / buffers.request_count;

    SolverConfig cfg{5.0, 0.0, true};  // high placement temperature

    // With high temperature, different seeds should sometimes yield different placements.
    int first_col = -1, first_row = -1;
    bool any_diff = false;
    for (int seed = 0; seed < 200 && !any_diff; ++seed) {
        RNG rng(seed);
        SolverResult r = solver_resolve(gs, ctx, tables, buffers, cfg, rng);
        ASSERT_TRUE(r.should_place);
        if (first_col < 0) {
            first_col = r.placement.column;
            first_row = r.placement.row;
        } else if (r.placement.column != first_col || r.placement.row != first_row) {
            any_diff = true;
        }
    }
    EXPECT_TRUE(any_diff) << "High temperature should sometimes pick non-greedy placement";
}

// ---------------------------------------------------------------------------
// Result placement/score is in request list
// ---------------------------------------------------------------------------
TEST_F(SolverResolveTest, ResultPlacement_InRequestList) {
    gs.rolls_left = 0;
    solver_get_requests(gs, ctx, tables, buffers);
    for (int i = 0; i < buffers.request_count; ++i) buffers.evs[i] = (double)(i % 50);
    SolverResult result = solver_resolve_greedy(gs, ctx, tables, buffers);
    ASSERT_TRUE(result.should_place);

    bool found = false;
    for (int i = 0; i < buffers.request_count; ++i) {
        if (buffers.requests[i].placement.column == result.placement.column &&
            buffers.requests[i].placement.row    == result.placement.row &&
            buffers.requests[i].score == result.score) {
            found = true; break;
        }
    }
    EXPECT_TRUE(found) << "Returned placement not found in request list";
}

TEST_F(SolverResolveTest, RerollEvaluatesV0NoTurbo) {
    // When rolls_left == 1, rerolling drops rolls_left to 0, where Turbo is illegal.
    gs.rolls_left = 1;
    solver_get_requests(gs, ctx, tables, buffers);

    // Give non-Turbo placements a terrible EV, but Turbo an excellent EV.
    for (int i = 0; i < buffers.request_count; ++i) {
        if (buffers.requests[i].placement.column == kColTurbo) {
            buffers.evs[i] = 1000.0;
        } else {
            buffers.evs[i] = 0.0;
        }
    }

    SolverResult result = solver_resolve_greedy(gs, ctx, tables, buffers);

    // The solver must realize that if it stops, it secures 1000.0 from Turbo.
    // If it rerolls, it loses access to Turbo, reducing its max possible EV to 0.0.
    // Thus, it must unequivocally choose to stop and place in Turbo.
    EXPECT_TRUE(result.should_place);
    EXPECT_EQ(result.placement.column, kColTurbo);
}

TEST_F(SolverResolveTest, RollsLeft0_ProhibitsTurbo) {
    // When rolls_left == 0, Turbo columns are strictly illegal even if they have high EV.
    // Crucially, the solver-resolve must handle this even if the request buffer 
    // was generated when Turbo was still legal (e.g. rolls_left == 2).
    gs.rolls_left = 2;
    solver_get_requests(gs, ctx, tables, buffers);

    // Ensure we HAVE a Turbo request in the list.
    bool has_turbo = false;
    for (int i = 0; i < buffers.request_count; ++i) {
        if (buffers.requests[i].placement.column == kColTurbo) {
            has_turbo = true;
            buffers.evs[i] = 1000.0; // Excellent EV for Turbo
        } else {
            buffers.evs[i] = 0.0;    // Terrible EV for others
        }
    }
    ASSERT_TRUE(has_turbo);

    // Now pretend we are at rolls_left == 0
    gs.rolls_left = 0;
    SolverResult result = solver_resolve_greedy(gs, ctx, tables, buffers);

    // Should place, but NOT in Turbo.
    EXPECT_TRUE(result.should_place);
    EXPECT_NE(result.placement.column, kColTurbo);
    EXPECT_EQ(result.expected_value, 0.0); // Should match the best non-Turbo EV
}

// ---------------------------------------------------------------------------
// "Lucky Yams" first-roll bonus: solver places the wildcard max in the cell
// with the best EV, ignoring the dice faces.
// ---------------------------------------------------------------------------
TEST_F(SolverResolveTest, LuckyYams_PicksMaxScoringCell) {
    set_game_rules(GameRules{/*yams_first_roll_bonus=*/true});

    // First roll, five 3s → bonus active.
    gs.rolls_left = 2;
    for (int i = 0; i < kNumDice; ++i) gs.dice[i] = 3;
    ASSERT_TRUE(yams_bonus_active(gs));

    solver_get_requests(gs, ctx, tables, buffers);

    // Find the (Free, Yams=100) afterstate and give it the single best EV.
    int target = -1;
    for (int i = 0; i < buffers.request_count; ++i) {
        buffers.evs[i] = 0.0;
        if (buffers.requests[i].placement.column == kColFree &&
            buffers.requests[i].placement.row == kRowY &&
            buffers.requests[i].score == 100) {
            target = i;
        }
    }
    ASSERT_GE(target, 0);
    buffers.evs[target] = 50.0;

    SolverResult result = solver_resolve_greedy(gs, ctx, tables, buffers);
    EXPECT_TRUE(result.should_place);
    EXPECT_EQ(result.placement.column, kColFree);
    EXPECT_EQ(result.placement.row, kRowY);
    EXPECT_EQ(result.score, 100);           // wildcard max, not the dice-derived 80
    EXPECT_EQ(result.chosen_request_idx, target);

    set_game_rules(GameRules{});  // restore default
}
