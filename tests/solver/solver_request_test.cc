#include <gtest/gtest.h>
#include "solver/precomputed_tables.h"
#include "solver/solver.h"
#include "engine/board_init.h"
#include "engine/game_flow.h"
#include "engine/placement.h"

// ---------------------------------------------------------------------------
// Shared fixture: initialise tables once.
// ---------------------------------------------------------------------------
class SolverRequestTest : public ::testing::Test {
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
PrecomputedTables SolverRequestTest::tables;
bool SolverRequestTest::initialised = false;
int SolverRequestTest::seed_ = 1000;

// ---------------------------------------------------------------------------
// Basic request generation on fresh board
// ---------------------------------------------------------------------------
TEST_F(SolverRequestTest, FreshBoard_RequestCountPositive) {
    solver_get_requests(gs, ctx, tables, buffers);
    EXPECT_GT(buffers.request_count, 0);
    EXPECT_LT(buffers.request_count, kMaxAfterstateRequests);
}

TEST_F(SolverRequestTest, FreshBoard_ScratchIncludedForEveryCell) {
    solver_get_requests(gs, ctx, tables, buffers);
    int p = gs.board.current_player;
    const LegalPlacementCache& legal = get_legal_placements(gs, ctx);

    // For every legal cell, verify scratch (score=0) is in requests.
    for (int li = 0; li < legal.count; ++li) {
        int col = legal.placements[li].column;
        int row = legal.placements[li].row;
        bool found_scratch = false;
        for (int i = 0; i < buffers.request_count; ++i) {
            if (buffers.requests[i].placement.column == col &&
                buffers.requests[i].placement.row    == row &&
                buffers.requests[i].score == 0) {
                found_scratch = true;
                break;
            }
        }
        EXPECT_TRUE(found_scratch)
            << "No scratch request for col=" << col << " row=" << row;
    }
}

TEST_F(SolverRequestTest, FreshBoard_AllScoresNonNegative) {
    solver_get_requests(gs, ctx, tables, buffers);
    for (int i = 0; i < buffers.request_count; ++i)
        EXPECT_GE(buffers.requests[i].score, 0);
}

TEST_F(SolverRequestTest, FreshBoard_AllColsAndRowsValid) {
    solver_get_requests(gs, ctx, tables, buffers);
    for (int i = 0; i < buffers.request_count; ++i) {
        EXPECT_GE(buffers.requests[i].placement.column, 0);
        EXPECT_LT(buffers.requests[i].placement.column, kNumColumns);
        EXPECT_GE(buffers.requests[i].placement.row, 0);
        EXPECT_LT(buffers.requests[i].placement.row, kNumRows);
    }
}

// ---------------------------------------------------------------------------
// Golden Rule filtering
// ---------------------------------------------------------------------------
TEST_F(SolverRequestTest, GoldenRule_ScoresBelowThresholdExcluded) {
    // Set golden_max for row 3s (kRow3s = row 2) in column 0 to 12.
    // Scores 3, 6, 9 should be excluded; 12, 15 should be included.
    int p = gs.board.current_player;
    ctx.golden_max[kColFree][kRow3s] = 12;

    solver_get_requests(gs, ctx, tables, buffers);

    // Verify scores 3, 6, 9 are not in requests for (kColFree, kRow3s).
    for (int i = 0; i < buffers.request_count; ++i) {
        if (buffers.requests[i].placement.column == kColFree &&
            buffers.requests[i].placement.row    == kRow3s) {
            int s = buffers.requests[i].score;
            EXPECT_NE(s, 3)  << "Score 3 should be excluded by Golden Rule";
            EXPECT_NE(s, 6)  << "Score 6 should be excluded by Golden Rule";
            EXPECT_NE(s, 9)  << "Score 9 should be excluded by Golden Rule";
        }
    }

    // Verify 12 and 15 are included.
    bool found12 = false, found15 = false;
    for (int i = 0; i < buffers.request_count; ++i) {
        if (buffers.requests[i].placement.column == kColFree &&
            buffers.requests[i].placement.row    == kRow3s) {
            if (buffers.requests[i].score == 12) found12 = true;
            if (buffers.requests[i].score == 15) found15 = true;
        }
    }
    EXPECT_TRUE(found12) << "Score 12 (>= threshold 12) should be included";
    EXPECT_TRUE(found15) << "Score 15 (>= threshold 12) should be included";
}

// ---------------------------------------------------------------------------
// SS/LS constraint filtering
// ---------------------------------------------------------------------------
TEST_F(SolverRequestTest, SS_ForcedScratch_WhenLSScratched) {
    // If LS is already scratched in a column, SS must scratch.
    int p = gs.board.current_player;
    int col = kColFree;

    // Simulate LS scratched by setting ls_scratched flag.
    ctx.ls_scratched[p][col] = true;
    // Place a scratch in LS to make it a valid state.
    apply_placement(p, col, kRowLS, 0, gs.board, ctx);

    solver_get_requests(gs, ctx, tables, buffers);

    // SS requests for this column should only contain scratch.
    for (int i = 0; i < buffers.request_count; ++i) {
        if (buffers.requests[i].placement.column == col &&
            buffers.requests[i].placement.row == kRowSS) {
            EXPECT_EQ(buffers.requests[i].score, 0)
                << "SS should be forced scratch when LS is scratched";
        }
    }
}

TEST_F(SolverRequestTest, LS_ForcedScratch_WhenSSScratched) {
    int p = gs.board.current_player;
    int col = kColFree;
    ctx.ss_scratched[p][col] = true;
    apply_placement(p, col, kRowSS, 0, gs.board, ctx);

    solver_get_requests(gs, ctx, tables, buffers);

    for (int i = 0; i < buffers.request_count; ++i) {
        if (buffers.requests[i].placement.column == col &&
            buffers.requests[i].placement.row == kRowLS) {
            EXPECT_EQ(buffers.requests[i].score, 0)
                << "LS should be forced scratch when SS is scratched";
        }
    }
}

// ---------------------------------------------------------------------------
// Turbo gating — Turbo limited to 2 rolls max (available when rolls_left > 0)
// ---------------------------------------------------------------------------
TEST_F(SolverRequestTest, TurboIncluded_WhenRollsLeft1) {
    // When rolls_left > 0, Turbo IS available (within 2-roll limit).
    gs.rolls_left = 1;
    solver_get_requests(gs, ctx, tables, buffers);
    bool found_turbo = false;
    for (int i = 0; i < buffers.request_count; ++i)
        if (buffers.requests[i].placement.column == kColTurbo) { found_turbo = true; break; }
    EXPECT_TRUE(found_turbo) << "Turbo should be included when rolls_left>0";
}

TEST_F(SolverRequestTest, TurboExcluded_WhenRollsLeft0) {
    // When rolls_left == 0, all 3 rolls used → Turbo NOT available (exceeds 2-roll limit).
    gs.rolls_left = 0;
    solver_get_requests(gs, ctx, tables, buffers);
    for (int i = 0; i < buffers.request_count; ++i)
        EXPECT_NE(buffers.requests[i].placement.column, kColTurbo)
            << "Turbo should be excluded when rolls_left=0 (3 rolls used, exceeds limit)";
}
