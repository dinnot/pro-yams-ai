#include <gtest/gtest.h>

#include <torch/torch.h>

#include "solver/precomputed_tables.h"
#include "ui/session_manager.h"

class SessionManagerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        torch::set_num_threads(1);
        init_precomputed_tables(tables_);
    }

    static PrecomputedTables tables_;
};

PrecomputedTables SessionManagerTest::tables_;

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

TEST_F(SessionManagerTest, CreateAndRetrieve) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);
    EXPECT_GT(id, 0);

    GameSession copy;
    EXPECT_TRUE(mgr.get_session_copy(id, copy));
    EXPECT_EQ(copy.session_id, id);
    EXPECT_FALSE(copy.game_over);
    EXPECT_EQ(copy.player_types[0], PlayerType::kHeuristic);
    EXPECT_EQ(copy.player_types[1], PlayerType::kHeuristic);
}

TEST_F(SessionManagerTest, SessionNotFound) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    GameSession copy;
    EXPECT_FALSE(mgr.get_session_copy(999, copy));
}

TEST_F(SessionManagerTest, RemoveSession) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);
    mgr.remove_session(id);

    GameSession copy;
    EXPECT_FALSE(mgr.get_session_copy(id, copy));
}

TEST_F(SessionManagerTest, MultipleSessionsHaveUniqueIds) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id1 = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 1);
    int id2 = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 2);
    EXPECT_NE(id1, id2);
}

// ---------------------------------------------------------------------------
// Bot vs bot
// ---------------------------------------------------------------------------

TEST_F(SessionManagerTest, HeuristicVsHeuristicStepByStep) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 123);

    int turns = 0;
    while (true) {
        GameSession copy;
        mgr.get_session_copy(id, copy);
        if (copy.game_over) break;

        auto record = mgr.advance_turn(id);
        turns++;
        ASSERT_LE(turns, 160);  // Safety: max 156 turns + some margin.
    }

    // Both players should have played 78 turns each = 156 total.
    EXPECT_EQ(turns, 156);

    GameSession final_state;
    mgr.get_session_copy(id, final_state);
    EXPECT_TRUE(final_state.game_over);
    // Result should be 0.0, 0.5, or 1.0.
    EXPECT_GE(final_state.result, 0.0);
    EXPECT_LE(final_state.result, 1.0);
    EXPECT_EQ(static_cast<int>(final_state.history.size()), 156);
}

TEST_F(SessionManagerTest, HeuristicVsHeuristicPlayToCompletion) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 456);

    mgr.play_to_completion(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    EXPECT_TRUE(copy.game_over);
    EXPECT_EQ(static_cast<int>(copy.history.size()), 156);
}

TEST_F(SessionManagerTest, TurnHistoryHasValidContent) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 789);
    mgr.play_to_completion(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);

    for (const auto& turn : copy.history) {
        EXPECT_GE(turn.player, 0);
        EXPECT_LE(turn.player, 1);
        EXPECT_GE(turn.placement.column, 0);
        EXPECT_LT(turn.placement.column, 6);
        EXPECT_GE(turn.placement.row, 0);
        EXPECT_LT(turn.placement.row, 13);
        // Score can be 0 (scratch) or positive.
        EXPECT_GE(turn.score, 0);
        EXPECT_LE(turn.score, 100);
    }
}

// ---------------------------------------------------------------------------
// Human interaction
// ---------------------------------------------------------------------------

TEST_F(SessionManagerTest, HumanTurnWaiting) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHuman, PlayerType::kHeuristic, 42);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    EXPECT_TRUE(copy.waiting_for_human);
}

TEST_F(SessionManagerTest, HumanGetOptions) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHuman, PlayerType::kHeuristic, 42);

    auto opts = [&]() { bool cr = false; return mgr.get_human_options(id, cr); }();
    EXPECT_FALSE(opts.empty());

    for (const auto& [placement, score] : opts) {
        EXPECT_GE(placement.column, 0);
        EXPECT_LT(placement.column, 6);
        EXPECT_GE(placement.row, 0);
        EXPECT_LT(placement.row, 13);
    }
}

TEST_F(SessionManagerTest, HumanHoldAndReroll) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHuman, PlayerType::kHeuristic, 42);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    int initial_rolls = copy.state.rolls_left;

    // Hold first die (mask = 1).
    bool ok = mgr.human_hold(id, 1);
    EXPECT_TRUE(ok);

    mgr.get_session_copy(id, copy);
    EXPECT_EQ(copy.state.rolls_left, initial_rolls - 1);
}

TEST_F(SessionManagerTest, HumanPlace) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHuman, PlayerType::kHeuristic, 42);

    // Get legal options and place the first one.
    auto opts = [&]() { bool cr = false; return mgr.get_human_options(id, cr); }();
    ASSERT_FALSE(opts.empty());

    auto [placement, score] = opts[0];
    bool ok = mgr.human_place(id, placement.column, placement.row);
    EXPECT_TRUE(ok);

    // After human places, bot plays, then it's human's turn again (or game over).
    GameSession copy;
    mgr.get_session_copy(id, copy);
    EXPECT_GE(static_cast<int>(copy.history.size()), 1);
}

TEST_F(SessionManagerTest, HumanPlaceIllegalFails) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHuman, PlayerType::kHeuristic, 42);

    // Try placing in a cell that might not be legal (column 2 = Up, row 0 = 1s — Up
    // must start from bottom, so row 0 is illegal unless it's already built up).
    // We need to find a definitely illegal placement.
    auto opts = [&]() { bool cr = false; return mgr.get_human_options(id, cr); }();

    // Find a (col, row) pair that is NOT in the legal options.
    bool found_illegal = false;
    for (int c = 0; c < 6 && !found_illegal; c++) {
        for (int r = 0; r < 13 && !found_illegal; r++) {
            bool legal = false;
            for (const auto& [p, s] : opts) {
                if (p.column == c && p.row == r) { legal = true; break; }
            }
            if (!legal) {
                bool ok = mgr.human_place(id, c, r);
                EXPECT_FALSE(ok);
                found_illegal = true;
            }
        }
    }
    EXPECT_TRUE(found_illegal);
}

TEST_F(SessionManagerTest, HumanVsHeuristicFullGame) {
    // Play a full game as human, always choosing the first legal option.
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHuman, PlayerType::kHeuristic, 100);

    int moves = 0;
    while (true) {
        GameSession copy;
        mgr.get_session_copy(id, copy);
        if (copy.game_over) break;

        if (copy.waiting_for_human) {
            // Use all rerolls first (like a real human who keeps rerolling).
            while (can_reroll(copy.state, copy.ctx)) {
                mgr.human_hold(id, 0);
                mgr.get_session_copy(id, copy);
            }
            auto opts = [&]() { bool cr = false; return mgr.get_human_options(id, cr); }();
            ASSERT_FALSE(opts.empty())
                << "No legal options at move " << moves
                << ", current_player=" << static_cast<int>(copy.state.board.current_player)
                << ", rolls_left=" << static_cast<int>(copy.state.rolls_left);
            auto [pl, sc] = opts[0];
            bool ok = mgr.human_place(id, pl.column, pl.row);
            ASSERT_TRUE(ok);
            moves++;
        } else {
            mgr.advance_turn(id);
        }
        ASSERT_LE(moves, 80);
    }

    GameSession final_state;
    mgr.get_session_copy(id, final_state);
    EXPECT_TRUE(final_state.game_over);
    EXPECT_EQ(static_cast<int>(final_state.history.size()), 156);
}

// ---------------------------------------------------------------------------
// Debug mode
// ---------------------------------------------------------------------------

TEST_F(SessionManagerTest, DebugModePopulatesHoldAndPlacementEvals) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42,
                                 /*debug_mode=*/true);
    // Play enough turns to capture eval data.
    for (int i = 0; i < 6; i++) mgr.advance_turn(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);

    // At least one turn should have placement_evals.
    bool found_placement = false;
    for (const auto& turn : copy.history) {
        if (!turn.placement_evals.empty()) {
            found_placement = true;
            // placement_evals must be sorted descending by eval_value.
            for (size_t k = 1; k < turn.placement_evals.size(); ++k) {
                EXPECT_GE(turn.placement_evals[k - 1].eval_value,
                          turn.placement_evals[k].eval_value);
            }
        }
        // hold_evals: each reroll step must be sorted descending.
        for (const auto& step : turn.hold_evals) {
            EXPECT_EQ(step.size(), static_cast<size_t>(32));  // all 32 masks
            for (size_t k = 1; k < step.size(); ++k) {
                EXPECT_GE(step[k - 1].expected_value, step[k].expected_value);
            }
        }
    }
    EXPECT_TRUE(found_placement);
}

TEST_F(SessionManagerTest, NonDebugModeHasNoEvalData) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42,
                                 /*debug_mode=*/false);
    for (int i = 0; i < 6; i++) mgr.advance_turn(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);

    for (const auto& turn : copy.history) {
        EXPECT_TRUE(turn.hold_evals.empty());
        EXPECT_TRUE(turn.placement_evals.empty());
    }
}

// ---------------------------------------------------------------------------
// NN unavailable fallback
// ---------------------------------------------------------------------------

TEST_F(SessionManagerTest, NNUnavailableFallsBackToHeuristic) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    EXPECT_FALSE(mgr.has_nn());

    // Creating an NN session should still work (falls back to heuristic).
    int id = mgr.create_session(PlayerType::kNNSolver, PlayerType::kHeuristic, 42);
    mgr.play_to_completion(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    EXPECT_TRUE(copy.game_over);
}

// ---------------------------------------------------------------------------
// Seeding a session from an explicit board position
// ---------------------------------------------------------------------------

TEST_F(SessionManagerTest, CreateFromBoardSeedsCellsAndRollsDice) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);

    BoardState1v1 board{};
    for (int p = 0; p < 2; ++p)
        for (int c = 0; c < kNumColumns; ++c)
            for (int r = 0; r < kNumRows; ++r)
                board.cells[p][c][r] = kCellEmpty;
    int8_t coeffs[kNumColumns] = {8, 10, 12, 14, 16, 18};
    for (int c = 0; c < kNumColumns; ++c) board.coefficients[c] = coeffs[c];
    board.current_player = 1;
    board.cells[0][0][0] = 5;   // P0 Down/1s
    board.cells[0][3][6] = 0;   // P0 Mid/SS scratched
    board.cells[1][1][12] = 50; // P1 Free/Y
    board.cells_filled = 999;   // deliberately wrong; must be recomputed

    PlayerType pts[2] = {PlayerType::kHeuristic, PlayerType::kHuman};
    int id = mgr.create_session_from_board(pts, 2, board, 42);

    GameSession copy;
    ASSERT_TRUE(mgr.get_session_copy(id, copy));
    EXPECT_FALSE(copy.game_over);
    EXPECT_EQ(copy.state.board.current_player, 1);
    EXPECT_EQ(copy.state.board.cells[0][0][0], 5);
    EXPECT_EQ(copy.state.board.cells[0][3][6], 0);
    EXPECT_EQ(copy.state.board.cells[1][1][12], 50);
    EXPECT_EQ(copy.state.board.cells_filled, 3);  // recomputed from cells
    for (int c = 0; c < kNumColumns; ++c)
        EXPECT_EQ(copy.state.board.coefficients[c], coeffs[c]);
    EXPECT_EQ(copy.state.rolls_left, 2);          // fresh dice were rolled
    EXPECT_TRUE(copy.waiting_for_human);          // seat 1 is human

    // Context was rebuilt: the current player has legal placements.
    bool cr = false;
    auto opts = mgr.get_human_options(id, cr);
    EXPECT_FALSE(opts.empty());
}

TEST_F(SessionManagerTest, CreateFromTerminalBoardOpensGameOver) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);

    // Produce a real terminal board by finishing a game, then re-seed from it.
    int played = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 7);
    mgr.play_to_completion(played);
    GameSession finished;
    ASSERT_TRUE(mgr.get_session_copy(played, finished));
    ASSERT_TRUE(finished.game_over);

    PlayerType pts[2] = {PlayerType::kHeuristic, PlayerType::kHeuristic};
    int id = mgr.create_session_from_board(pts, 2, finished.state.board, 1);

    GameSession copy;
    ASSERT_TRUE(mgr.get_session_copy(id, copy));
    EXPECT_TRUE(copy.game_over);
    EXPECT_EQ(copy.result, finished.result);  // same board ⇒ same duel outcome
}

TEST_F(SessionManagerTest, CreateFromMidGameBoardPlaysToCompletion) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);

    // Play part of a game, snapshot the board, then resume from it.
    int played = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 321);
    for (int i = 0; i < 20; ++i) mgr.advance_turn(played);
    GameSession mid;
    ASSERT_TRUE(mgr.get_session_copy(played, mid));
    ASSERT_FALSE(mid.game_over);

    PlayerType pts[2] = {PlayerType::kHeuristic, PlayerType::kHeuristic};
    int id = mgr.create_session_from_board(pts, 2, mid.state.board, 5);
    mgr.play_to_completion(id);

    GameSession copy;
    ASSERT_TRUE(mgr.get_session_copy(id, copy));
    EXPECT_TRUE(copy.game_over);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

TEST_F(SessionManagerTest, SameSeedSameResult) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);

    int id1 = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 999);
    mgr.play_to_completion(id1);
    GameSession c1;
    mgr.get_session_copy(id1, c1);

    int id2 = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 999);
    mgr.play_to_completion(id2);
    GameSession c2;
    mgr.get_session_copy(id2, c2);

    EXPECT_EQ(c1.result, c2.result);
    EXPECT_EQ(c1.history.size(), c2.history.size());
    for (size_t i = 0; i < c1.history.size(); i++) {
        EXPECT_EQ(c1.history[i].score, c2.history[i].score);
        EXPECT_EQ(c1.history[i].placement.column, c2.history[i].placement.column);
        EXPECT_EQ(c1.history[i].placement.row, c2.history[i].placement.row);
    }
}
