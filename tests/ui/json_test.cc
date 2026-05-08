#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include "engine/constants.h"
#include "engine/game_flow.h"
#include "solver/precomputed_tables.h"
#include "ui/json_serialization.h"
#include "ui/session_manager.h"

using json = nlohmann::json;

class JsonSerializationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        torch::set_num_threads(1);
        init_precomputed_tables(tables_);
    }

    static PrecomputedTables tables_;
};

PrecomputedTables JsonSerializationTest::tables_;

// ---------------------------------------------------------------------------
// Game state serialization
// ---------------------------------------------------------------------------

TEST_F(JsonSerializationTest, InitialStateHasRequiredFields) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    EXPECT_TRUE(j.contains("session_id"));
    EXPECT_TRUE(j.contains("game_over"));
    EXPECT_TRUE(j.contains("result"));
    EXPECT_TRUE(j.contains("current_player"));
    EXPECT_TRUE(j.contains("rolls_left"));
    EXPECT_TRUE(j.contains("waiting_for_human"));
    EXPECT_TRUE(j.contains("dice"));
    EXPECT_TRUE(j.contains("coefficients"));
    EXPECT_TRUE(j.contains("boards"));
    EXPECT_TRUE(j.contains("history"));
    EXPECT_TRUE(j.contains("player_types"));
}

TEST_F(JsonSerializationTest, DiceArrayHasFiveElements) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    EXPECT_EQ(j["dice"].size(), static_cast<size_t>(kNumDice));
    for (const auto& d : j["dice"]) {
        EXPECT_GE(d.get<int>(), 1);
        EXPECT_LE(d.get<int>(), 6);
    }
}

TEST_F(JsonSerializationTest, CoefficientsArrayHasSixElements) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    EXPECT_EQ(j["coefficients"].size(), static_cast<size_t>(kNumColumns));
    // Coefficients are a permutation of {8, 10, 12, 14, 16, 18}.
    std::vector<int> coefs;
    for (const auto& c : j["coefficients"]) coefs.push_back(c.get<int>());
    std::sort(coefs.begin(), coefs.end());
    std::vector<int> expected = {8, 10, 12, 14, 16, 18};
    EXPECT_EQ(coefs, expected);
}

TEST_F(JsonSerializationTest, BoardsHaveTwoPlayersAndAllColumns) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    EXPECT_TRUE(j["boards"].contains("player0"));
    EXPECT_TRUE(j["boards"].contains("player1"));

    const char* col_keys[] = {"down", "free", "up", "mid", "turbo", "updown"};
    for (const char* ck : col_keys) {
        EXPECT_TRUE(j["boards"]["player0"].contains(ck)) << "Missing column: " << ck;
        EXPECT_TRUE(j["boards"]["player1"].contains(ck)) << "Missing column: " << ck;
    }
}

TEST_F(JsonSerializationTest, EmptyCellsAreMinusOne) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    // At the start, most cells should be -1 (empty).
    int empty_count = 0;
    for (const auto& [colName, colData] : j["boards"]["player0"].items()) {
        for (const auto& [rowName, value] : colData.items()) {
            if (value.get<int>() == -1) empty_count++;
        }
    }
    // All 78 cells should be empty at start.
    EXPECT_EQ(empty_count, kNumColumns * kNumRows);
}

TEST_F(JsonSerializationTest, MidGameHasMixedCells) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);

    // Play some turns.
    for (int i = 0; i < 20; i++) mgr.advance_turn(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    // Should have some non-empty cells now.
    int filled = 0;
    for (const auto& [colName, colData] : j["boards"]["player0"].items()) {
        for (const auto& [rowName, value] : colData.items()) {
            if (value.get<int>() != -1) filled++;
        }
    }
    EXPECT_GT(filled, 0);
}

TEST_F(JsonSerializationTest, HistoryGrowsWithTurns) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j0 = game_state_to_json(copy);
    EXPECT_EQ(j0["history"].size(), static_cast<size_t>(0));

    mgr.advance_turn(id);
    mgr.get_session_copy(id, copy);
    json j1 = game_state_to_json(copy);
    EXPECT_EQ(j1["history"].size(), static_cast<size_t>(1));
}

TEST_F(JsonSerializationTest, HistoryEntryHasRequiredFields) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);
    mgr.advance_turn(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    const auto& entry = j["history"][0];
    EXPECT_TRUE(entry.contains("player"));
    EXPECT_TRUE(entry.contains("initial_dice"));
    EXPECT_TRUE(entry.contains("holds"));
    EXPECT_TRUE(entry.contains("placement"));
    EXPECT_TRUE(entry.contains("score"));

    EXPECT_EQ(entry["initial_dice"].size(), static_cast<size_t>(kNumDice));
    EXPECT_TRUE(entry["placement"].contains("column"));
    EXPECT_TRUE(entry["placement"].contains("row"));
}

TEST_F(JsonSerializationTest, PlayerTypesCorrect) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHuman, PlayerType::kHeuristic, 42);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    EXPECT_EQ(j["player_types"][0], "human");
    EXPECT_EQ(j["player_types"][1], "heuristic_v1");
}

TEST_F(JsonSerializationTest, GameOverState) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);
    mgr.play_to_completion(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    EXPECT_TRUE(j["game_over"].get<bool>());
    EXPECT_FALSE(j["result"].is_null());
    double result = j["result"].get<double>();
    EXPECT_TRUE(result == 0.0 || result == 0.5 || result == 1.0);
}

// ---------------------------------------------------------------------------
// Options serialization
// ---------------------------------------------------------------------------

TEST_F(JsonSerializationTest, OptionsHasRequiredFields) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHuman, PlayerType::kHeuristic, 42);

    auto opts = [&]() { bool cr = false; return mgr.get_human_options(id, cr); }();
    json j = options_to_json(opts, true);

    EXPECT_TRUE(j.contains("can_reroll"));
    EXPECT_TRUE(j.contains("placements"));
    EXPECT_TRUE(j["can_reroll"].get<bool>());
    EXPECT_FALSE(j["placements"].empty());
}

TEST_F(JsonSerializationTest, PlacementOptionsHaveReadableNames) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHuman, PlayerType::kHeuristic, 42);

    auto opts = [&]() { bool cr = false; return mgr.get_human_options(id, cr); }();
    json j = options_to_json(opts, true);

    for (const auto& p : j["placements"]) {
        EXPECT_TRUE(p.contains("column"));
        EXPECT_TRUE(p.contains("row"));
        EXPECT_TRUE(p.contains("score"));
        EXPECT_TRUE(p.contains("column_name"));
        EXPECT_TRUE(p.contains("row_name"));

        std::string cn = p["column_name"].get<std::string>();
        std::string rn = p["row_name"].get<std::string>();
        EXPECT_NE(cn, "unknown");
        EXPECT_NE(rn, "unknown");
    }
}

TEST_F(JsonSerializationTest, EmptyOptionsSerializes) {
    std::vector<std::pair<Placement, int>> empty;
    json j = options_to_json(empty, false);

    EXPECT_FALSE(j["can_reroll"].get<bool>());
    EXPECT_TRUE(j["placements"].empty());
}

// ---------------------------------------------------------------------------
// Hold held_flags and placement names in history
// ---------------------------------------------------------------------------

TEST_F(JsonSerializationTest, HistoryHoldEntryHasHeldFlags) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);
    // Play enough turns that at least one will have holds.
    for (int i = 0; i < 20; i++) mgr.advance_turn(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    for (const auto& turn : j["history"]) {
        for (const auto& hold : turn["holds"]) {
            EXPECT_TRUE(hold.contains("held_flags"));
            EXPECT_EQ(hold["held_flags"].size(), static_cast<size_t>(kNumDice));
            for (const auto& flag : hold["held_flags"])
                EXPECT_TRUE(flag.is_boolean());
        }
    }
}

TEST_F(JsonSerializationTest, HistoryPlacementHasColumnAndRowNames) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42);
    mgr.advance_turn(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    const auto& placement = j["history"][0]["placement"];
    EXPECT_TRUE(placement.contains("column_name"));
    EXPECT_TRUE(placement.contains("row_name"));
    EXPECT_NE(placement["column_name"].get<std::string>(), "unknown");
    EXPECT_NE(placement["row_name"].get<std::string>(), "unknown");
}

// ---------------------------------------------------------------------------
// Debug mode serialization
// ---------------------------------------------------------------------------

TEST_F(JsonSerializationTest, DebugModeProducesEvalData) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42,
                                 /*debug_mode=*/true);
    // Play a handful of turns to get history with debug data.
    for (int i = 0; i < 6; i++) mgr.advance_turn(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    // At least some turns should have a debug block.
    bool found_debug = false;
    for (const auto& turn : j["history"]) {
        if (!turn.contains("debug")) continue;
        found_debug = true;

        const auto& dbg = turn["debug"];
        EXPECT_TRUE(dbg.contains("hold_evals"));
        EXPECT_TRUE(dbg.contains("placement_evals"));

        // placement_evals must be sorted descending by eval_value.
        const auto& pevs = dbg["placement_evals"];
        EXPECT_GT(pevs.size(), 0u);
        for (size_t k = 1; k < pevs.size(); ++k) {
            EXPECT_GE(pevs[k - 1]["eval_value"].get<float>(),
                      pevs[k]["eval_value"].get<float>());
        }

        // hold_evals: each reroll step has candidates sorted descending.
        for (const auto& step : dbg["hold_evals"]) {
            EXPECT_GT(step.size(), 0u);
            for (size_t k = 1; k < step.size(); ++k) {
                EXPECT_GE(step[k - 1]["expected_value"].get<float>(),
                          step[k]["expected_value"].get<float>());
            }
            // Each candidate must have held_flags of length kNumDice.
            for (const auto& cand : step) {
                EXPECT_TRUE(cand.contains("held_flags"));
                EXPECT_EQ(cand["held_flags"].size(), static_cast<size_t>(kNumDice));
                EXPECT_TRUE(cand.contains("mask"));
                EXPECT_TRUE(cand.contains("expected_value"));
            }
        }
    }
    EXPECT_TRUE(found_debug) << "Expected at least one turn with debug data";
}

TEST_F(JsonSerializationTest, NonDebugModeProducesNoDebugField) {
    SessionManager mgr(tables_, nullptr, torch::kCPU);
    int id = mgr.create_session(PlayerType::kHeuristic, PlayerType::kHeuristic, 42,
                                 /*debug_mode=*/false);
    for (int i = 0; i < 6; i++) mgr.advance_turn(id);

    GameSession copy;
    mgr.get_session_copy(id, copy);
    json j = game_state_to_json(copy);

    for (const auto& turn : j["history"])
        EXPECT_FALSE(turn.contains("debug")) << "Non-debug session should have no debug field";
}
