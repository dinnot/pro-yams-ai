#include "ui/json_serialization.h"

#include "engine/constants.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// game_state_to_json
// ---------------------------------------------------------------------------

json game_state_to_json(const GameSession& session) {
    const GameState&   gs  = session.state;
    const BoardState&  bs  = gs.board;

    json j;
    j["session_id"]        = session.session_id;
    j["game_over"]         = session.game_over;
    j["result"]            = session.game_over
                             ? json(session.result) : json(nullptr);
    j["current_player"]    = static_cast<int>(bs.current_player);
    j["rolls_left"]        = static_cast<int>(gs.rolls_left);
    j["waiting_for_human"] = session.waiting_for_human;

    // Dice
    json dice = json::array();
    for (int i = 0; i < kNumDice; ++i)
        dice.push_back(static_cast<int>(gs.dice[i]));
    j["dice"] = dice;

    // Column coefficients
    json coefs = json::array();
    for (int c = 0; c < kNumColumns; ++c)
        coefs.push_back(static_cast<int>(bs.coefficients[c]));
    j["coefficients"] = coefs;

    // Player boards
    json boards = json::object();
    for (int p = 0; p < kNumPlayers; ++p) {
        json pboard = json::object();
        for (int c = 0; c < kNumColumns; ++c) {
            json col = json::object();
            for (int r = 0; r < kNumRows; ++r)
                col[row_name(r)] = static_cast<int>(bs.cells[p][c][r]);
            pboard[column_name(c)] = col;
        }
        boards["player" + std::to_string(p)] = pboard;
    }
    j["boards"] = boards;

    // Turn history
    json hist = json::array();
    for (const auto& turn : session.history) {
        json t;
        t["player"] = turn.player;

        json init = json::array();
        for (int i = 0; i < kNumDice; ++i)
            init.push_back(static_cast<int>(turn.initial_dice[i]));
        t["initial_dice"] = init;

        json holds = json::array();
        for (size_t hi = 0; hi < turn.hold_masks.size(); ++hi) {
            json h;
            uint8_t mask = turn.hold_masks[hi];
            h["mask"] = static_cast<int>(mask);
            json da = json::array();
            json hf = json::array();
            if (hi < turn.dice_after_hold.size()) {
                const int8_t* prev_dice = (hi == 0) ? turn.initial_dice : turn.dice_after_hold[hi - 1].data();
                
                int8_t held_values[kNumDice];
                int held_count = 0;
                for (int i = 0; i < kNumDice; ++i) {
                    if ((mask >> i) & 1) {
                        held_values[held_count++] = prev_dice[i];
                    }
                }

                bool mapped[kNumDice] = {false};
                for (int i = 0; i < kNumDice; ++i) {
                    da.push_back(static_cast<int>(turn.dice_after_hold[hi][i]));
                    bool is_held = false;
                    for (int j = 0; j < held_count; ++j) {
                        if (!mapped[j] && held_values[j] == turn.dice_after_hold[hi][i]) {
                            mapped[j] = true;
                            is_held = true;
                            break;
                        }
                    }
                    hf.push_back(is_held);
                }
            }
            h["dice_after"]  = da;
            h["held_flags"]  = hf;
            holds.push_back(h);
        }
        t["holds"] = holds;

        t["placement"] = {
            {"column",      static_cast<int>(turn.placement.column)},
            {"row",         static_cast<int>(turn.placement.row)},
            {"column_name", column_name(static_cast<int>(turn.placement.column))},
            {"row_name",    row_name(static_cast<int>(turn.placement.row))}
        };
        t["score"] = static_cast<int>(turn.score);

        // Debug block (omitted entirely when debug data is absent).
        if (!turn.hold_evals.empty() || !turn.placement_evals.empty() ||
            turn.has_board_nn_value) {
            json dbg;

            json he_arr = json::array();
            for (const auto& step_evals : turn.hold_evals) {
                json step = json::array();
                for (const auto& cand : step_evals) {
                    json c;
                    c["mask"]           = static_cast<int>(cand.hold_mask);
                    c["expected_value"] = static_cast<float>(cand.expected_value);
                    // Expand mask to per-die flags for easy frontend rendering.
                    json flags = json::array();
                    for (int i = 0; i < kNumDice; ++i)
                        flags.push_back(static_cast<bool>((cand.hold_mask >> i) & 1));
                    c["held_flags"] = flags;
                    step.push_back(c);
                }
                he_arr.push_back(step);
            }
            dbg["hold_evals"] = he_arr;

            json pe_arr = json::array();
            for (const auto& cand : turn.placement_evals) {
                json c;
                c["column"]      = static_cast<int>(cand.placement.column);
                c["row"]         = static_cast<int>(cand.placement.row);
                c["column_name"] = column_name(static_cast<int>(cand.placement.column));
                c["row_name"]    = row_name(static_cast<int>(cand.placement.row));
                c["score"]       = static_cast<int>(cand.score);
                c["eval_value"]  = static_cast<float>(cand.eval_value);
                pe_arr.push_back(c);
            }
            dbg["placement_evals"] = pe_arr;

            if (turn.has_board_nn_value)
                dbg["board_nn_value"] = turn.board_nn_value;

            t["debug"] = dbg;
        }

        hist.push_back(t);
    }
    j["history"] = hist;

    // Current board NN evaluation (debug mode only, if NN is available).
    if (session.has_current_board_nn)
        j["current_board_nn_value"] = session.current_board_nn_value;

    // Player types
    auto pt_str = [](PlayerType pt) -> const char* {
        switch (pt) {
            case PlayerType::kHuman:         return "human";
            case PlayerType::kHeuristic:     return "heuristic_v1";
            case PlayerType::kHeuristicV2:   return "heuristic_v2";
            case PlayerType::kHeuristicV3:   return "heuristic_v3";
            case PlayerType::kHeuristicV4:   return "heuristic_v4";
            case PlayerType::kHeuristicV5:   return "heuristic_v5";
            case PlayerType::kHeuristicV6:   return "heuristic_v6";
            case PlayerType::kHeuristicV7:   return "heuristic_v7";
            case PlayerType::kHeuristicV8:   return "heuristic_v8";
            case PlayerType::kHeuristicV9:   return "heuristic_v9";
            case PlayerType::kHeuristicV10:  return "heuristic_v10";
            case PlayerType::kHeuristicV11:  return "heuristic_v11";
            case PlayerType::kHeuristicV12:  return "heuristic_v12";
            case PlayerType::kHeuristicV13:  return "heuristic_v13";
            case PlayerType::kHeuristicV14:  return "heuristic_v14";
            case PlayerType::kHeuristicV15:  return "heuristic_v15";
            case PlayerType::kNNSolver:      return "nn";
            case PlayerType::kMCRollout:     return "mc";
        }
        return "unknown";
    };
    j["player_types"] = json::array({pt_str(session.player_types[0]),
                                      pt_str(session.player_types[1])});

    return j;
}

// ---------------------------------------------------------------------------
// options_to_json
// ---------------------------------------------------------------------------

json options_to_json(const std::vector<std::pair<Placement, int>>& options,
                     bool can_reroll_flag) {
    json j;
    j["can_reroll"] = can_reroll_flag;

    json placements = json::array();
    for (const auto& [pl, score] : options) {
        json p;
        p["column"]      = static_cast<int>(pl.column);
        p["row"]         = static_cast<int>(pl.row);
        p["score"]       = score;
        p["column_name"] = column_name(static_cast<int>(pl.column));
        p["row_name"]    = row_name(static_cast<int>(pl.row));
        placements.push_back(p);
    }
    j["placements"] = placements;
    return j;
}
