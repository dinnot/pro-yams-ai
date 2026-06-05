#include "ui/json_serialization.h"

#include "engine/constants.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// build_holds_json — serialize a turn's reroll steps (committed or in-progress)
// as [{ mask, dice_after, held_flags }]. held_flags marks which dice of the
// post-roll set were carried over from the pre-roll dice (greedy match by
// value), matching the reveal the frontend animates. Shared by the history
// loop and the live (in-progress) turn.
// ---------------------------------------------------------------------------
static json build_holds_json(
        const int8_t* initial_dice,
        const std::vector<uint8_t>& hold_masks,
        const std::vector<std::array<int8_t, kNumDice>>& dice_after_hold) {
    json holds = json::array();
    for (size_t hi = 0; hi < hold_masks.size(); ++hi) {
        json h;
        uint8_t mask = hold_masks[hi];
        h["mask"] = static_cast<int>(mask);
        json da = json::array();
        json hf = json::array();
        if (hi < dice_after_hold.size()) {
            const int8_t* prev_dice = (hi == 0) ? initial_dice
                                                : dice_after_hold[hi - 1].data();
            int8_t held_values[kNumDice];
            int held_count = 0;
            for (int i = 0; i < kNumDice; ++i)
                if ((mask >> i) & 1) held_values[held_count++] = prev_dice[i];

            bool mapped[kNumDice] = {false};
            for (int i = 0; i < kNumDice; ++i) {
                da.push_back(static_cast<int>(dice_after_hold[hi][i]));
                bool is_held = false;
                for (int j = 0; j < held_count; ++j) {
                    if (!mapped[j] && held_values[j] == dice_after_hold[hi][i]) {
                        mapped[j] = true;
                        is_held = true;
                        break;
                    }
                }
                hf.push_back(is_held);
            }
        }
        h["dice_after"] = da;
        h["held_flags"] = hf;
        holds.push_back(h);
    }
    return holds;
}

// ---------------------------------------------------------------------------
// game_state_to_json<Traits>
// ---------------------------------------------------------------------------

template <typename Traits>
json game_state_to_json(const GameSessionT<Traits>& session) {
    const auto& gs = session.state;
    const auto& bs = gs.board;

    json j;
    j["session_id"]        = session.session_id;
    j["game_over"]         = session.game_over;
    j["result"]            = session.game_over
                             ? json(session.result) : json(nullptr);
    j["current_player"]    = static_cast<int>(bs.current_player);
    j["rolls_left"]        = static_cast<int>(gs.rolls_left);
    j["waiting_for_human"] = session.waiting_for_human;
    // "Lucky Yams" bonus active for the current dice (first-roll five-of-a-kind
    // with the rule enabled). The frontend shows a banner and the per-cell
    // option scores are already the wildcard maxima.
    j["yams_bonus"]        = yams_bonus_active<Traits>(gs);

    // Variant metadata so the frontend can pick the right layout.
    j["game_variant"]      = (Traits::kNumPlayers == 4) ? "2v2" : "1v1";
    j["num_players"]       = Traits::kNumPlayers;

    // Shared-multiplayer disconnect notice: seats where a human teammate timed
    // out and the AI took over. Empty for normal (single-client) games. Lets the
    // surviving player's UI show a "teammate left" banner.
    json takeover = json::array();
    for (int p = 0; p < Traits::kNumPlayers; ++p)
        if (session.seat_nn_takeover[p]) takeover.push_back(p);
    j["nn_takeover_seats"] = takeover;

    // Human seats that have gone silent past the grace period — candidates the
    // surviving teammate may choose to switch to the AI (never automatic).
    json disconnected = json::array();
    for (int p = 0; p < Traits::kNumPlayers; ++p)
        if (session.seat_disconnected[p]) disconnected.push_back(p);
    j["disconnected_seats"] = disconnected;

    json dice = json::array();
    for (int i = 0; i < kNumDice; ++i)
        dice.push_back(static_cast<int>(gs.dice[i]));
    j["dice"] = dice;

    json coefs = json::array();
    for (int c = 0; c < kNumColumns; ++c)
        coefs.push_back(static_cast<int>(bs.coefficients[c]));
    j["coefficients"] = coefs;

    json boards = json::object();
    for (int p = 0; p < Traits::kNumPlayers; ++p) {
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

    json hist = json::array();
    for (const auto& turn : session.history) {
        json t;
        t["player"] = turn.player;

        json init = json::array();
        for (int i = 0; i < kNumDice; ++i)
            init.push_back(static_cast<int>(turn.initial_dice[i]));
        t["initial_dice"] = init;

        t["holds"] = build_holds_json(turn.initial_dice, turn.hold_masks,
                                      turn.dice_after_hold);

        t["placement"] = {
            {"column",      static_cast<int>(turn.placement.column)},
            {"row",         static_cast<int>(turn.placement.row)},
            {"column_name", column_name(static_cast<int>(turn.placement.column))},
            {"row_name",    row_name(static_cast<int>(turn.placement.row))}
        };
        t["score"] = static_cast<int>(turn.score);

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

    // In-progress turn (live streaming for shared two-human games). While the
    // server waits on a human, this exposes the partial turn — the initial roll,
    // any committed rerolls, and the tentative (uncommitted) hold preview — so a
    // teammate can watch the turn unfold instead of seeing it all at once when
    // it finally lands in history. The top-level `dice`/`rolls_left` already
    // give the current dice; this adds the per-step holds and the preview mask.
    if (session.waiting_for_human) {
        const auto& ct = session.current_turn;
        json lt;
        lt["player"] = ct.player;
        json init = json::array();
        for (int i = 0; i < kNumDice; ++i)
            init.push_back(static_cast<int>(ct.initial_dice[i]));
        lt["initial_dice"] = init;
        lt["holds"]        = build_holds_json(ct.initial_dice, ct.hold_masks,
                                              ct.dice_after_hold);
        lt["preview_mask"] =
            (session.live_hold_player == static_cast<int>(bs.current_player))
                ? static_cast<int>(session.live_hold_mask) : 0;
        j["live_turn"] = lt;
    }

    if (session.has_current_board_nn)
        j["current_board_nn_value"] = session.current_board_nn_value;

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
            case PlayerType::kHeuristicV16:  return "heuristic_v16";
            case PlayerType::kHeuristicV17:  return "heuristic_v17";
            case PlayerType::kNNSolver:      return "nn";
            case PlayerType::kMCRollout:     return "mc";
        }
        return "unknown";
    };
    json pt_arr = json::array();
    for (int p = 0; p < Traits::kNumPlayers; ++p)
        pt_arr.push_back(pt_str(session.player_types[p]));
    j["player_types"] = pt_arr;

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

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template json game_state_to_json<Yams1v1>(const GameSessionT<Yams1v1>&);
template json game_state_to_json<Yams2v2>(const GameSessionT<Yams2v2>&);
