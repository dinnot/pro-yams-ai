#include "heuristic/heuristic_bot.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "engine/duel.h"
#include "engine/game_flow.h"
#include "engine/placement.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "solver/dp_eval.h"

// ---------------------------------------------------------------------------
// V1 — score × column coefficient.
// ---------------------------------------------------------------------------
void heuristic_evaluate(const BoardState& board, const GameContext& /*ctx*/,
                        const AfterstateRequest* requests, int request_count,
                        double* evs) {
    for (int i = 0; i < request_count; ++i) {
        int col = requests[i].placement.column;
        int row = requests[i].placement.row;
        int score = requests[i].score;
        if (row > 0 && row < kRow6s) {
            if (score >= (row + 1) * 3) score += 6;
        }
        if (score == 0) {
            score = (18 - board.coefficients[col]) * 0.1;
            if (row <= kRow6s) score *= 2;
        }
        evs[i] = static_cast<double>(score) * board.coefficients[col];
    }
}

// ---------------------------------------------------------------------------
// V2 — DP-driven expected global duel margin.
// ---------------------------------------------------------------------------
void heuristic_evaluate_v2(const BoardState& base_board, const GameContext& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables) {
    const int p_me  = base_board.current_player;
    const int p_opp = 1 - p_me;
    const DPTables& dp = tables.dp_tables;

    for (int i = 0; i < request_count; ++i) {
        // Apply candidate placement to a clone of the board+ctx.
        BoardState b = base_board;
        GameContext c = base_ctx;
        apply_placement(p_me,
                        requests[i].placement.column,
                        requests[i].placement.row,
                        requests[i].score, b, c);

        const int filled_me  = count_filled_cells(b, p_me);
        const int filled_opp = count_filled_cells(b, p_opp);

        double expected_global_margin = 0.0;

        for (int col = 0; col < kNumColumns; ++col) {
            const int empty_me  = count_empty_cells(b, p_me,  col);
            const int empty_opp = count_empty_cells(b, p_opp, col);

            // Approximate turns remaining for this column. The minimum
            // guarantees enough turns to fill the column; the per-section
            // amortization applies inside get_E_raw via apportion_turns().
            const int T_me  = std::max(empty_me,
                                       empty_me  + (78 - filled_me)  / 6);
            const int T_opp = std::max(empty_opp,
                                       empty_opp + (78 - filled_opp) / 6);

            const float E_me  = get_E_raw(p_me,  col, T_me,  b, c, dp);
            const float E_opp = get_E_raw(p_opp, col, T_opp, b, c, dp);

            const int e_me_i  = static_cast<int>(std::lround(E_me));
            const int e_opp_i = static_cast<int>(std::lround(E_opp));
            const int crush_me  = crush_multiplier(e_me_i,  e_opp_i);
            const int crush_opp = crush_multiplier(e_opp_i, e_me_i);
            const int active_crush = std::max(crush_me, crush_opp);

            // Clean column bonus expected value.
            const float P_clean_me  = get_P_clean(p_me,  col, T_me,  b, c, dp);
            const float P_clean_opp = get_P_clean(p_opp, col, T_opp, b, c, dp);
            const float bonus_val = (active_crush > 1) ? 100.0f : 200.0f;
            const float E_clean_me  = P_clean_me  * bonus_val;
            const float E_clean_opp = P_clean_opp * bonus_val;

            const double margin = ((static_cast<double>(E_me)  + E_clean_me) -
                                   (static_cast<double>(E_opp) + E_clean_opp))
                                * active_crush
                                * b.coefficients[col];
            expected_global_margin += margin;
        }

        evs[i] = expected_global_margin;
    }
}

// ---------------------------------------------------------------------------
// Play one complete turn.
// ---------------------------------------------------------------------------
void heuristic_play_turn(GameState& state, GameContext& ctx,
                         const PrecomputedTables& tables,
                         SolverBuffers& buffers, RNG& rng,
                         HeuristicVersion version) {
    buffers.dp_computed = false;

    // Step 1: get afterstate requests (static for the turn).
    solver_get_requests(state, ctx, tables, buffers);
    assert(buffers.request_count > 0);

    // Step 2: evaluate with heuristic (static for the turn).
    if (version == HeuristicVersion::V2) {
        heuristic_evaluate_v2(state.board, ctx,
                              buffers.requests, buffers.request_count,
                              buffers.evs, tables);
    } else {
        heuristic_evaluate(state.board, ctx,
                           buffers.requests, buffers.request_count, buffers.evs);
    }

    while (true) {
        // Step 3: resolve best action (uses cached DP tables on rerolls).
        SolverResult result = solver_resolve_greedy(state, ctx, tables, buffers);

        if (result.should_place) {
            perform_placement(state, ctx, result.placement.column,
                              result.placement.row, rng);
            return;
        }

        // Hold and reroll.
        if (!can_reroll(state, ctx)) {
            // Safety break to prevent infinite loop.
            int current_id = get_dice_state_id(state.dice, tables);
            int16_t req_idx = buffers.stop_request_idx[current_id];
            if (req_idx < 0) req_idx = 0;
            perform_placement(state, ctx, buffers.requests[req_idx].placement.column,
                              buffers.requests[req_idx].placement.row, rng);
            return;
        }
        perform_reroll(state, result.hold_mask, rng);
    }
}

// ---------------------------------------------------------------------------
// Play a complete game.
// ---------------------------------------------------------------------------
int play_heuristic_game(RNG& rng, const PrecomputedTables& tables,
                        HeuristicVersion version) {
    GameState state;
    GameContext ctx;
    SolverBuffers buffers;

    init_game(state, ctx, rng);

    while (!is_terminal(state.board)) {
        heuristic_play_turn(state, ctx, tables, buffers, rng, version);
    }

    return get_game_result(state, ctx);
}
