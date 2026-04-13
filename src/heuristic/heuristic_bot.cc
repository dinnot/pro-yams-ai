#include "heuristic/heuristic_bot.h"

#include <cassert>

#include "engine/duel.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"

// ---------------------------------------------------------------------------
// Heuristic evaluation — score × column coefficient.
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
            // else if (board.coefficients[col] >= 14) {
            //     score = score / (row + 1);
            // }
        }
        if (score == 0) {
            score = (18 - board.coefficients[col]) * 0.1;
            if (row <= kRow6s) score *= 2;
        }
        evs[i] = static_cast<double>(score) * board.coefficients[col];
    }
}

// ---------------------------------------------------------------------------
// Play one complete turn.
// ---------------------------------------------------------------------------
void heuristic_play_turn(GameState& state, GameContext& ctx,
                         const PrecomputedTables& tables,
                         SolverBuffers& buffers, RNG& rng) {
    buffers.dp_computed = false;

    // Step 1: get afterstate requests (static for the turn).
    solver_get_requests(state, ctx, tables, buffers);
    assert(buffers.request_count > 0);

    // Step 2: evaluate with heuristic (static for the turn).
    heuristic_evaluate(state.board, ctx,
                       buffers.requests, buffers.request_count, buffers.evs);

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
            // If we get here, the solver thought we should reroll but the engine says no.
            // Force a placement using the solver's fallback to prevent hang.
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
int play_heuristic_game(RNG& rng, const PrecomputedTables& tables) {
    GameState state;
    GameContext ctx;
    SolverBuffers buffers;

    init_game(state, ctx, rng);

    while (!is_terminal(state.board)) {
        heuristic_play_turn(state, ctx, tables, buffers, rng);
    }

    return get_game_result(state, ctx);
}
