#pragma once

#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_state.h"
#include "engine/rng.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// Heuristic versions.
//   V1 — score × column coefficient (greedy, no DP).
//   V2 — DP-driven expected global duel margin (uses dp_tables).
// ---------------------------------------------------------------------------
enum class HeuristicVersion {
    V1 = 1,
    V2 = 2,
};

// ---------------------------------------------------------------------------
// V1 evaluator — score × column coefficient.
//
// Captures the insight that higher scores in higher-coefficient columns are
// better, without any strategic reasoning.
// ---------------------------------------------------------------------------
void heuristic_evaluate(const BoardState& board, const GameContext& ctx,
                        const AfterstateRequest* requests, int request_count,
                        double* evs);

// ---------------------------------------------------------------------------
// V2 evaluator — DP-driven expected global duel margin.
//
// For each candidate placement, simulate it on a clone, then sum across all
// 6 columns the expected (E_me - E_opp) * coefficient * crush_multiplier,
// including a clean-column bonus contribution. Returns absolute expected
// duel points (can be negative). Heavy negative penalty applied when the
// candidate is a voluntary scratch (score == 0).
// ---------------------------------------------------------------------------
void heuristic_evaluate_v2(const BoardState& base_board, const GameContext& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables);

// ---------------------------------------------------------------------------
// Play one complete turn using the heuristic bot.
//
// Calls solver_get_requests → heuristic_evaluate(_v2) → solver_resolve in a
// loop until a placement is made. Makes hold/reroll decisions, then applies
// the chosen placement and advances to the next player's turn.
//
// On entry:  dice are already rolled (start_turn has been called).
// On exit:   the placement has been applied; next player's dice are rolled.
// ---------------------------------------------------------------------------
void heuristic_play_turn(GameState& state, GameContext& ctx,
                         const PrecomputedTables& tables,
                         SolverBuffers& buffers, RNG& rng,
                         HeuristicVersion version = HeuristicVersion::V2);

// ---------------------------------------------------------------------------
// Play a complete game using the heuristic bot for both players.
//
// Returns the raw duel score from player 0's perspective (see compute_duel).
// ---------------------------------------------------------------------------
int play_heuristic_game(RNG& rng, const PrecomputedTables& tables,
                        HeuristicVersion version = HeuristicVersion::V2);
