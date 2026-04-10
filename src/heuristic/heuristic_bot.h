#pragma once

#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_state.h"
#include "engine/rng.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// Heuristic evaluation — score × column coefficient.
//
// Deliberately simple: captures the insight that higher scores in higher-
// coefficient columns are better, without any strategic reasoning.
// ---------------------------------------------------------------------------
void heuristic_evaluate(const BoardState& board, const GameContext& ctx,
                        const AfterstateRequest* requests, int request_count,
                        double* evs);

// ---------------------------------------------------------------------------
// Play one complete turn using the heuristic bot.
//
// Calls solver_get_requests → heuristic_evaluate → solver_resolve in a loop
// until a placement is made. Makes hold/reroll decisions, then applies the
// chosen placement and advances to the next player's turn.
//
// On entry:  dice are already rolled (start_turn has been called).
// On exit:   the placement has been applied; next player's dice are rolled.
// ---------------------------------------------------------------------------
void heuristic_play_turn(GameState& state, GameContext& ctx,
                         const PrecomputedTables& tables,
                         SolverBuffers& buffers, RNG& rng);

// ---------------------------------------------------------------------------
// Play a complete game using the heuristic bot for both players.
//
// Returns the raw duel score from player 0's perspective (see compute_duel).
// ---------------------------------------------------------------------------
int play_heuristic_game(RNG& rng, const PrecomputedTables& tables);
