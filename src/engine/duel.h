#pragma once

#include "engine/board_state.h"
#include "engine/game_context.h"

// ---------------------------------------------------------------------------
// Upper section bonus (per column, per player).
// Applied to the sum of rows 0-5 in a completed column.
// ---------------------------------------------------------------------------
int upper_section_bonus(int sum);

// ---------------------------------------------------------------------------
// Crush multiplier calculation.
// Returns 1-5 based on ratio of my_raw to opp_raw.
// The crush is directional: caller checks both directions and takes the max.
// ---------------------------------------------------------------------------
int crush_multiplier(int my_raw, int opp_raw);

// ---------------------------------------------------------------------------
// compute_duel — calculate the final game result.
//
// Must only be called on a completed board (is_terminal() == true).
// Returns total duel points from player 0's perspective:
//   > 0  → player 0 wins
//   < 0  → player 1 wins
//   == 0 → draw
//
// @param board  Fully filled board (all 156 cells)
// @param ctx    Game context (upper sums, scratch tracking)
// @return Total duel points for player 0
// ---------------------------------------------------------------------------
int compute_duel(const BoardState& board, const GameContext& ctx);
