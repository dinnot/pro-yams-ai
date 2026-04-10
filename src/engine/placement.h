#pragma once

#include "engine/constants.h"
#include "engine/board_state.h"
#include "engine/game_context.h"

// ---------------------------------------------------------------------------
// apply_placement — the main board mutation function.
//
// Applies a score to a cell and updates ALL cached state in GameContext:
//   - Writes the cell value
//   - Increments cells_filled
//   - Updates golden_max (and recalculates if mutual SS/LS destruction occurs)
//   - Updates upper_sum (rows 0-5)
//   - Updates SS/LS scratch flags and applies mutual destruction
//   - Updates lower_has_scratch
//   - Calls update_legal_placements_after_move
//
// @param player  The placing player (0 or 1)
// @param column  Target column (0-5)
// @param row     Target row (0-12)
// @param score   Score to place (0 = scratch, >0 = valid score)
// @param board   Board state to modify
// @param ctx     Game context to update
// ---------------------------------------------------------------------------
void apply_placement(int player, int column, int row, int score,
                     BoardState& board, GameContext& ctx);
