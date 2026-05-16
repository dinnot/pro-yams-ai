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
//   - Calls update_legal_placements_after_move (unless `update_legal_cache`
//     is false — set false when the resulting context is consumed for
//     evaluation only and the legal-move caches are never read again).
// ---------------------------------------------------------------------------
template <typename Traits>
void apply_placement(int player, int column, int row, int score,
                     BoardStateT<Traits>& board,
                     GameContextT<Traits>& ctx,
                     bool update_legal_cache = true);
