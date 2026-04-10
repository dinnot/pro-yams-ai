#pragma once

#include "engine/board_state.h"
#include "engine/game_context.h"

// ---------------------------------------------------------------------------
// rebuild_context_from_board — reconstruct a complete GameContext by scanning
// all cells of a BoardState.
//
// Used by MC rollouts which start from a cloned board position and cannot
// rely on incrementally maintained context. Slower than incremental updates
// but O(156) — trivial compared to a full game simulation.
// ---------------------------------------------------------------------------
void rebuild_context_from_board(const BoardState& board, GameContext& ctx);
