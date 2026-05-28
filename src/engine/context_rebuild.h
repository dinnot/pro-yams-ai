#pragma once

#include "engine/board_state.h"
#include "engine/game_context.h"

// ---------------------------------------------------------------------------
// rebuild_context_from_board — reconstruct a complete GameContext by scanning
// all cells of a BoardState.
//
// Used by MC rollouts which start from a cloned board position and cannot
// rely on incrementally maintained context. Slower than incremental updates
// but O(kTotalCells) — trivial compared to a full game simulation.
// ---------------------------------------------------------------------------
template <typename Traits>
void rebuild_context_from_board(const BoardStateT<Traits>& board,
                                GameContextT<Traits>& ctx);
