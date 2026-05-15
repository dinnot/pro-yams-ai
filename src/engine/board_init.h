#pragma once

#include "engine/board_state.h"
#include "engine/game_context.h"

// Forward declare RNG (defined in rng.h)
class RNG;

// ---------------------------------------------------------------------------
// Board and context initialisation
// ---------------------------------------------------------------------------

/// Initialize a fresh board: all cells empty, coefficients randomly shuffled,
/// starting player randomly chosen.
template <typename Traits>
void init_board(BoardStateT<Traits>& board, RNG& rng);

/// Initialize the GameContext to match a freshly initialized (empty) BoardState.
/// Sets up initial legal placements for all players, zeroes all caches.
template <typename Traits>
void init_context(GameContextT<Traits>& ctx, const BoardStateT<Traits>& board);
