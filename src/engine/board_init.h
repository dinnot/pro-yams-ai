#pragma once

#include "engine/board_state.h"
#include "engine/game_context.h"

// Forward declare RNG (defined in rng.h, implemented in Task 03)
class RNG;

// ---------------------------------------------------------------------------
// Board and context initialisation
// ---------------------------------------------------------------------------

/// Initialize a fresh board: all cells empty, coefficients randomly shuffled,
/// starting player randomly chosen.
///
/// @param board  Output: the initialized board
/// @param rng    Random engine for coefficient shuffling and starting player
void init_board(BoardState& board, RNG& rng);

/// Initialize the GameContext to match a freshly initialized (empty) BoardState.
/// Sets up initial legal placements for both players, zeroes all caches.
///
/// @param ctx   Output: initialized context
/// @param board A freshly initialized board (all cells empty)
void init_context(GameContext& ctx, const BoardState& board);
