#pragma once

#include "engine/constants.h"
#include "engine/board_state.h"
#include "engine/game_context.h"

// ---------------------------------------------------------------------------
// Legal placement generation
// ---------------------------------------------------------------------------

/// Check if a row has a filled neighbour in the given column for the given player.
/// Adjacency wraps: row 0 and row 12 are neighbours (circular).
bool has_filled_neighbor(int player, int column, int row, const BoardState& board);

/// Rebuild both legal placement caches (legal_all and legal_no_turbo) for a
/// player from scratch. Used during initialisation and for test verification.
void rebuild_legal_placements(int player, const BoardState& board, GameContext& ctx);

/// Incrementally update both caches after a placement was made.
/// More efficient than a full rebuild — only modifies affected entries.
///
/// @param player The player who just placed
/// @param column The column of the placement
/// @param row    The row of the placement
/// @param board  Board state AFTER the placement
/// @param ctx    Context to update
void update_legal_placements_after_move(int player, int column, int row,
                                         const BoardState& board, GameContext& ctx);
