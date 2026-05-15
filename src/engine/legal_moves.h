#pragma once

#include "engine/constants.h"
#include "engine/board_state.h"
#include "engine/game_context.h"

// ---------------------------------------------------------------------------
// Legal placement generation
// ---------------------------------------------------------------------------

/// Check if a row has a filled neighbour in the given column for the given player.
/// Adjacency wraps: row 0 and row 12 are neighbours (circular).
template <typename Traits>
bool has_filled_neighbor(int player, int column, int row,
                         const BoardStateT<Traits>& board);

/// Rebuild both legal placement caches (legal_all and legal_no_turbo) for a
/// player from scratch. Used during initialisation and for test verification.
template <typename Traits>
void rebuild_legal_placements(int player,
                              const BoardStateT<Traits>& board,
                              GameContextT<Traits>& ctx);

/// Incrementally update both caches after a placement was made.
/// More efficient than a full rebuild — only modifies affected entries.
template <typename Traits>
void update_legal_placements_after_move(int player, int column, int row,
                                        const BoardStateT<Traits>& board,
                                        GameContextT<Traits>& ctx);
