#pragma once

#include <cstdint>
#include "engine/constants.h"
#include "engine/board_state.h"
#include "engine/game_context.h"

// ---------------------------------------------------------------------------
// Dice utility functions (variant-independent)
// ---------------------------------------------------------------------------

/// Sort dice in ascending order in-place (insertion sort — tiny array).
void sort_dice(int8_t dice[kNumDice]);

/// Compute frequency counts for each face value.
/// counts[0] is unused; counts[1..6] hold the count for each face.
void dice_counts(const int8_t dice[kNumDice], int counts[7]);

/// Compute sum of all dice.
int dice_sum(const int8_t dice[kNumDice]);

// ---------------------------------------------------------------------------
// Score calculation
// ---------------------------------------------------------------------------

/// Calculate the score a player would receive for placing in a specific cell.
///
/// Applies:
///   - Dice-based scoring (raw value from the dice)
///   - Golden Rule (score must be >= golden_max for that col/row, else 0)
///   - SS/LS interlock (mutual scratch forcing, ordering constraints)
///
/// Does NOT check column order constraints — that is get_legal_placements' job.
/// The caller is responsible for ensuring (column, row) is a legal placement.
template <typename Traits>
int calculate_score(int row, const int8_t dice[kNumDice],
                    int player, int column,
                    const BoardStateT<Traits>& board,
                    const GameContextT<Traits>& ctx);

// ---------------------------------------------------------------------------
// Board query utilities
// ---------------------------------------------------------------------------

/// Check if the game is over (all kTotalCells cells filled).
template <typename Traits>
bool is_terminal(const BoardStateT<Traits>& board);

/// Get the number of empty cells remaining for a player.
template <typename Traits>
int cells_remaining(const BoardStateT<Traits>& board, int player);

/// Get the number of empty cells remaining in a specific column for a player.
template <typename Traits>
int column_cells_remaining(const BoardStateT<Traits>& board, int player, int column);
