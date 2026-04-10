#pragma once

#include <cstdint>
#include "engine/constants.h"
#include "engine/board_state.h"
#include "engine/game_context.h"

// ---------------------------------------------------------------------------
// Dice utility functions
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
///
/// @param row    Target row (0-12)
/// @param dice   5 dice values (1-6), need not be sorted
/// @param player The placing player (0 or 1)
/// @param column Target column (0-5)
/// @param board  Current board state
/// @param ctx    Current game context (golden_max, ss/ls status)
/// @return Score to place (0 = scratch, >0 = valid score)
int calculate_score(int row, const int8_t dice[kNumDice],
                    int player, int column,
                    const BoardState& board, const GameContext& ctx);

// ---------------------------------------------------------------------------
// Board query utilities
// ---------------------------------------------------------------------------

/// Check if the game is over (all 156 cells filled).
bool is_terminal(const BoardState& board);

/// Get the number of empty cells remaining for a player.
int cells_remaining(const BoardState& board, int player);

/// Get the number of empty cells remaining in a specific column for a player.
int column_cells_remaining(const BoardState& board, int player, int column);
