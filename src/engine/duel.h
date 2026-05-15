#pragma once

#include "engine/board_state.h"
#include "engine/game_context.h"

// ---------------------------------------------------------------------------
// Upper section bonus (per column, per player).
// Applied to the sum of rows 0-5 in a completed column.
// ---------------------------------------------------------------------------
int upper_section_bonus(int sum);

// ---------------------------------------------------------------------------
// Crush multiplier calculation.
// Returns 1-5 based on ratio of my_raw to opp_raw.
// The crush is directional: caller checks both directions and takes the max.
// ---------------------------------------------------------------------------
int crush_multiplier(int my_raw, int opp_raw);

// ---------------------------------------------------------------------------
// compute_duel — calculate the final game result.
//
// Must only be called on a completed board (is_terminal() == true).
// Returns total duel points from Team 0's perspective:
//   > 0  → Team 0 wins
//   < 0  → Team 1 wins
//   == 0 → draw
//
// 1v1: Team 0 = {P0}, Team 1 = {P1}. Single pairing (0, 1) per column.
// 2v2: Team 0 = {P0, P2}, Team 1 = {P1, P3}. Four cross-team pairings per
//       column. Crush multiplier and clean-column bonus value are computed
//       independently per pairing.
// ---------------------------------------------------------------------------
template <typename Traits>
int compute_duel(const BoardStateT<Traits>& board,
                 const GameContextT<Traits>& ctx);
