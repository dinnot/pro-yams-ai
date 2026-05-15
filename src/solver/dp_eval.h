#pragma once

#include <cstdint>

#include "engine/board_state.h"
#include "engine/game_context.h"
#include "solver/dp_tables.h"

// ---------------------------------------------------------------------------
// dp_eval — shared DP-table-driven column expectations for both the tensor
// generator and the V2 heuristic bot.
//
// All helpers are pure functions of (board, ctx, dp_tables); no state.
// ---------------------------------------------------------------------------

// Snap a Golden Rule maximum (gmax) to the nearest legal Sc constraint value
// at-or-above gmax.  Returns 0 if there is no constraint.
int8_t snap_gmax(int gmax, const int8_t* vals, int count);

// Map a column index to the DP variant used by the tables.
Variant get_variant(int col);

// Build the upper/middle/lower Sc[] arrays for a column from the live board
// state. Sets EU, EM, EL to the count of empty cells in each section.
template <typename Traits>
void build_Sc(int p, int col, const BoardStateT<Traits>& board,
              const GameContextT<Traits>& ctx,
              int8_t Sc_U[6], int8_t Sc_M[2], int8_t Sc_L[5],
              int& EU, int& EM, int& EL);

// Apportion T turns across upper/middle/lower DP queries proportional to
// the count of empty cells in each section.
void apportion_turns(int T, int EU, int EM, int EL,
                     int& TU, int& TM, int& TL);

// Compute expected raw column score for player p in column col over T turns.
// Includes upper-bonus expectation (from get_upper_ev), expected
// middle/lower points, and already-filled cell scores in the lower section
// (rows 6-12).
template <typename Traits>
float get_E_raw(int p, int col, int T, const BoardStateT<Traits>& board,
                const GameContextT<Traits>& ctx, const DPTables& dp);

// Compute the expected "clean column" probability for player p in column
// col over T turns: P(upper_sum reaches 60) * P(mid no-scratch) *
// P(low no-scratch). Returns 0 if a lower scratch already exists.
template <typename Traits>
float get_P_clean(int p, int col, int T, const BoardStateT<Traits>& board,
                  const GameContextT<Traits>& ctx, const DPTables& dp);

// Compute the expected raw column variance Var(X) = E[X^2] - E[X]^2.
template <typename Traits>
float get_E_raw_var(int p, int col, int T, const BoardStateT<Traits>& board,
                    const GameContextT<Traits>& ctx, const DPTables& dp);
