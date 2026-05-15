#pragma once

#include "engine/board_state.h"
#include "engine/game_context.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"  // AfterstateRequest

// ---------------------------------------------------------------------------
// State observation tensor for the neural network (V2.1 layout).
//
// Encodes a board position (after placement) from one player's perspective
// into a flat float array.  No libtorch dependency — produces raw floats that
// the model library wraps into torch::Tensor.
//
// Layout (986 features):
//   Group A (312): per-player × per-column × per-row cell values (2 per cell)
//   Group B (108): per-player × per-column derived + my-perspective duel features
//   Group C (156): per-player × per-column × per-row 1-turn non-scratch probability
//   Group D  (14): global aggregates and phase flags
//   Group E (216): per-player × per-column × {T_min,T_mid,T_max} upper P/EV
//   Group F (180): per-player × per-column × {T_min,T_mid,T_max} mid/low P/EV + P_clean
// ---------------------------------------------------------------------------

/// 1v1 backward-compat alias — equals Yams1v1::kTensorSize. Remove at end of
/// migration once no caller depends on the bare name.
constexpr int kTensorSize = Yams1v1::kTensorSize;

/// Maximum possible score per row (used for normalisation denominators).
/// Indexed by row (0–12): 1s, 2s, 3s, 4s, 5s, 6s, SS, LS, FH, K, STR, U8, Y
constexpr int kMaxScorePerRow[kNumRows] = {
    5, 10, 15, 20, 25, 30,   // upper rows 0-5
    29, 30,                  // SS, LS (rows 6-7)
    50, 54, 50, 75, 100      // FH, K, STR, U8, Y (rows 8-12)
};

// ---------------------------------------------------------------------------
// Main tensor generation function.
//
// The output is in CANONICAL VIEW relative to the perspective player:
//   1v1: [Active, Opp]
//   2v2: [Active, NextOpp, Teammate, PrevOpp]
// Per-player feature groups iterate this canonical order; per-pairing
// feature groups iterate Traits::kCanonicalPairing{T0,T1} in order. The
// network never sees raw seat indices — this is the rotational-invariance
// guarantee from Task 5.2.
//
// @param board   Board state to encode
// @param ctx     Game context (cached derived data)
// @param player  Perspective player (active seat)
// @param tables  Precomputed tables (score tables, probability tables)
// @param out     Output buffer of exactly Traits::kTensorSize floats
// ---------------------------------------------------------------------------
template <typename Traits>
void generate_tensor(const BoardStateT<Traits>& board,
                     const GameContextT<Traits>& ctx,
                     int player, const PrecomputedTables& tables,
                     float* out);

// ---------------------------------------------------------------------------
// Batch tensor generation for afterstates.
//
// For each request, applies the placement as a simple cell-write to a cloned
// board (without full GameContext cache maintenance) and generates a tensor.
// Writes tensors contiguously: out[i * kTensorSize .. (i+1) * kTensorSize).
// ---------------------------------------------------------------------------
template <typename Traits>
void generate_tensor_batch(const BoardStateT<Traits>& board,
                           const GameContextT<Traits>& ctx,
                           int player,
                           const AfterstateRequest* requests, int request_count,
                           const PrecomputedTables& tables,
                           float* out);

// ---------------------------------------------------------------------------
// Helper functions (also used by tensor generation internally).
//
// These are variant-agnostic in their internal logic — they take a player
// index and read board.cells[player] / ctx.upper_sum[player]. Templated on
// Traits only so the engine/heuristic can call them with either 1v1 or 2v2
// data structures.
// ---------------------------------------------------------------------------

/// Compute raw column score: sum of positive filled cells + upper section bonus.
/// Does NOT include clean column bonus (that depends on duel context).
template <typename Traits>
int compute_column_raw_score(const BoardStateT<Traits>& board,
                             const GameContextT<Traits>& ctx,
                             int player, int column);

/// Compute potential column score: raw score + max possible for empty cells
/// + potential upper bonus (if all empty upper cells were maxed).
template <typename Traits>
int compute_column_potential_score(const BoardStateT<Traits>& board,
                                   const GameContextT<Traits>& ctx,
                                   int player, int column);

/// Sum potential max scores across all columns for a player.
template <typename Traits>
int compute_total_potential(const BoardStateT<Traits>& board, int player);

/// Count empty cells in a column for a player.
template <typename Traits>
int count_empty_cells(const BoardStateT<Traits>& board, int player, int column);

/// Count total filled cells for a player (all columns).
template <typename Traits>
int count_filled_cells(const BoardStateT<Traits>& board, int player);

/// Sum all positive filled cell values for a player (all columns).
template <typename Traits>
int sum_all_filled(const BoardStateT<Traits>& board, int player);
