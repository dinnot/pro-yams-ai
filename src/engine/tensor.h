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

/// Total number of float features in the state observation tensor.
constexpr int kTensorSize = 986;

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
// @param board   Board state to encode
// @param ctx     Game context (cached derived data)
// @param player  Perspective player (0 or 1); their data appears first
// @param tables  Precomputed tables (score tables, probability tables)
// @param out     Output buffer of exactly kTensorSize floats
// ---------------------------------------------------------------------------
void generate_tensor(const BoardState& board, const GameContext& ctx,
                     int player, const PrecomputedTables& tables,
                     float* out);

// ---------------------------------------------------------------------------
// Batch tensor generation for afterstates.
//
// For each request, applies the placement as a simple cell-write to a cloned
// board (without full GameContext cache maintenance) and generates a tensor.
// Writes tensors contiguously: out[i * kTensorSize .. (i+1) * kTensorSize).
//
// @param board          Base board state (before any placement)
// @param ctx            Game context for the base state
// @param player         Perspective player (0 or 1)
// @param requests       Array of (placement, score) afterstate descriptors
// @param request_count  Number of afterstates
// @param tables         Precomputed tables
// @param out            Output buffer: request_count × kTensorSize floats
// ---------------------------------------------------------------------------
void generate_tensor_batch(const BoardState& board, const GameContext& ctx,
                            int player,
                            const AfterstateRequest* requests, int request_count,
                            const PrecomputedTables& tables,
                            float* out);

// ---------------------------------------------------------------------------
// Helper functions (also used by tensor generation internally)
// ---------------------------------------------------------------------------

/// Compute raw column score: sum of positive filled cells + upper section bonus.
/// Does NOT include clean column bonus (that depends on duel context).
int compute_column_raw_score(const BoardState& board, const GameContext& ctx,
                              int player, int column);

/// Compute potential column score: raw score + max possible for empty cells
/// + potential upper bonus (if all empty upper cells were maxed).
int compute_column_potential_score(const BoardState& board, const GameContext& ctx,
                                    int player, int column);

/// Sum potential max scores across all columns for a player.
int compute_total_potential(const BoardState& board, int player);

/// Count empty cells in a column for a player.
int count_empty_cells(const BoardState& board, int player, int column);

/// Count total filled cells for a player (all columns).
int count_filled_cells(const BoardState& board, int player);

/// Sum all positive filled cell values for a player (all columns).
int sum_all_filled(const BoardState& board, int player);
