#include "engine/tensor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#include "engine/duel.h"  // upper_section_bonus

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

int count_empty_cells(const BoardState& board, int player, int column) {
    int n = 0;
    for (int row = 0; row < kNumRows; ++row)
        if (board.cells[player][column][row] == kCellEmpty) ++n;
    return n;
}

int count_filled_cells(const BoardState& board, int player) {
    int n = 0;
    for (int col = 0; col < kNumColumns; ++col)
        for (int row = 0; row < kNumRows; ++row)
            if (board.cells[player][col][row] != kCellEmpty) ++n;
    return n;
}

int sum_all_filled(const BoardState& board, int player) {
    int s = 0;
    for (int col = 0; col < kNumColumns; ++col) {
        int upper_realistic_sum = 0;
        for (int row = 0; row < kNumRows; ++row) {
            int8_t v = board.cells[player][col][row];
            if (v > 0) { 
                s += v;
                if (row <= kRow6s) upper_realistic_sum += v;
            } else if (row <= kRow6s) {
                upper_realistic_sum += 3 * (row + 1);
            }
        }
    }
    return s;
}

int compute_column_raw_score(const BoardState& board, const GameContext& ctx,
                              int player, int column) {
    int cell_sum = 0;
    for (int row = 0; row < kNumRows; ++row) {
        int8_t v = board.cells[player][column][row];
        if (v > 0) cell_sum += v;
    }
    return cell_sum + upper_section_bonus(ctx.upper_sum[player][column]);
}

int compute_column_potential_score(const BoardState& board, const GameContext& ctx,
                                    int player, int column) {
    int cell_sum = 0;
    int upper_potential = ctx.upper_sum[player][column];
    for (int row = 0; row < kNumRows; ++row) {
        int8_t v = board.cells[player][column][row];
        if (v > 0) {
            cell_sum += v;
        } else if (v == kCellEmpty) {
            cell_sum += kMaxScorePerRow[row];
            if (row <= kRow6s)
                upper_potential += kMaxScorePerRow[row];
        }
        // v == 0 (scratch): contributes nothing
    }
    return cell_sum + upper_section_bonus(upper_potential);
}

int compute_total_potential(const BoardState& board, int player) {
    // Build a fake GameContext fragment to call compute_column_potential_score.
    // We only need upper_sum, which we compute on the fly.
    int total = 0;
    for (int col = 0; col < kNumColumns; ++col) {
        int cell_sum = 0;
        int upper_sum = 0;
        int upper_potential = 0;
        for (int row = 0; row < kNumRows; ++row) {
            int8_t v = board.cells[player][col][row];
            if (v > 0) {
                cell_sum += v;
                if (row <= kRow6s) upper_sum += v;
            } else if (v == kCellEmpty) {
                cell_sum += kMaxScorePerRow[row];
                if (row <= kRow6s) upper_potential += kMaxScorePerRow[row];
            }
        }
        total += cell_sum + upper_section_bonus(upper_sum + upper_potential);
    }
    return total;
}

// ---------------------------------------------------------------------------
// generate_tensor — main implementation
// ---------------------------------------------------------------------------

void generate_tensor(const BoardState& board, const GameContext& ctx,
                     int player, const PrecomputedTables& tables,
                     float* out) {
    int idx = 0;

    const int opp = 1 - player;

    // =========================================================================
    // Group A: Per-player × per-column × per-row cell values (3 features/cell)
    // 2 × 6 × 13 × 3 = 468 features
    // =========================================================================
    for (int pi = 0; pi < kNumPlayers; ++pi) {
        int p = (pi == 0) ? player : opp;
        for (int col = 0; col < kNumColumns; ++col) {
            for (int row = 0; row < kNumRows; ++row) {
                int8_t v = board.cells[p][col][row];
                if (v == kCellEmpty) {
                    out[idx++] = 0.0f;  // is_filled
                    out[idx++] = 0.0f;  // score / cell_max
                    out[idx++] = 0.0f;  // score / section_max
                } else {
                    out[idx++] = 1.0f;  // is_filled
                    float score_f = static_cast<float>(v);
                    out[idx++] = score_f / static_cast<float>(kMaxScorePerRow[row]);
                    out[idx++] = (row <= kRow6s) ? (score_f / 30.0f)
                                                 : (score_f / 100.0f);
                }
            }
        }
    }

    // =========================================================================
    // Group B: Per-player × per-column derived features (8 features/column)
    // 2 × 6 × 8 = 96 features
    // =========================================================================
    for (int pi = 0; pi < kNumPlayers; ++pi) {
        int p = (pi == 0) ? player : opp;
        for (int col = 0; col < kNumColumns; ++col) {
            // 1. Upper sum normalised
            int upper_sum = ctx.upper_sum[p][col];
            out[idx++] = std::min(1.0f, static_cast<float>(upper_sum) / 100.0f);

            // 2. Upper potential (max if empty upper cells filled to max)
            int upper_potential = upper_sum;
            int upper_realistic = upper_sum;
            for (int row = 0; row <= kRow6s; ++row)
                if (board.cells[p][col][row] == kCellEmpty) {
                    upper_potential += kMaxScorePerRow[row];
                    upper_realistic += 3 * (row + 1);
                }
            out[idx++] = std::min(1.0f, static_cast<float>(upper_potential) / 100.0f);

            // 3. Clean column eligibility
            bool clean = (upper_realistic >= 60) && (!ctx.lower_has_scratch[p][col]);
            out[idx++] = clean ? 1.0f : 0.0f;

            // 4-5. Column duel advantage / disadvantage
            int my_raw  = compute_column_raw_score(board, ctx, p,     col);
            int opp_raw = compute_column_raw_score(board, ctx, 1 - p, col);
            float diff_f = static_cast<float>(my_raw - opp_raw);
            out[idx++] = std::min(1.0f, std::max(0.0f,  diff_f) / 1000.0f);  // advantage
            out[idx++] = std::min(1.0f, std::max(0.0f, -diff_f) / 1000.0f);  // disadvantage

            // 6-7. Potential duel advantage / disadvantage
            int my_pot  = compute_column_potential_score(board, ctx, p,     col);
            int opp_pot = compute_column_potential_score(board, ctx, 1 - p, col);
            float pot_diff_f = static_cast<float>(my_pot - opp_pot);
            out[idx++] = std::min(1.0f, std::max(0.0f,  pot_diff_f) / 1000.0f);
            out[idx++] = std::min(1.0f, std::max(0.0f, -pot_diff_f) / 1000.0f);

            // 8. Cells remaining in this column
            int remaining = count_empty_cells(board, p, col);
            out[idx++] = static_cast<float>(remaining) / static_cast<float>(kNumRows);
        }
    }

    // =========================================================================
    // Group C: Per-player × per-column × per-row probability features
    // 2 × 6 × 13 = 156 features
    // =========================================================================
    for (int pi = 0; pi < kNumPlayers; ++pi) {
        int p = (pi == 0) ? player : opp;

        // Total non-Turbo placements made = count filled cells for player p
        // (78 cells total, so turns_remaining = 78 - placements_made)
        int placements_made = count_filled_cells(board, p);
        int turns_remaining = 78 - placements_made;

        for (int col = 0; col < kNumColumns; ++col) {
            for (int row = 0; row < kNumRows; ++row) {
                int8_t v = board.cells[p][col][row];

                if (v != kCellEmpty) {
                    // Already filled — probability = 1.0
                    out[idx++] = 1.0f;
                    continue;
                }

                // --- Empty cell: compute probability of non-scratch ---

                // SS/LS forced scratch checks
                if (row == kRowSS && ctx.ls_scratched[p][col]) {
                    out[idx++] = 0.0f;
                    continue;
                }
                if (row == kRowLS && ctx.ss_scratched[p][col]) {
                    out[idx++] = 0.0f;
                    continue;
                }

                // Determine golden minimum
                int golden_min = ctx.golden_max[col][row];
                if (golden_min == 0) golden_min = 1;

                // SS ordering: must be < LS if LS is filled
                if (row == kRowSS) {
                    int8_t ls_val = board.cells[p][col][kRowLS];
                    if (ls_val != kCellEmpty && ls_val > 0) {
                        // SS score must satisfy golden_min <= SS < ls_val
                        if (golden_min >= ls_val) {
                            out[idx++] = 0.0f;
                            continue;
                        }
                    }
                }

                // LS ordering: must be > highest SS recorded by anyone
                if (row == kRowLS) {
                    int max_ss = ctx.golden_max[col][kRowSS];
                    if (max_ss > 0) {
                        // LS score must be > max_ss → threshold at least max_ss+1
                        golden_min = std::max(golden_min, max_ss + 1);
                    }
                }

                // Cap threshold at max for this row
                int max_for_row = kMaxScorePerRow[row];
                if (golden_min > max_for_row) {
                    out[idx++] = 0.0f;
                    continue;
                }

                // Clamp threshold to [0, 100] (table range)
                int thresh = std::min(golden_min, 100);

                // --- REPLACE EXPENSIVE POW WITH LOOKUP ---
                int safe_turns = std::max(0, std::min(78, turns_remaining));
                
                float p_compound = (col == kColTurbo)
                    ? tables.prob_tables.prob_2rolls_compound[row][thresh][safe_turns]
                    : tables.prob_tables.prob_3rolls_compound[row][thresh][safe_turns];

                out[idx++] = p_compound;
            }
        }
    }

    // =========================================================================
    // Group D: Global features (89 features)
    // =========================================================================

    // D1: Column coefficients / 18  (6 features)
    for (int col = 0; col < kNumColumns; ++col)
        out[idx++] = static_cast<float>(board.coefficients[col]) / 18.0f;

    // D2: Game progress (1 feature)
    out[idx++] = static_cast<float>(board.cells_filled) / static_cast<float>(kTotalCells);

    // D3: Golden Rule max per (col, row) / max_per_row (78 features)
    for (int col = 0; col < kNumColumns; ++col) {
        for (int row = 0; row < kNumRows; ++row) {
            int gmax = ctx.golden_max[col][row];
            int denom = kMaxScorePerRow[row];
            out[idx++] = static_cast<float>(gmax) / static_cast<float>(denom);
        }
    }

    // D4-D5: My and opponent total scores (2 features)
    int my_total  = sum_all_filled(board, player);
    int opp_total = sum_all_filled(board, opp);
    out[idx++] = std::min(1.0f, static_cast<float>(my_total)  / 6000.0f);
    out[idx++] = std::min(1.0f, static_cast<float>(opp_total) / 6000.0f);

    // D6-D7: My and opponent potential max totals (2 features)
    int my_pot  = compute_total_potential(board, player);
    int opp_pot = compute_total_potential(board, opp);
    out[idx++] = std::min(1.0f, static_cast<float>(my_pot)  / 6000.0f);
    out[idx++] = std::min(1.0f, static_cast<float>(opp_pot) / 6000.0f);

    assert(idx == kTensorSize);
}

// ---------------------------------------------------------------------------
// generate_tensor_batch
// ---------------------------------------------------------------------------

void generate_tensor_batch(const BoardState& board, const GameContext& ctx,
                            int player,
                            const AfterstateRequest* requests, int request_count,
                            const PrecomputedTables& tables,
                            float* out) {
    for (int i = 0; i < request_count; ++i) {
        // Clone board (trivially copyable struct)
        BoardState board_clone = board;
        // Clone context for patching (full copy, ~1KB)
        GameContext ctx_clone = ctx;

        const AfterstateRequest& req = requests[i];
        int col   = req.placement.column;
        int row   = req.placement.row;
        int score = req.score;
        int p     = player;

        // --- Apply placement to cloned board ---
        board_clone.cells[p][col][row] = static_cast<int8_t>(score);
        board_clone.cells_filled++;

        // --- Patch derived context fields ---

        // Golden rule: update max for this (col, row)
        if (score > ctx_clone.golden_max[col][row])
            ctx_clone.golden_max[col][row] = static_cast<int8_t>(score);

        // Upper section sum
        if (row <= kRow6s)
            ctx_clone.upper_sum[p][col] += static_cast<int16_t>(score);

        // SS/LS scratch tracking and mutual destruction
        if (row == kRowSS && score == 0) {
            ctx_clone.ss_scratched[p][col] = true;
            // Mutual destruction: if LS is already filled non-zero, scratch it
            int8_t ls_val = board.cells[p][col][kRowLS];  // from base board
            if (ls_val != kCellEmpty && ls_val != 0) {
                board_clone.cells[p][col][kRowLS] = 0;
                ctx_clone.ls_scratched[p][col]   = true;
                ctx_clone.lower_has_scratch[p][col] = true;
                // Recalculate golden_max for LS: max of both players' LS values
                int8_t opp_ls = board.cells[1 - p][col][kRowLS];
                int8_t new_max = (opp_ls > 0) ? opp_ls : 0;
                ctx_clone.golden_max[col][kRowLS] = new_max;
            }
        } else if (row == kRowLS && score == 0) {
            ctx_clone.ls_scratched[p][col] = true;
            // Mutual destruction: if SS is already filled non-zero, scratch it
            int8_t ss_val = board.cells[p][col][kRowSS];
            if (ss_val != kCellEmpty && ss_val != 0) {
                board_clone.cells[p][col][kRowSS] = 0;
                ctx_clone.ss_scratched[p][col]   = true;
                ctx_clone.lower_has_scratch[p][col] = true;
                // Recalculate golden_max for SS
                int8_t opp_ss = board.cells[1 - p][col][kRowSS];
                int8_t new_max = (opp_ss > 0) ? opp_ss : 0;
                ctx_clone.golden_max[col][kRowSS] = new_max;
            }
        }

        // Lower section scratch flag
        if (row >= kRowSS && score == 0)
            ctx_clone.lower_has_scratch[p][col] = true;

        // Generate tensor for this afterstate
        generate_tensor(board_clone, ctx_clone, player, tables,
                        out + static_cast<ptrdiff_t>(i) * kTensorSize);
    }
}
