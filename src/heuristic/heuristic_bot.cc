#include "heuristic/heuristic_bot.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "engine/duel.h"
#include "engine/game_flow.h"
#include "engine/placement.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "solver/dp_eval.h"

// ---------------------------------------------------------------------------
// V1 — score × column coefficient.
// ---------------------------------------------------------------------------
void heuristic_evaluate(const BoardState& board, const GameContext& /*ctx*/,
                        const AfterstateRequest* requests, int request_count,
                        double* evs) {
    for (int i = 0; i < request_count; ++i) {
        int col = requests[i].placement.column;
        int row = requests[i].placement.row;
        int score = requests[i].score;
        if (row > 0 && row < kRow6s) {
            if (score >= (row + 1) * 3) score += 6;
        }
        if (score == 0) {
            score = (18 - board.coefficients[col]) * 0.1;
            if (row <= kRow6s) score *= 2;
        }
        evs[i] = static_cast<double>(score) * board.coefficients[col];
    }
}

// ---------------------------------------------------------------------------
// V2 — DP-driven expected global duel margin.
// ---------------------------------------------------------------------------
void heuristic_evaluate_v2(const BoardState& base_board, const GameContext& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables) {
    const int p_me  = base_board.current_player;
    const int p_opp = 1 - p_me;
    const DPTables& dp = tables.dp_tables;

    for (int i = 0; i < request_count; ++i) {
        // Apply candidate placement to a clone of the board+ctx.
        BoardState b = base_board;
        GameContext c = base_ctx;
        apply_placement(p_me,
                        requests[i].placement.column,
                        requests[i].placement.row,
                        requests[i].score, b, c);

        const int filled_me  = count_filled_cells(b, p_me);
        const int filled_opp = count_filled_cells(b, p_opp);

        double expected_global_margin = 0.0;

        for (int col = 0; col < kNumColumns; ++col) {
            const int empty_me  = count_empty_cells(b, p_me,  col);
            const int empty_opp = count_empty_cells(b, p_opp, col);

            // Approximate turns remaining for this column. The minimum
            // guarantees enough turns to fill the column; the per-section
            // amortization applies inside get_E_raw via apportion_turns().
            const int T_me  = std::max(empty_me,
                                       empty_me  + (78 - filled_me)  / 6);
            const int T_opp = std::max(empty_opp,
                                       empty_opp + (78 - filled_opp) / 6);

            const float E_me  = get_E_raw(p_me,  col, T_me,  b, c, dp);
            const float E_opp = get_E_raw(p_opp, col, T_opp, b, c, dp);

            const int e_me_i  = static_cast<int>(std::lround(E_me));
            const int e_opp_i = static_cast<int>(std::lround(E_opp));
            const int crush_me  = crush_multiplier(e_me_i,  e_opp_i);
            const int crush_opp = crush_multiplier(e_opp_i, e_me_i);
            const int active_crush = std::max(crush_me, crush_opp);

            // Clean column bonus expected value.
            const float P_clean_me  = get_P_clean(p_me,  col, T_me,  b, c, dp);
            const float P_clean_opp = get_P_clean(p_opp, col, T_opp, b, c, dp);
            const float bonus_val = (active_crush > 1) ? 100.0f : 200.0f;
            const float E_clean_me  = P_clean_me  * bonus_val;
            const float E_clean_opp = P_clean_opp * bonus_val;

            const double margin = ((static_cast<double>(E_me)  + E_clean_me) -
                                   (static_cast<double>(E_opp) + E_clean_opp))
                                * active_crush
                                * b.coefficients[col];
            expected_global_margin += margin;
        }

        evs[i] = expected_global_margin;
    }
}

// ---------------------------------------------------------------------------
// V3 — V2 baseline plus four hard-coded strategic rules.
//
// Rule 1: For each column, penalise an expected raw column score below 150,
//         unless the opponent has scratched so much that opp_max_potential
//         <= 2 * opp_current (the column is effectively decided in our favour
//         and the 150 aim is dropped).
// Rule 2: On Up/Down columns, penalise being behind on E_me. When the column
//         coefficient is 14/16/18, also reward placing high upper-section
//         scores there to raise golden_max and force opponent scratches.
// Rule 3: On high-coefficient (14/16/18) columns, amplify the value of the
//         clean-column bonus expectation.
// Rule 4: For any column, give a quadratic bonus on clean-bonus probability
//         (rewards finishing near-clean columns regardless of coefficient).
// ---------------------------------------------------------------------------
// Mutable V3 weights — defaults chosen so V2's signal still dominates.
//
// Tuning notes (head-to-head V3 vs V2 over N=1000+ games):
//   * V2 is already a strong evaluator: it weights margin, crush multipliers,
//     and clean-bonus expectations by coefficient. Additive rule terms only
//     improve play modestly because they partially overlap V2's signal.
//   * R3 (high-coeff clean amplification) is the only rule that consistently
//     produced a positive lift in isolation (~52% wins, +500..900 margin).
//   * R1/R2/R4 contribute marginal/neutral signal alone but are kept active
//     at small weights to honour the user's strategic spec.
//   * Larger weights either fail to move the win rate (signal already
//     captured by V2) or actively regress it (rules pull V3 away from V2's
//     locally-optimal choices).
//   * Realistic ceiling for additive rules on top of V2: ~52-53% wins.
//     Higher win rates require structural changes (deeper lookahead,
//     exact-enumeration endgame, variance-aware EV).
static V3Weights g_v3_weights = {
    /*w_r1_per_point=*/  0.005,
    /*w_r2_block_high=*/ 0.10,
    /*w_r3_high_coeff=*/ 0.10,
    /*w_r4_near_clean=*/ 0.20,
    /*r4_threshold=*/    0.50,
};

const V3Weights& v3_weights() { return g_v3_weights; }
void set_v3_weights(const V3Weights& w) { g_v3_weights = w; }

void heuristic_evaluate_v3(const BoardState& base_board, const GameContext& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables) {
    const V3Weights w = g_v3_weights;
    constexpr double kCleanValue = 200.0;  // Clean-bonus value before crush halving
    constexpr double kColumnAim  = 150.0;  // Per-column raw-score aim

    const int p_me  = base_board.current_player;
    const int p_opp = 1 - p_me;
    const DPTables& dp = tables.dp_tables;

    // Start from V2's baseline.
    heuristic_evaluate_v2(base_board, base_ctx, requests, request_count, evs, tables);

    for (int i = 0; i < request_count; ++i) {
        BoardState b = base_board;
        GameContext c = base_ctx;
        apply_placement(p_me,
                        requests[i].placement.column,
                        requests[i].placement.row,
                        requests[i].score, b, c);

        const int filled_me  = count_filled_cells(b, p_me);
        const int filled_opp = count_filled_cells(b, p_opp);

        double rule_terms = 0.0;

        for (int col = 0; col < kNumColumns; ++col) {
            const int coeff   = b.coefficients[col];
            const bool is_high_coeff = (coeff == 14 || coeff == 16 || coeff == 18);
            const bool is_up_or_down = (col == kColDown || col == kColUp);

            const int empty_me  = count_empty_cells(b, p_me,  col);
            const int empty_opp = count_empty_cells(b, p_opp, col);
            if (empty_me == 0 && empty_opp == 0) continue;

            const int T_me  = std::max(empty_me,
                                       empty_me  + (78 - filled_me)  / 6);

            const float E_me = (empty_me > 0) ? get_E_raw(p_me, col, T_me, b, c, dp) : 0.0f;

            // -----------------------------------------------------------
            // Rule 1: aim for at least 150 raw points per column.
            // Skipped when (a) the column is already mathematically out of
            // reach for me, or (b) opp is so scratched that opp_max can't
            // exceed 2 * opp_current (the column is decided in our favour).
            // -----------------------------------------------------------
            if (empty_me > 0) {
                const int my_max = compute_column_potential_score(b, c, p_me, col);
                if (my_max >= static_cast<int>(kColumnAim)) {
                    const int opp_current = compute_column_raw_score(b, c, p_opp, col);
                    const int opp_max     = compute_column_potential_score(b, c, p_opp, col);
                    const bool aim_dropped = (opp_max <= 2 * opp_current);
                    if (!aim_dropped) {
                        const double deficit = kColumnAim - static_cast<double>(E_me);
                        if (deficit > 0.0) {
                            rule_terms -= deficit * coeff * w.w_r1_per_point;
                        }
                    }
                }
            }

            // -----------------------------------------------------------
            // Rule 2: block the opponent on high-coeff Up/Down columns by
            // placing high upper-section scores that raise the golden_max.
            // V2 already captures the margin signal, so V3 only adds the
            // extra blocking incentive (no symmetric "behind" penalty).
            // -----------------------------------------------------------
            if (is_up_or_down && is_high_coeff &&
                requests[i].placement.column == col &&
                requests[i].placement.row <= kRow6s &&
                requests[i].score > 0) {
                rule_terms += static_cast<double>(requests[i].score)
                            * coeff * w.w_r2_block_high;
            }

            // -----------------------------------------------------------
            // Rules 3 & 4: clean-bonus emphasis.
            //   R3 — small extra weight on high-coeff cleans only.
            //   R4 — linear boost only above kR4_Threshold so it focuses on
            //        columns that are truly near-clean (e.g. yams already
            //        set, low remaining lower-section risk).
            // -----------------------------------------------------------
            if (empty_me > 0) {
                const float P_clean = get_P_clean(p_me, col, T_me, b, c, dp);
                // Rule 3: extra weight on cleans for high-coeff columns.
                if (is_high_coeff) {
                    rule_terms += static_cast<double>(P_clean) * kCleanValue
                                * coeff * w.w_r3_high_coeff;
                }
                // Rule 4: focus on near-clean columns (threshold-gated to
                // avoid double-rewarding low-prob cleans V2 already covers).
                if (P_clean > w.r4_threshold) {
                    rule_terms += (static_cast<double>(P_clean) - w.r4_threshold)
                                * kCleanValue * coeff * w.w_r4_near_clean;
                }
            }
        }

        evs[i] += rule_terms;
    }
}

// ---------------------------------------------------------------------------
// Research evaluator — configurable V2-style with structural knobs.
// ---------------------------------------------------------------------------
namespace {

inline int compute_T(TStrategy strat, int empty_in_col,
                     int filled_player, int total_empty_player) {
    switch (strat) {
        case TStrategy::V2_DEFAULT:
            return std::max(empty_in_col,
                            empty_in_col + (78 - filled_player) / 6);
        case TStrategy::EMPTY_ONLY:
            return empty_in_col;
        case TStrategy::EMPTY_PLUS_1:
            return empty_in_col + 1;
        case TStrategy::EMPTY_PLUS_2:
            return empty_in_col + 2;
        case TStrategy::EMPTY_DOUBLE:
            return empty_in_col * 2;
        case TStrategy::PROPORTIONAL: {
            if (total_empty_player <= 0 || empty_in_col == 0) return empty_in_col;
            int prop = ((78 - filled_player) * empty_in_col) / total_empty_player;
            return std::max(empty_in_col, prop);
        }
    }
    return empty_in_col;
}

inline float crush_smooth(float my, float opp) {
    if (opp <= 0.0f) return (my > 0.0f) ? 5.0f : 1.0f;
    float ratio = my / opp;
    if (ratio >= 5.0f) return 5.0f;
    if (ratio <= 1.0f) return 1.0f;
    // Linear interpolation between 1×, 2×, 3×, 4×, 5× thresholds.
    return 1.0f + (ratio - 1.0f);  // ratio∈[1,5] → result∈[1,5]
}

inline float compute_crush(CrushMode mode, float E_me, float E_opp) {
    if (mode == CrushMode::FLOAT_SMOOTH) {
        float a = crush_smooth(E_me, E_opp);
        float b = crush_smooth(E_opp, E_me);
        return std::max(a, b);
    }
    int e_me_i, e_opp_i;
    if (mode == CrushMode::FLOOR_INT) {
        e_me_i  = static_cast<int>(std::floor(E_me));
        e_opp_i = static_cast<int>(std::floor(E_opp));
    } else { // ROUND_INT
        e_me_i  = static_cast<int>(std::lround(E_me));
        e_opp_i = static_cast<int>(std::lround(E_opp));
    }
    int a = crush_multiplier(e_me_i,  e_opp_i);
    int b = crush_multiplier(e_opp_i, e_me_i);
    return static_cast<float>(std::max(a, b));
}

}  // namespace

void heuristic_evaluate_research(const BoardState& base_board, const GameContext& base_ctx,
                                 const AfterstateRequest* requests, int request_count,
                                 double* evs, const PrecomputedTables& tables,
                                 const ResearchConfig& cfg) {
    const int p_me  = base_board.current_player;
    const int p_opp = 1 - p_me;
    const DPTables& dp = tables.dp_tables;

    for (int i = 0; i < request_count; ++i) {
        BoardState b = base_board;
        GameContext c = base_ctx;
        apply_placement(p_me,
                        requests[i].placement.column,
                        requests[i].placement.row,
                        requests[i].score, b, c);

        const int filled_me  = count_filled_cells(b, p_me);
        const int filled_opp = count_filled_cells(b, p_opp);
        const int total_empty_me  = 78 - filled_me;
        const int total_empty_opp = 78 - filled_opp;

        double margin = 0.0;

        for (int col = 0; col < kNumColumns; ++col) {
            const int empty_me  = count_empty_cells(b, p_me,  col);
            const int empty_opp = count_empty_cells(b, p_opp, col);
            const int coeff = b.coefficients[col];

            const int T_me  = compute_T(cfg.t_me,  empty_me,
                                        filled_me,  total_empty_me);
            const int T_opp = compute_T(cfg.t_opp, empty_opp,
                                        filled_opp, total_empty_opp);

            const float E_me  = get_E_raw(p_me,  col, T_me,  b, c, dp);
            const float E_opp = get_E_raw(p_opp, col, T_opp, b, c, dp);

            const float active_crush = compute_crush(cfg.crush, E_me, E_opp);

            // Clean column bonus expected value.
            const float P_clean_me  = get_P_clean(p_me,  col, T_me,  b, c, dp);
            const float P_clean_opp = get_P_clean(p_opp, col, T_opp, b, c, dp);
            const float bonus_val = (active_crush > 1.0f) ? 100.0f : 200.0f;
            const float pow_me  = (cfg.clean_power == 1.0)
                                  ? P_clean_me
                                  : std::pow(P_clean_me,  static_cast<float>(cfg.clean_power));
            const float pow_opp = (cfg.clean_power == 1.0)
                                  ? P_clean_opp
                                  : std::pow(P_clean_opp, static_cast<float>(cfg.clean_power));
            const float E_clean_me  = pow_me  * bonus_val;
            const float E_clean_opp = pow_opp * bonus_val;

            double col_term = ((static_cast<double>(E_me)  + E_clean_me) -
                               (static_cast<double>(E_opp) + E_clean_opp))
                            * active_crush
                            * coeff;

            // Opponent-aware push: when behind on E in a column, amplify.
            if (cfg.opp_aware_factor != 0.0 && E_opp > E_me) {
                col_term *= (1.0 + cfg.opp_aware_factor);
            }

            margin += col_term;
        }

        // Voluntary-scratch penalty: when the candidate is a scratch (score=0)
        // in a row that hasn't been claimed yet (golden_max=0).
        if (cfg.scratch_penalty != 0.0 && requests[i].score == 0) {
            const int col = requests[i].placement.column;
            const int row = requests[i].placement.row;
            const int gmax = base_ctx.golden_max[col][row];
            if (gmax == 0) {
                int row_max;
                if      (row <  kRowSS) row_max = (row + 1) * 5;   // upper rows: 5,10,15,20,25,30
                else if (row == kRowSS) row_max = 29;
                else if (row == kRowLS) row_max = 30;
                else if (row == kRowFH) row_max = 50;
                else if (row == kRowK)  row_max = 54;
                else if (row == kRowSTR)row_max = 50;
                else if (row == kRowU8) row_max = 75;
                else                     row_max = 100;            // Yams
                margin -= cfg.scratch_penalty * row_max * base_board.coefficients[col];
            }
        }

        // Variance penalty for high-variance rows when filled positively.
        if (cfg.variance_penalty != 0.0 && requests[i].score > 0) {
            const int row = requests[i].placement.row;
            const int col = requests[i].placement.column;
            if (row == kRowY || row == kRowU8) {
                margin -= cfg.variance_penalty * base_board.coefficients[col];
            }
        }

        // Early high-coeff bonus: reward placements in high-coeff columns
        // when many cells still remain (push to start them sooner).
        if (cfg.early_high_coeff_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 14) {
                const int empty_in_col = count_empty_cells(base_board, p_me, col);
                margin += cfg.early_high_coeff_bonus * empty_in_col * coeff;
            }
        }

        // Progressive variant — fades to zero as the game fills up.
        if (cfg.early_progressive_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 14) {
                const int empty_in_col = count_empty_cells(base_board, p_me, col);
                const double phase = std::max(0, 78 - filled_me) / 78.0;
                margin += cfg.early_progressive_bonus * empty_in_col * coeff * phase;
            }
        }

        // Coefficient² bonus — amplifies the very highest-coeff columns.
        if (cfg.coeff_sq_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            const int empty_in_col = count_empty_cells(base_board, p_me, col);
            margin += cfg.coeff_sq_bonus * empty_in_col * coeff * coeff;
        }

        // Completion bonus — prefer columns with more cells already filled.
        if (cfg.completion_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            const int filled_in_col = 13 - count_empty_cells(base_board, p_me, col);
            margin += cfg.completion_bonus * filled_in_col * coeff;
        }

        // Dominance bonus — bonus for placements in columns where we have
        // strictly more positive cells than the opponent.
        if (cfg.dominance_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            int my_pos = 0, opp_pos = 0;
            for (int r = 0; r < kNumRows; ++r) {
                if (base_board.cells[p_me][col][r]  > 0) ++my_pos;
                if (base_board.cells[p_opp][col][r] > 0) ++opp_pos;
            }
            if (my_pos > opp_pos) {
                margin += cfg.dominance_bonus * (my_pos - opp_pos) * coeff;
            }
        }

        // Upper-100 bonus — push the upper section toward the giant 500-bonus
        // threshold by rewarding upper-section placements proportional to the
        // current upper_sum.
        if (cfg.upper100_bonus != 0.0 && requests[i].placement.row <= kRow6s
            && requests[i].score > 0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            const int upper_now = base_ctx.upper_sum[p_me][col];
            // Weight grows as upper_sum approaches 100 (peak around 80-95).
            const double t = std::min(1.0, std::max(0.0, upper_now / 100.0));
            margin += cfg.upper100_bonus * requests[i].score * coeff * t;
        }

        // High-coeff scratch penalty.
        if (cfg.hc_scratch_penalty != 0.0 && requests[i].score == 0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 14) {
                margin -= cfg.hc_scratch_penalty * coeff;
            }
        }

        // Upper priority bonus — reward filling 4s/5s/6s rows specifically
        // (the rows that drive the big 100-upper bonus).
        if (cfg.upper_priority_bonus != 0.0 && requests[i].score > 0) {
            const int row = requests[i].placement.row;
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (row >= kRow4s && row <= kRow6s) {
                margin += cfg.upper_priority_bonus * requests[i].score * coeff;
            }
        }

        // Coeff² bonus restricted to high-coefficient columns only.
        if (cfg.coeff_sq_high_only != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 14) {
                const int empty_in_col = count_empty_cells(base_board, p_me, col);
                margin += cfg.coeff_sq_high_only * empty_in_col * coeff * coeff;
            }
        }

        // Turbo avoidance — small penalty for Turbo placements, fades with
        // game progress (encourages saving Turbo for the endgame).
        if (cfg.turbo_avoidance != 0.0
            && requests[i].placement.column == kColTurbo) {
            const double phase = std::max(0, 78 - filled_me) / 78.0;
            margin -= cfg.turbo_avoidance * phase * 100.0;
        }

        // Last-cell bonus — reward placements that fill the last empty cell
        // in a column (column completion locks in upper-bonus realisations).
        if (cfg.last_cell_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (count_empty_cells(base_board, p_me, col) == 1) {
                margin += cfg.last_cell_bonus * coeff;
            }
        }

        // Upper-bonus protection penalty — heavy penalty for upper-section
        // placements that hurt the chances of reaching the 60-point upper
        // bonus on columns with coefficient >= 12, excluding the Down column.
        // The penalty is proportional to the deficit below par (3 × face).
        if (cfg.upper_bonus_penalty != 0.0
            && requests[i].placement.row <= kRow6s && requests[i].placement.row >= kRow3s
            && requests[i].placement.column != kColDown) {
            const int col   = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 12) {
                const int row   = requests[i].placement.row;
                const int face  = row + 1;              // 1s..6s
                const int par   = face * 3;             // target per cell for 60+ total
                const int score = requests[i].score;
                if (score < par) {
                    margin -= cfg.upper_bonus_penalty * coeff;
                }
            }
        }

        evs[i] = margin;

        // Optional V3 rule layer (additive).
        if (cfg.use_v3_rules) {
            // Apply only this candidate's rule terms. Easiest: temporarily
            // swap g_v3_weights, evaluate just this request, restore. Avoid
            // nested heuristic_evaluate_v3 calls — just inline minimal R3 (the
            // proven-positive rule) here.
            const float P_clean_me_local = get_P_clean(p_me,
                requests[i].placement.column,
                compute_T(cfg.t_me,
                          count_empty_cells(b, p_me, requests[i].placement.column),
                          filled_me, total_empty_me),
                b, c, dp);
            const int coeff = b.coefficients[requests[i].placement.column];
            const bool is_high = (coeff >= 14);
            if (is_high && cfg.v3.w_r3_high_coeff > 0.0) {
                evs[i] += static_cast<double>(P_clean_me_local) * 200.0
                        * coeff * cfg.v3.w_r3_high_coeff;
            }
        }
    }
}

void heuristic_play_turn_research(GameState& state, GameContext& ctx,
                                  const PrecomputedTables& tables,
                                  SolverBuffers& buffers, RNG& rng,
                                  const ResearchConfig& cfg) {
    buffers.dp_computed = false;
    solver_get_requests(state, ctx, tables, buffers);
    assert(buffers.request_count > 0);
    heuristic_evaluate_research(state.board, ctx,
                                buffers.requests, buffers.request_count,
                                buffers.evs, tables, cfg);
    while (true) {
        SolverResult result = solver_resolve_greedy(state, ctx, tables, buffers);
        if (result.should_place) {
            perform_placement(state, ctx, result.placement.column,
                              result.placement.row, rng);
            return;
        }
        if (!can_reroll(state, ctx)) {
            int current_id = get_dice_state_id(state.dice, tables);
            int16_t req_idx = buffers.stop_request_idx[current_id];
            if (req_idx < 0) req_idx = 0;
            perform_placement(state, ctx, buffers.requests[req_idx].placement.column,
                              buffers.requests[req_idx].placement.row, rng);
            return;
        }
        perform_reroll(state, result.hold_mask, rng);
    }
}

// ---------------------------------------------------------------------------
// Research-derived numbered versions V4..V15.
// Each maps to a fixed ResearchConfig discovered by the heuristic_sweep
// benchmark (head-to-head vs V2 over N≥1500 paired games).
// ---------------------------------------------------------------------------
const ResearchConfig& get_research_config_for(HeuristicVersion v) {
    static ResearchConfig cache[16]; // V4..V15
    static bool initialized[16] = {false};

    int idx = static_cast<int>(v);
    if (idx < 4 || idx > 15) {
        throw std::invalid_argument("get_research_config_for: V1/V2/V3 use their own evaluators");
    }

    if (initialized[idx]) {
        return cache[idx];
    }

    ResearchConfig c;
    switch (v) {
        case HeuristicVersion::V4:
            // smooth + early-high-coeff(1.0)              — ~56% vs V2
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 1.0;
            break;
        case HeuristicVersion::V5:
            // smooth + early-high-coeff(1.5)              — ~55%
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 1.5;
            break;
        case HeuristicVersion::V6:
            // smooth + coeff²(0.20) standalone            — ~57%
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.20;
            break;
        case HeuristicVersion::V7:
            // smooth + coeff²(0.10) + turbo-avoid         — ~57.4%
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.10;
            c.turbo_avoidance = 1.0;
            break;
        case HeuristicVersion::V8:
            // smooth + coeff²(0.20) + turbo-avoid         — ~57.1%
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.20;
            c.turbo_avoidance = 1.0;
            break;
        case HeuristicVersion::V9:
            // smooth + coeff²(0.50) + turbo-avoid         — ~57%
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.50;
            c.turbo_avoidance = 1.0;
            break;
        case HeuristicVersion::V10:
            // smooth + ehc(0.5) + csq(0.12)               — ~56.5%
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 0.5;
            c.coeff_sq_bonus = 0.12;
            break;
        case HeuristicVersion::V11:
            // smooth + ehc(1.0) + csq(0.05)               — ~56.7%
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 1.0;
            c.coeff_sq_bonus = 0.05;
            break;
        case HeuristicVersion::V12:
            // smooth + ehc(0.5) + csq(0.20)               — ~55.5%
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 0.5;
            c.coeff_sq_bonus = 0.20;
            break;
        case HeuristicVersion::V13:
            // smooth + coeff²(0.15) + turbo + R3(0.10)    — ~57.6% (BEST)
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.15;
            c.turbo_avoidance = 1.0;
            c.use_v3_rules = true;
            c.v3.w_r3_high_coeff = 0.10;
            break;
        case HeuristicVersion::V14:
            // V13 + upper-bonus protection on coeff>=12 non-down columns
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.15;
            c.turbo_avoidance = 1.0;
            c.use_v3_rules = true;
            c.v3.w_r3_high_coeff = 0.10;
            c.upper_bonus_penalty = 20.0;
            break;
        case HeuristicVersion::V15:
            // V13 + aggressive upper-bonus protection (200.0)
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.15;
            c.turbo_avoidance = 1.0;
            c.use_v3_rules = true;
            c.v3.w_r3_high_coeff = 0.10;
            c.upper_bonus_penalty = 200.0;
            break;
        default:
            throw std::invalid_argument("get_research_config_for: unknown version");
    }
    cache[idx] = c;
    initialized[idx] = true;
    return cache[idx];
}

// ---------------------------------------------------------------------------
// Play one complete turn.
// ---------------------------------------------------------------------------
void heuristic_play_turn(GameState& state, GameContext& ctx,
                         const PrecomputedTables& tables,
                         SolverBuffers& buffers, RNG& rng,
                         HeuristicVersion version) {
    buffers.dp_computed = false;

    // Step 1: get afterstate requests (static for the turn).
    solver_get_requests(state, ctx, tables, buffers);
    assert(buffers.request_count > 0);

    // Step 2: evaluate with heuristic (static for the turn).
    if (static_cast<int>(version) >= static_cast<int>(HeuristicVersion::V4)) {
        const ResearchConfig& cfg = get_research_config_for(version);
        heuristic_evaluate_research(state.board, ctx,
                                    buffers.requests, buffers.request_count,
                                    buffers.evs, tables, cfg);
    } else if (version == HeuristicVersion::V3) {
        heuristic_evaluate_v3(state.board, ctx,
                              buffers.requests, buffers.request_count,
                              buffers.evs, tables);
    } else if (version == HeuristicVersion::V2) {
        heuristic_evaluate_v2(state.board, ctx,
                              buffers.requests, buffers.request_count,
                              buffers.evs, tables);
    } else {
        heuristic_evaluate(state.board, ctx,
                           buffers.requests, buffers.request_count, buffers.evs);
    }

    while (true) {
        // Step 3: resolve best action (uses cached DP tables on rerolls).
        SolverResult result = solver_resolve_greedy(state, ctx, tables, buffers);

        if (result.should_place) {
            perform_placement(state, ctx, result.placement.column,
                              result.placement.row, rng);
            return;
        }

        // Hold and reroll.
        if (!can_reroll(state, ctx)) {
            // Safety break to prevent infinite loop.
            int current_id = get_dice_state_id(state.dice, tables);
            int16_t req_idx = buffers.stop_request_idx[current_id];
            if (req_idx < 0) req_idx = 0;
            perform_placement(state, ctx, buffers.requests[req_idx].placement.column,
                              buffers.requests[req_idx].placement.row, rng);
            return;
        }
        perform_reroll(state, result.hold_mask, rng);
    }
}

// ---------------------------------------------------------------------------
// Play a complete game.
// ---------------------------------------------------------------------------
int play_heuristic_game(RNG& rng, const PrecomputedTables& tables,
                        HeuristicVersion version) {
    GameState state;
    GameContext ctx;
    SolverBuffers buffers;

    init_game(state, ctx, rng);

    while (!is_terminal(state.board)) {
        heuristic_play_turn(state, ctx, tables, buffers, rng, version);
    }

    return get_game_result(state, ctx);
}

// ---------------------------------------------------------------------------
// Pick a random HeuristicVersion uniformly from all available versions.
// ---------------------------------------------------------------------------
HeuristicVersion random_heuristic_version(RNG& rng) {
    static constexpr HeuristicVersion kAll[] = {
        HeuristicVersion::V1,  HeuristicVersion::V2,  HeuristicVersion::V3,
        HeuristicVersion::V4,  HeuristicVersion::V5,  HeuristicVersion::V6,
        HeuristicVersion::V7,  HeuristicVersion::V8,  HeuristicVersion::V9,
        HeuristicVersion::V10, HeuristicVersion::V11, HeuristicVersion::V12,
        HeuristicVersion::V13, HeuristicVersion::V14, HeuristicVersion::V15,
    };
    constexpr int kCount = sizeof(kAll) / sizeof(kAll[0]);
    return kAll[rng.next() % kCount];
}
