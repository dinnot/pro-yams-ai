#include "heuristic/heuristic_bot.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "engine/duel.h"
#include "engine/game_flow.h"
#include "engine/game_traits.h"
#include "engine/placement.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "solver/dp_eval.h"

inline double fast_erf(double x) {
    if (x == 0.0) return 0.0;
    double sign = (x < 0) ? -1.0 : 1.0;
    x = std::abs(x);
    constexpr double a1 =  0.254829592, a2 = -0.284496736, a3 =  1.421413741;
    constexpr double a4 = -1.453152027, a5 =  1.061405429, p  =  0.3275911;
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);
    return sign * y;
}

// ---------------------------------------------------------------------------
// V1 — score × column coefficient.
// ---------------------------------------------------------------------------
template <typename Traits>
void heuristic_evaluate(const BoardStateT<Traits>& board,
                        const GameContextT<Traits>& /*ctx*/,
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
// Internal helpers — duel/crush math shared by V2, V3, research.
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
    return 1.0f + (ratio - 1.0f);
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

// ---------------------------------------------------------------------------
// V2 — DP-driven expected team duel margin.
//
// 1v1: one pairing (P0, P1) per column. Bit-for-bit identical to the previous
// 1v1 implementation.
//
// 2v2: four cross-team pairings per column. Per-pairing crush + clean-bonus
// value (matching compute_duel<Yams2v2>). Return value is the active player's
// TEAM margin (positive = my team ahead).
// ---------------------------------------------------------------------------
template <typename Traits>
void heuristic_evaluate_v2(const BoardStateT<Traits>& base_board,
                           const GameContextT<Traits>& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables) {
    const int p_me = base_board.current_player;
    const DPTables& dp = tables.dp_tables;
    const int team_sign = Traits::are_teammates(p_me, 0) ? +1 : -1;

    for (int i = 0; i < request_count; ++i) {
        BoardStateT<Traits> b = base_board;
        GameContextT<Traits> c = base_ctx;
        apply_placement<Traits>(p_me,
                                requests[i].placement.column,
                                requests[i].placement.row,
                                requests[i].score, b, c);

        // Per-player turn counts (V2 default strategy).
        int filled[Traits::kNumPlayers];
        for (int p = 0; p < Traits::kNumPlayers; ++p)
            filled[p] = count_filled_cells<Traits>(b, p);

        double team0_margin = 0.0;

        for (int col = 0; col < kNumColumns; ++col) {
            // Per-player E_raw and P_clean for this column.
            float E[Traits::kNumPlayers];
            float P_clean[Traits::kNumPlayers];
            for (int p = 0; p < Traits::kNumPlayers; ++p) {
                const int empty = count_empty_cells<Traits>(b, p, col);
                const int T = std::max(empty, empty + (78 - filled[p]) / 6);
                E[p]       = get_E_raw<Traits>(p, col, T, b, c, dp);
                P_clean[p] = get_P_clean<Traits>(p, col, T, b, c, dp);
            }

            // Cross-team pairings (1v1: 1×1, 2v2: 2×2).
            for (int ti = 0; ti < Traits::kPlayersPerTeam; ++ti) {
                for (int tj = 0; tj < Traits::kPlayersPerTeam; ++tj) {
                    const int t0p = Traits::kTeam0[ti];
                    const int t1p = Traits::kTeam1[tj];

                    const int e_t0_i = static_cast<int>(std::lround(E[t0p]));
                    const int e_t1_i = static_cast<int>(std::lround(E[t1p]));
                    const int crush_t0 = crush_multiplier(e_t0_i, e_t1_i);
                    const int crush_t1 = crush_multiplier(e_t1_i, e_t0_i);
                    const int active_crush = std::max(crush_t0, crush_t1);

                    const float bonus_val = (active_crush > 1) ? 100.0f : 200.0f;
                    const float E_clean_t0 = P_clean[t0p] * bonus_val;
                    const float E_clean_t1 = P_clean[t1p] * bonus_val;

                    const double pair_margin =
                        ((static_cast<double>(E[t0p]) + E_clean_t0) -
                         (static_cast<double>(E[t1p]) + E_clean_t1))
                      * active_crush
                      * b.coefficients[col];
                    team0_margin += pair_margin;
                }
            }
        }

        evs[i] = team_sign * team0_margin;
    }
}

// ---------------------------------------------------------------------------
// V3 — V2 baseline plus four hard-coded strategic rules.
//
// 2v2 design decisions:
//   Rule 1: drop the "aim for 150" only when ALL opponents are scratched
//           (opp_max_potential <= 2 × opp_current). Most conservative.
//   Rule 2: no change — placing high upper-section scores raises golden_max,
//           which blocks ALL opponents simultaneously.
//   Rules 3/4: pure self-evaluation, no opp reference — unchanged.
// ---------------------------------------------------------------------------

static V3Weights g_v3_weights = {
    /*w_r1_per_point=*/  0.005,
    /*w_r2_block_high=*/ 0.10,
    /*w_r3_high_coeff=*/ 0.10,
    /*w_r4_near_clean=*/ 0.20,
    /*r4_threshold=*/    0.50,
};

const V3Weights& v3_weights() { return g_v3_weights; }
void set_v3_weights(const V3Weights& w) { g_v3_weights = w; }

template <typename Traits>
void heuristic_evaluate_v3(const BoardStateT<Traits>& base_board,
                           const GameContextT<Traits>& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables) {
    const V3Weights w = g_v3_weights;
    constexpr double kCleanValue = 200.0;
    constexpr double kColumnAim  = 150.0;

    const int p_me = base_board.current_player;
    const DPTables& dp = tables.dp_tables;

    // Start from V2's baseline (which is already team-aware).
    heuristic_evaluate_v2<Traits>(base_board, base_ctx, requests, request_count, evs, tables);

    for (int i = 0; i < request_count; ++i) {
        BoardStateT<Traits> b = base_board;
        GameContextT<Traits> c = base_ctx;
        apply_placement<Traits>(p_me,
                                requests[i].placement.column,
                                requests[i].placement.row,
                                requests[i].score, b, c);

        const int filled_me = count_filled_cells<Traits>(b, p_me);

        double rule_terms = 0.0;

        for (int col = 0; col < kNumColumns; ++col) {
            const int coeff = b.coefficients[col];
            const bool is_high_coeff = (coeff == 14 || coeff == 16 || coeff == 18);
            const bool is_up_or_down = (col == kColDown || col == kColUp);

            const int empty_me = count_empty_cells<Traits>(b, p_me, col);
            if (empty_me == 0) continue;

            const int T_me = std::max(empty_me, empty_me + (78 - filled_me) / 6);
            const float E_me = get_E_raw<Traits>(p_me, col, T_me, b, c, dp);

            // Rule 1: aim for at least 150 raw points per column.
            // 2v2: drop the aim only when ALL opponents are so scratched they
            //      can't catch up (opp_max <= 2 * opp_current).
            {
                const int my_max = compute_column_potential_score<Traits>(b, c, p_me, col);
                if (my_max >= static_cast<int>(kColumnAim)) {
                    bool aim_dropped_for_all_opps = true;
                    bool any_opp = false;
                    for (int op = 0; op < Traits::kNumPlayers; ++op) {
                        if (Traits::are_teammates(op, p_me)) continue;
                        any_opp = true;
                        const int opp_current = compute_column_raw_score<Traits>(b, c, op, col);
                        const int opp_max     = compute_column_potential_score<Traits>(b, c, op, col);
                        if (opp_max > 2 * opp_current) {
                            aim_dropped_for_all_opps = false;
                            break;
                        }
                    }
                    if (any_opp && !aim_dropped_for_all_opps) {
                        const double deficit = kColumnAim - static_cast<double>(E_me);
                        if (deficit > 0.0) {
                            rule_terms -= deficit * coeff * w.w_r1_per_point;
                        }
                    }
                }
            }

            // Rule 2: block on high-coeff Up/Down by raising upper-section
            // scores. Mechanism (golden_max) already blocks ALL opponents;
            // no team-awareness change needed.
            if (is_up_or_down && is_high_coeff &&
                requests[i].placement.column == col &&
                requests[i].placement.row <= kRow6s &&
                requests[i].score > 0) {
                rule_terms += static_cast<double>(requests[i].score)
                            * coeff * w.w_r2_block_high;
            }

            // Rules 3 & 4: clean-bonus emphasis (self-only).
            {
                const float P_clean = get_P_clean<Traits>(p_me, col, T_me, b, c, dp);
                if (is_high_coeff) {
                    rule_terms += static_cast<double>(P_clean) * kCleanValue
                                * coeff * w.w_r3_high_coeff;
                }
                if (P_clean > w.r4_threshold) {
                    rule_terms += (static_cast<double>(P_clean) - w.r4_threshold)
                                * kCleanValue * coeff * w.w_r4_near_clean;
                }
            }
        }

        // Rules contribute to MY TEAM perspective — V2 baseline already
        // returns my-team margin, and Rule 1's penalty is on MY column EV
        // (a self-improvement signal), so the rule terms add to MY-team
        // margin directly without a team-sign flip.
        evs[i] += rule_terms;
    }
}

// ---------------------------------------------------------------------------
// Research evaluator — configurable V2-style with structural knobs.
//
// 2v2 design decisions:
//   opp_aware_factor: amplify a column's contribution to my-team margin when
//                     max(E[opp] for opp not teammate) > E_me ("behind worst threat").
//   dominance_bonus:  bonus when my_positive_cells > avg(opp_positive_cells),
//                     scaled by (my_pos - avg_opp_pos).
//   output_win_odds:  variance summed across all cross-team pair-duels.
// ---------------------------------------------------------------------------
template <typename Traits>
void heuristic_evaluate_research(const BoardStateT<Traits>& base_board,
                                 const GameContextT<Traits>& base_ctx,
                                 const AfterstateRequest* requests, int request_count,
                                 double* evs, const PrecomputedTables& tables,
                                 const ResearchConfig& cfg) {
    const int p_me = base_board.current_player;
    const DPTables& dp = tables.dp_tables;
    const int team_sign = Traits::are_teammates(p_me, 0) ? +1 : -1;

    for (int i = 0; i < request_count; ++i) {
        BoardStateT<Traits> b = base_board;
        GameContextT<Traits> c = base_ctx;
        apply_placement<Traits>(p_me,
                                requests[i].placement.column,
                                requests[i].placement.row,
                                requests[i].score, b, c);

        int filled[Traits::kNumPlayers];
        int total_empty[Traits::kNumPlayers];
        for (int p = 0; p < Traits::kNumPlayers; ++p) {
            filled[p] = count_filled_cells<Traits>(b, p);
            total_empty[p] = 78 - filled[p];
        }

        double team0_margin = 0.0;
        double global_variance = 0.0;

        for (int col = 0; col < kNumColumns; ++col) {
            const int coeff = b.coefficients[col];

            // Per-player E_raw, P_clean, Var (using player's T strategy).
            float E[Traits::kNumPlayers];
            float P_clean[Traits::kNumPlayers];
            float Var[Traits::kNumPlayers];  // populated only if output_win_odds
            int T[Traits::kNumPlayers];
            for (int p = 0; p < Traits::kNumPlayers; ++p) {
                const int empty = count_empty_cells<Traits>(b, p, col);
                // t_me applies to the active player; t_opp applies to everyone else
                // (including the teammate in 2v2).
                const TStrategy strat = (p == p_me) ? cfg.t_me : cfg.t_opp;
                T[p] = compute_T(strat, empty, filled[p], total_empty[p]);
                E[p] = get_E_raw<Traits>(p, col, T[p], b, c, dp);
                P_clean[p] = get_P_clean<Traits>(p, col, T[p], b, c, dp);
                if (cfg.output_win_odds) {
                    Var[p] = get_E_raw_var<Traits>(p, col, T[p], b, c, dp);
                }
            }

            // "Behind the worst threat?" — true if any non-teammate's E exceeds mine.
            // Used to gate opp_aware_factor amplification on pairings involving me.
            bool me_behind_worst_threat = false;
            if (cfg.opp_aware_factor != 0.0) {
                for (int op = 0; op < Traits::kNumPlayers; ++op) {
                    if (Traits::are_teammates(op, p_me)) continue;
                    if (E[op] > E[p_me]) { me_behind_worst_threat = true; break; }
                }
            }

            // Cross-team pairings.
            for (int ti = 0; ti < Traits::kPlayersPerTeam; ++ti) {
                for (int tj = 0; tj < Traits::kPlayersPerTeam; ++tj) {
                    const int t0p = Traits::kTeam0[ti];
                    const int t1p = Traits::kTeam1[tj];

                    const float active_crush = compute_crush(cfg.crush, E[t0p], E[t1p]);
                    const float bonus_val = (active_crush > 1.0f) ? 100.0f : 200.0f;

                    const float pow_t0 = (cfg.clean_power == 1.0)
                        ? P_clean[t0p]
                        : std::pow(P_clean[t0p], static_cast<float>(cfg.clean_power));
                    const float pow_t1 = (cfg.clean_power == 1.0)
                        ? P_clean[t1p]
                        : std::pow(P_clean[t1p], static_cast<float>(cfg.clean_power));
                    const float E_clean_t0 = pow_t0 * bonus_val;
                    const float E_clean_t1 = pow_t1 * bonus_val;

                    double pair_term =
                        ((static_cast<double>(E[t0p]) + E_clean_t0) -
                         (static_cast<double>(E[t1p]) + E_clean_t1))
                      * active_crush
                      * coeff;

                    double coeff_mult = active_crush * coeff;

                    // opp_aware_factor: amplify pairings involving active player
                    // when active is behind the worst threat in this column.
                    if (me_behind_worst_threat &&
                        (t0p == p_me || t1p == p_me)) {
                        pair_term *= (1.0 + cfg.opp_aware_factor);
                        coeff_mult *= (1.0 + cfg.opp_aware_factor);
                    }

                    team0_margin += pair_term;

                    if (cfg.output_win_odds) {
                        const float var_clean_t0 = P_clean[t0p] * (1.0f - P_clean[t0p])
                                                 * bonus_val * bonus_val;
                        const float var_clean_t1 = P_clean[t1p] * (1.0f - P_clean[t1p])
                                                 * bonus_val * bonus_val;
                        global_variance += (Var[t0p] + var_clean_t0 +
                                            Var[t1p] + var_clean_t1)
                                         * (coeff_mult * coeff_mult);
                    }
                }
            }
        }

        // Flip sign if active player is on Team 1.
        double margin = team_sign * team0_margin;

        // ----- additive per-candidate bonuses/penalties (all from p_me's perspective) -----

        if (cfg.scratch_penalty != 0.0 && requests[i].score == 0) {
            const int col = requests[i].placement.column;
            const int row = requests[i].placement.row;
            const int gmax = base_ctx.golden_max[col][row];
            if (gmax == 0) {
                int row_max;
                if      (row <  kRowSS) row_max = (row + 1) * 5;
                else if (row == kRowSS) row_max = 29;
                else if (row == kRowLS) row_max = 30;
                else if (row == kRowFH) row_max = 50;
                else if (row == kRowK)  row_max = 54;
                else if (row == kRowSTR)row_max = 50;
                else if (row == kRowU8) row_max = 75;
                else                     row_max = 100;
                margin -= cfg.scratch_penalty * row_max * base_board.coefficients[col];
            }
        }

        if (cfg.variance_penalty != 0.0 && requests[i].score > 0) {
            const int row = requests[i].placement.row;
            const int col = requests[i].placement.column;
            if (row == kRowY || row == kRowU8) {
                margin -= cfg.variance_penalty * base_board.coefficients[col];
            }
        }

        if (cfg.early_high_coeff_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 14) {
                const int empty_in_col = count_empty_cells<Traits>(base_board, p_me, col);
                margin += cfg.early_high_coeff_bonus * empty_in_col * coeff;
            }
        }

        if (cfg.early_progressive_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 14) {
                const int empty_in_col = count_empty_cells<Traits>(base_board, p_me, col);
                const double phase = std::max(0, 78 - filled[p_me]) / 78.0;
                margin += cfg.early_progressive_bonus * empty_in_col * coeff * phase;
            }
        }

        if (cfg.coeff_sq_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            const int empty_in_col = count_empty_cells<Traits>(base_board, p_me, col);
            margin += cfg.coeff_sq_bonus * empty_in_col * coeff * coeff;
        }

        if (cfg.completion_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            const int filled_in_col = 13 - count_empty_cells<Traits>(base_board, p_me, col);
            margin += cfg.completion_bonus * filled_in_col * coeff;
        }

        // dominance_bonus — 2v2: "more cells than AVERAGE opponent."
        if (cfg.dominance_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            int my_pos = 0;
            for (int r = 0; r < kNumRows; ++r)
                if (base_board.cells[p_me][col][r] > 0) ++my_pos;

            int opp_pos_sum = 0;
            int opp_count = 0;
            for (int op = 0; op < Traits::kNumPlayers; ++op) {
                if (Traits::are_teammates(op, p_me)) continue;
                ++opp_count;
                for (int r = 0; r < kNumRows; ++r)
                    if (base_board.cells[op][col][r] > 0) ++opp_pos_sum;
            }
            if (opp_count > 0) {
                const double avg_opp_pos = static_cast<double>(opp_pos_sum) / opp_count;
                if (my_pos > avg_opp_pos) {
                    margin += cfg.dominance_bonus * (my_pos - avg_opp_pos) * coeff;
                }
            }
        }

        if (cfg.upper100_bonus != 0.0 && requests[i].placement.row <= kRow6s
            && requests[i].score > 0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            const int upper_now = base_ctx.upper_sum[p_me][col];
            const double t = std::min(1.0, std::max(0.0, upper_now / 100.0));
            margin += cfg.upper100_bonus * requests[i].score * coeff * t;
        }

        if (cfg.hc_scratch_penalty != 0.0 && requests[i].score == 0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 14) {
                margin -= cfg.hc_scratch_penalty * coeff;
            }
        }

        if (cfg.upper_priority_bonus != 0.0 && requests[i].score > 0) {
            const int row = requests[i].placement.row;
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (row >= kRow4s && row <= kRow6s) {
                margin += cfg.upper_priority_bonus * requests[i].score * coeff;
            }
        }

        if (cfg.coeff_sq_high_only != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 14) {
                const int empty_in_col = count_empty_cells<Traits>(base_board, p_me, col);
                margin += cfg.coeff_sq_high_only * empty_in_col * coeff * coeff;
            }
        }

        if (cfg.turbo_avoidance != 0.0
            && requests[i].placement.column == kColTurbo) {
            const double phase = std::max(0, 78 - filled[p_me]) / 78.0;
            margin -= cfg.turbo_avoidance * phase * 100.0;
        }

        if (cfg.last_cell_bonus != 0.0) {
            const int col = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (count_empty_cells<Traits>(base_board, p_me, col) == 1) {
                margin += cfg.last_cell_bonus * coeff;
            }
        }

        if (cfg.upper_bonus_penalty != 0.0
            && requests[i].placement.row <= kRow6s && requests[i].placement.row >= kRow3s
            && requests[i].placement.column != kColDown) {
            const int col   = requests[i].placement.column;
            const int coeff = base_board.coefficients[col];
            if (coeff >= 12) {
                const int row   = requests[i].placement.row;
                const int face  = row + 1;
                const int par   = face * 3;
                const int score = requests[i].score;
                if (score < par) {
                    margin -= cfg.upper_bonus_penalty * coeff;
                }
            }
        }

        evs[i] = margin;

        // Optional V3 rule layer (additive, R3-only inline).
        if (cfg.use_v3_rules) {
            const int col_i = requests[i].placement.column;
            const int empty_in_col = count_empty_cells<Traits>(b, p_me, col_i);
            const int T_me_local = compute_T(cfg.t_me, empty_in_col, filled[p_me],
                                             total_empty[p_me]);
            const float P_clean_me_local = get_P_clean<Traits>(p_me, col_i, T_me_local,
                                                               b, c, dp);
            const int coeff = b.coefficients[col_i];
            const bool is_high = (coeff >= 14);
            if (is_high && cfg.v3.w_r3_high_coeff > 0.0) {
                evs[i] += static_cast<double>(P_clean_me_local) * 200.0
                        * coeff * cfg.v3.w_r3_high_coeff;
            }
        }

        if (cfg.output_win_odds) {
            if (global_variance <= 1e-6) {
                evs[i] = evs[i] > 0.0 ? 1.0 : (evs[i] < 0.0 ? 0.0 : 0.5);
            } else {
                double std_dev = std::sqrt(std::max(0.0, global_variance));
                evs[i] = 0.5 * (1.0 + fast_erf(evs[i] / (std_dev * 1.41421356237)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Research config registry — V4..V17.
// ---------------------------------------------------------------------------
const ResearchConfig& get_research_config_for(HeuristicVersion v) {
    static ResearchConfig cache[18];
    static bool initialized[18] = {false};

    int idx = static_cast<int>(v);
    if (idx < 4 || idx > 17) {
        throw std::invalid_argument("get_research_config_for: V1/V2/V3 use their own evaluators");
    }

    if (initialized[idx]) {
        return cache[idx];
    }

    ResearchConfig c;
    switch (v) {
        case HeuristicVersion::V4:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 1.0;
            break;
        case HeuristicVersion::V5:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 1.5;
            break;
        case HeuristicVersion::V6:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.20;
            break;
        case HeuristicVersion::V7:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.10;
            c.turbo_avoidance = 1.0;
            break;
        case HeuristicVersion::V8:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.20;
            c.turbo_avoidance = 1.0;
            break;
        case HeuristicVersion::V9:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.50;
            c.turbo_avoidance = 1.0;
            break;
        case HeuristicVersion::V10:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 0.5;
            c.coeff_sq_bonus = 0.12;
            break;
        case HeuristicVersion::V11:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 1.0;
            c.coeff_sq_bonus = 0.05;
            break;
        case HeuristicVersion::V12:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.early_high_coeff_bonus = 0.5;
            c.coeff_sq_bonus = 0.20;
            break;
        case HeuristicVersion::V13:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.15;
            c.turbo_avoidance = 1.0;
            c.use_v3_rules = true;
            c.v3.w_r3_high_coeff = 0.10;
            break;
        case HeuristicVersion::V14:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.15;
            c.turbo_avoidance = 1.0;
            c.use_v3_rules = true;
            c.v3.w_r3_high_coeff = 0.10;
            c.upper_bonus_penalty = 20.0;
            break;
        case HeuristicVersion::V15:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.15;
            c.turbo_avoidance = 1.0;
            c.use_v3_rules = true;
            c.v3.w_r3_high_coeff = 0.10;
            c.upper_bonus_penalty = 200.0;
            break;
        case HeuristicVersion::V16:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.15;
            c.turbo_avoidance = 1.0;
            c.use_v3_rules = true;
            c.v3.w_r3_high_coeff = 0.10;
            c.output_win_odds = true;
            break;
        case HeuristicVersion::V17:
            c.crush = CrushMode::FLOAT_SMOOTH;
            c.coeff_sq_bonus = 0.15;
            c.turbo_avoidance = 1.0;
            c.use_v3_rules = true;
            c.v3.w_r3_high_coeff = 0.10;
            c.upper_bonus_penalty = 20.0;
            c.output_win_odds = true;
            break;
        default:
            throw std::invalid_argument("get_research_config_for: unknown version");
    }
    cache[idx] = c;
    initialized[idx] = true;
    return cache[idx];
}

// ---------------------------------------------------------------------------
// Play-turn functions — currently 1v1-only.
// These depend on solver_get_requests / solver_resolve_greedy, which take the
// non-templated GameState/GameContext. Will be templated alongside the
// solver in a later migration task.
// ---------------------------------------------------------------------------

void heuristic_play_turn_research(GameState& state, GameContext& ctx,
                                  const PrecomputedTables& tables,
                                  SolverBuffers& buffers, RNG& rng,
                                  const ResearchConfig& cfg) {
    buffers.dp_computed = false;
    solver_get_requests(state, ctx, tables, buffers);
    assert(buffers.request_count > 0);
    heuristic_evaluate_research<Yams1v1>(state.board, ctx,
                                         buffers.requests, buffers.request_count,
                                         buffers.evs, tables, cfg);
    while (true) {
        SolverResult result = solver_resolve_greedy(state, ctx, tables, buffers);
        if (result.should_place) {
            perform_placement<Yams1v1>(state, ctx, result.placement.column,
                                       result.placement.row, rng);
            return;
        }
        if (!can_reroll<Yams1v1>(state, ctx)) {
            int current_id = get_dice_state_id(state.dice, tables);
            int16_t req_idx = buffers.stop_request_idx[current_id];
            if (req_idx < 0) req_idx = 0;
            perform_placement<Yams1v1>(state, ctx, buffers.requests[req_idx].placement.column,
                                       buffers.requests[req_idx].placement.row, rng);
            return;
        }
        perform_reroll<Yams1v1>(state, result.hold_mask, rng);
    }
}

void heuristic_play_turn(GameState& state, GameContext& ctx,
                         const PrecomputedTables& tables,
                         SolverBuffers& buffers, RNG& rng,
                         HeuristicVersion version) {
    buffers.dp_computed = false;

    solver_get_requests(state, ctx, tables, buffers);
    assert(buffers.request_count > 0);

    if (static_cast<int>(version) >= static_cast<int>(HeuristicVersion::V4)) {
        const ResearchConfig& cfg = get_research_config_for(version);
        heuristic_evaluate_research<Yams1v1>(state.board, ctx,
                                             buffers.requests, buffers.request_count,
                                             buffers.evs, tables, cfg);
    } else if (version == HeuristicVersion::V3) {
        heuristic_evaluate_v3<Yams1v1>(state.board, ctx,
                                       buffers.requests, buffers.request_count,
                                       buffers.evs, tables);
    } else if (version == HeuristicVersion::V2) {
        heuristic_evaluate_v2<Yams1v1>(state.board, ctx,
                                       buffers.requests, buffers.request_count,
                                       buffers.evs, tables);
    } else {
        heuristic_evaluate<Yams1v1>(state.board, ctx,
                                    buffers.requests, buffers.request_count, buffers.evs);
    }

    while (true) {
        SolverResult result = solver_resolve_greedy(state, ctx, tables, buffers);

        if (result.should_place) {
            perform_placement<Yams1v1>(state, ctx, result.placement.column,
                                       result.placement.row, rng);
            return;
        }

        if (!can_reroll<Yams1v1>(state, ctx)) {
            int current_id = get_dice_state_id(state.dice, tables);
            int16_t req_idx = buffers.stop_request_idx[current_id];
            if (req_idx < 0) req_idx = 0;
            perform_placement<Yams1v1>(state, ctx, buffers.requests[req_idx].placement.column,
                                       buffers.requests[req_idx].placement.row, rng);
            return;
        }
        perform_reroll<Yams1v1>(state, result.hold_mask, rng);
    }
}

int play_heuristic_game(RNG& rng, const PrecomputedTables& tables,
                        HeuristicVersion version) {
    GameState state;
    GameContext ctx;
    SolverBuffers buffers;

    init_game<Yams1v1>(state, ctx, rng);

    while (!is_terminal<Yams1v1>(state.board)) {
        heuristic_play_turn(state, ctx, tables, buffers, rng, version);
    }

    return get_game_result<Yams1v1>(state, ctx);
}

HeuristicVersion random_heuristic_version(RNG& rng) {
    static constexpr HeuristicVersion kAll[] = {
        HeuristicVersion::V1,  HeuristicVersion::V2,  HeuristicVersion::V3,
        HeuristicVersion::V4,  HeuristicVersion::V5,  HeuristicVersion::V6,
        HeuristicVersion::V7,  HeuristicVersion::V8,  HeuristicVersion::V9,
        HeuristicVersion::V10, HeuristicVersion::V11, HeuristicVersion::V12,
        HeuristicVersion::V13, HeuristicVersion::V14, HeuristicVersion::V15,
        HeuristicVersion::V16, HeuristicVersion::V17,
    };
    constexpr int kCount = sizeof(kAll) / sizeof(kAll[0]);
    return kAll[rng.next() % kCount];
}

// ---------------------------------------------------------------------------
// Explicit instantiations of the templated evaluators.
// ---------------------------------------------------------------------------

template void heuristic_evaluate<Yams1v1>(const BoardStateT<Yams1v1>&,
                                          const GameContextT<Yams1v1>&,
                                          const AfterstateRequest*, int, double*);
template void heuristic_evaluate<Yams2v2>(const BoardStateT<Yams2v2>&,
                                          const GameContextT<Yams2v2>&,
                                          const AfterstateRequest*, int, double*);
template void heuristic_evaluate_v2<Yams1v1>(const BoardStateT<Yams1v1>&,
                                             const GameContextT<Yams1v1>&,
                                             const AfterstateRequest*, int, double*,
                                             const PrecomputedTables&);
template void heuristic_evaluate_v2<Yams2v2>(const BoardStateT<Yams2v2>&,
                                             const GameContextT<Yams2v2>&,
                                             const AfterstateRequest*, int, double*,
                                             const PrecomputedTables&);
template void heuristic_evaluate_v3<Yams1v1>(const BoardStateT<Yams1v1>&,
                                             const GameContextT<Yams1v1>&,
                                             const AfterstateRequest*, int, double*,
                                             const PrecomputedTables&);
template void heuristic_evaluate_v3<Yams2v2>(const BoardStateT<Yams2v2>&,
                                             const GameContextT<Yams2v2>&,
                                             const AfterstateRequest*, int, double*,
                                             const PrecomputedTables&);
template void heuristic_evaluate_research<Yams1v1>(const BoardStateT<Yams1v1>&,
                                                   const GameContextT<Yams1v1>&,
                                                   const AfterstateRequest*, int, double*,
                                                   const PrecomputedTables&,
                                                   const ResearchConfig&);
template void heuristic_evaluate_research<Yams2v2>(const BoardStateT<Yams2v2>&,
                                                   const GameContextT<Yams2v2>&,
                                                   const AfterstateRequest*, int, double*,
                                                   const PrecomputedTables&,
                                                   const ResearchConfig&);
