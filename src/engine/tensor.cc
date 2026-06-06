#include "engine/tensor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#include "engine/duel.h"  // upper_section_bonus, crush_multiplier
#include "engine/game_traits.h"
#include "engine/placement.h"
#include "solver/dp_eval.h"
#include "solver/dp_tables.h"

// ---------------------------------------------------------------------------
// Existing helpers
// ---------------------------------------------------------------------------

template <typename Traits>
int count_empty_cells(const BoardStateT<Traits>& board, int player, int column) {
    int n = 0;
    for (int row = 0; row < kNumRows; ++row)
        if (board.cells[player][column][row] == kCellEmpty) ++n;
    return n;
}

template <typename Traits>
int count_filled_cells(const BoardStateT<Traits>& board, int player) {
    int n = 0;
    for (int col = 0; col < kNumColumns; ++col)
        for (int row = 0; row < kNumRows; ++row)
            if (board.cells[player][col][row] != kCellEmpty) ++n;
    return n;
}

template <typename Traits>
int sum_all_filled(const BoardStateT<Traits>& board, int player) {
    int s = 0;
    for (int col = 0; col < kNumColumns; ++col) {
        for (int row = 0; row < kNumRows; ++row) {
            int8_t v = board.cells[player][col][row];
            if (v > 0) s += v;
        }
    }
    return s;
}

template <typename Traits>
int compute_column_raw_score(const BoardStateT<Traits>& board,
                             const GameContextT<Traits>& ctx,
                             int player, int column) {
    int cell_sum = 0;
    for (int row = 0; row < kNumRows; ++row) {
        int8_t v = board.cells[player][column][row];
        if (v > 0) cell_sum += v;
    }
    return cell_sum + upper_section_bonus(ctx.upper_sum[player][column]);
}

template <typename Traits>
int compute_column_potential_score(const BoardStateT<Traits>& board,
                                   const GameContextT<Traits>& ctx,
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
    }
    return cell_sum + upper_section_bonus(upper_potential);
}

template <typename Traits>
int compute_total_potential(const BoardStateT<Traits>& board, int player) {
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
// Explicit instantiations of the helpers above (used by engine + heuristic).
// generate_tensor / generate_tensor_batch stay 1v1-only for now; Task 5
// rewrites them with rotational invariance.
// ---------------------------------------------------------------------------

template int count_empty_cells<Yams1v1>(const BoardStateT<Yams1v1>&, int, int);
template int count_empty_cells<Yams2v2>(const BoardStateT<Yams2v2>&, int, int);
template int count_filled_cells<Yams1v1>(const BoardStateT<Yams1v1>&, int);
template int count_filled_cells<Yams2v2>(const BoardStateT<Yams2v2>&, int);
template int sum_all_filled<Yams1v1>(const BoardStateT<Yams1v1>&, int);
template int sum_all_filled<Yams2v2>(const BoardStateT<Yams2v2>&, int);
template int compute_column_raw_score<Yams1v1>(const BoardStateT<Yams1v1>&,
                                               const GameContextT<Yams1v1>&,
                                               int, int);
template int compute_column_raw_score<Yams2v2>(const BoardStateT<Yams2v2>&,
                                               const GameContextT<Yams2v2>&,
                                               int, int);
template int compute_column_potential_score<Yams1v1>(const BoardStateT<Yams1v1>&,
                                                     const GameContextT<Yams1v1>&,
                                                     int, int);
template int compute_column_potential_score<Yams2v2>(const BoardStateT<Yams2v2>&,
                                                     const GameContextT<Yams2v2>&,
                                                     int, int);
template int compute_total_potential<Yams1v1>(const BoardStateT<Yams1v1>&, int);
template int compute_total_potential<Yams2v2>(const BoardStateT<Yams2v2>&, int);

// ---------------------------------------------------------------------------
// V2.1 helpers — DP state encoders + horizon allocators live in
// solver/dp_eval (shared with the V2 heuristic bot).
// ---------------------------------------------------------------------------
namespace {

// Crush-aware "points needed to crush at level N" projector, returned as a
// signed, scale-normalised feature.
inline float pts_to_Nx(int N, float me, float opp, float scale) {
    if (me >= opp) {
        return std::max(0.0f, N * opp + 1.0f - me) / scale;
    } else {
        return -std::max(0.0f, N * me + 1.0f - opp) / scale;
    }
}

template <typename Traits>
static void copy_tensor_context_fields(const GameContextT<Traits>& src,
                                       GameContextT<Traits>& dst) {
    std::memcpy(dst.golden_max, src.golden_max, sizeof(dst.golden_max));
    std::memcpy(dst.upper_sum, src.upper_sum, sizeof(dst.upper_sum));
    std::memcpy(dst.ss_scratched, src.ss_scratched, sizeof(dst.ss_scratched));
    std::memcpy(dst.ls_scratched, src.ls_scratched, sizeof(dst.ls_scratched));
    std::memcpy(dst.lower_has_scratch, src.lower_has_scratch,
                sizeof(dst.lower_has_scratch));
}

// Maximum POSITIVE Small-Sum value a player could still legally place in this
// column, used for SS/LS interlock-poison reasoning (Group G). Ignores the
// golden-rule floor (the caller compares against the poison threshold directly).
// Returns -1 if the player cannot place any positive SS here: SS already
// resolved, LS forced its scratch, or a low filled LS caps SS out of existence.
template <typename Traits>
static int max_placeable_ss(int q, int col, const BoardStateT<Traits>& board,
                            const GameContextT<Traits>& ctx) {
    if (board.cells[q][col][kRowSS] != kCellEmpty) return -1;  // already resolved
    if (ctx.ls_scratched[q][col])                  return -1;  // forced scratch
    const int ls = board.cells[q][col][kRowLS];
    if (ls == kCellEmpty) return 29;     // LS open → SS up to 29
    if (ls <= 0)          return -1;     // LS scratched (defensive; ls_scratched covers)
    if (ls <= 20)         return -1;     // SS must be < LS and >= 20 → impossible
    if (ls <= 29)         return ls - 1; // SS strictly below a filled LS
    return 29;                           // LS >= 30 → SS up to 29
}

}  // namespace

// ---------------------------------------------------------------------------
// generate_tensor — templated implementation with canonical rotational view.
// ---------------------------------------------------------------------------

struct PCData {
    int empty_in_col;
    int upper_sum;
    float e_raw;
    int raw_score;
    int pot_score;
    bool is_clean;        // matches compute_duel: upper_sum >= 60 && !lower_has_scratch
    bool clean_possible;  // upper_potential >= 60 && !lower_has_scratch (for pot_score upper bound)
    float p_one[13]; // kNumRows
    float grp_E[18];
    float grp_F[15];
};

template <typename Traits>
static void compute_pc_data(const BoardStateT<Traits>& board,
                            const GameContextT<Traits>& ctx,
                            int p, int col, int player_filled,
                            const PrecomputedTables& tables,
                            PCData& d) {
    const DPTables& dp = tables.dp_tables;
    const bool dp_ready = !dp.dp_t1.empty();
    static const int kUpperTargets[5] = {60, 70, 80, 90, 100};

    d.upper_sum = ctx.upper_sum[p][col];
    d.raw_score = compute_column_raw_score<Traits>(board, ctx, p, col);
    d.pot_score = compute_column_potential_score<Traits>(board, ctx, p, col);
    d.is_clean = (ctx.upper_sum[p][col] >= 60) && (!ctx.lower_has_scratch[p][col]);
    // Upper-bound clean-possible: column can still earn the clean bonus iff
    // the upper section can still reach 60 and no lower row has scratched yet.
    int upper_potential_for_clean = ctx.upper_sum[p][col];
    for (int row = 0; row <= kRow6s; ++row) {
        if (board.cells[p][col][row] == kCellEmpty)
            upper_potential_for_clean += kMaxScorePerRow[row];
    }
    d.clean_possible = (upper_potential_for_clean >= 60) && (!ctx.lower_has_scratch[p][col]);

    for (int row = 0; row < kNumRows; ++row) {
        int8_t v = board.cells[p][col][row];
        if (v != kCellEmpty) {
            d.p_one[row] = 1.0f;
            continue;
        }
        if (row == kRowSS && ctx.ls_scratched[p][col]) {
            d.p_one[row] = 0.0f; continue;
        }
        if (row == kRowLS && ctx.ss_scratched[p][col]) {
            d.p_one[row] = 0.0f; continue;
        }
        int golden_min = ctx.golden_max[col][row];
        if (golden_min == 0) golden_min = 1;
        if (row == kRowSS) {
            // SS must stay strictly below a filled LS (scoring.cc:67-69). With
            // SS's natural floor of 20, no legal SS sum remains once the
            // effective lower bound reaches the filled LS value → forced
            // scratch. (A still-open band below LS is not modelled here; p_one
            // then mildly overestimates, matching the DP's min-threshold
            // limitation.)
            int8_t ls_val = board.cells[p][col][kRowLS];
            if (ls_val != kCellEmpty && ls_val > 0) {
                int ss_floor = std::max(20, static_cast<int>(ctx.golden_max[col][kRowSS]));
                if (ss_floor >= ls_val) { d.p_one[row] = 0.0f; continue; }
            }
        }
        if (row == kRowLS) {
            int max_ss = ctx.golden_max[col][kRowSS];
            if (max_ss > 0) golden_min = std::max(golden_min, max_ss + 1);
        }
        int max_for_row = kMaxScorePerRow[row];
        if (golden_min > max_for_row) {
            d.p_one[row] = 0.0f; continue;
        }
        int thresh = std::min(golden_min, 100);
        float p_one = (col == kColTurbo)
            ? tables.prob_tables.prob_2rolls_compound[row][thresh][1]
            : tables.prob_tables.prob_3rolls_compound[row][thresh][1];
        d.p_one[row] = p_one;
    }

    int8_t Sc_U[6], Sc_M[3], Sc_L[5];
    int EU, EM, EL;
    build_Sc<Traits>(p, col, board, ctx, Sc_U, Sc_M, Sc_L, EU, EM, EL);
    d.empty_in_col = EU + EM + EL;

    int T_min = d.empty_in_col;
    int T_max = std::max(T_min, 78 - player_filled);
    int T_mid = (T_min + T_max) / 2;
    int Ts[3] = {T_min, T_mid, T_max};

    int cur_upper = ctx.upper_sum[p][col];
    int R_clean = std::max(0, 60 - cur_upper);
    bool low_scratch = ctx.lower_has_scratch[p][col];
    Variant v = get_variant(col);

    int e_idx = 0;
    int f_idx = 0;
    float eu_min = 0, em_min = 0, el_min = 0;

    for (int ti = 0; ti < 3; ++ti) {
        int T = Ts[ti];
        
        if (ti > 0 && T == Ts[ti - 1]) {
            for(int i=0; i<6; ++i) d.grp_E[e_idx + i] = d.grp_E[e_idx - 6 + i];
            e_idx += 6;
            for(int i=0; i<5; ++i) d.grp_F[f_idx + i] = d.grp_F[f_idx - 5 + i];
            f_idx += 5;
            continue;
        }

        int TU, TM, TL;
        apportion_turns(T, EU, EM, EL, TU, TM, TL);

        if (dp_ready) {
            for (int k = 0; k < 5; ++k) {
                int R = kUpperTargets[k] - cur_upper;
                if (R < 0) R = 0;
                d.grp_E[e_idx++] = get_upper_prob(dp, v, Sc_U, TU, R);
            }
            float eu = get_upper_ev(dp, v, Sc_U, TU, cur_upper);
            d.grp_E[e_idx++] = std::min(1.0f, std::max(0.0f, eu / 100.0f));

            float p_mid = get_middle_prob(dp, v, Sc_M, TM);
            float ev_mid = get_middle_ev (dp, v, Sc_M, TM);
            float p_low = get_lower_prob (dp, v, Sc_L, TL);
            float ev_low = get_lower_ev  (dp, v, Sc_L, TL);
            float p_up_60 = get_upper_prob(dp, v, Sc_U, TU, R_clean);

            d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, p_mid));
            d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, ev_mid / 60.0f));
            d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, p_low));
            d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, ev_low / 250.0f));
            float p_clean = low_scratch ? 0.0f : (p_up_60 * p_mid * p_low);
            d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, p_clean));

            if (ti == 0) {
                eu_min = eu;
                em_min = ev_mid;
                el_min = ev_low;
            }
        } else {
            for(int i=0; i<6; ++i) d.grp_E[e_idx++] = 0.0f;
            for(int i=0; i<5; ++i) d.grp_F[f_idx++] = 0.0f;
        }
    }

    int filled_score = 0;
    for (int r = 6; r <= 12; ++r) {
        int8_t cv = board.cells[p][col][r];
        if (cv > 0) filled_score += cv;
    }
    d.e_raw = eu_min + em_min + el_min + static_cast<float>(filled_score);
}

// Compute the canonical seat order relative to the active player.
//   canonical[ci] = (active + ci) % kNumPlayers
//
// For 1v1: [active, opp].
// For 2v2: [Active, NextOpp, Teammate, PrevOpp].
template <typename Traits>
static inline void make_canonical_order(int active, int (&canonical)[Traits::kNumPlayers]) {
    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        canonical[ci] = (active + ci) % Traits::kNumPlayers;
    }
}

template <typename Traits>
static void write_tensor_from_pc(const BoardStateT<Traits>& board, int player,
                                 const GameContextT<Traits>& ctx,
                                 const PCData pc[Traits::kNumPlayers][6], float* out) {
    int canonical[Traits::kNumPlayers];
    make_canonical_order<Traits>(player, canonical);
    int idx = 0;

    // Group A — per-player × per-cell {filled flag, normalised value}.
    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        const int p = canonical[ci];
        for (int col = 0; col < kNumColumns; ++col) {
            for (int row = 0; row < kNumRows; ++row) {
                int8_t v = board.cells[p][col][row];
                if (v == kCellEmpty) {
                    out[idx++] = 0.0f;
                    out[idx++] = 0.0f;
                } else {
                    out[idx++] = 1.0f;
                    out[idx++] = static_cast<float>(v) / static_cast<float>(kMaxScorePerRow[row]);
                }
            }
        }
    }

    // Group B.1 — per-player × per-column {upper_sum/100, e_raw/500}.
    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        for (int col = 0; col < kNumColumns; ++col) {
            out[idx++] = std::min(1.0f, static_cast<float>(pc[ci][col].upper_sum) / 100.0f);
            out[idx++] = std::min(1.0f, std::max(0.0f, pc[ci][col].e_raw / 500.0f));
        }
    }

    // Group B.2 — per-PAIRING × per-column duel summaries (14 features each).
    // Pairings are in canonical order: 1v1 has 1; 2v2 has 4 (Active×NextOpp,
    // Active×PrevOpp, Teammate×NextOpp, Teammate×PrevOpp). The summed margin
    // across all pairings is later emitted in Group D as a team aggregate.
    long long total_duel_now = 0;
    double    total_duel_E = 0.0;
    for (int pi = 0; pi < Traits::kNumPairings; ++pi) {
        const int t0_ci = Traits::kCanonicalPairingT0[pi];
        const int t1_ci = Traits::kCanonicalPairingT1[pi];
        for (int col = 0; col < kNumColumns; ++col) {
            const int coeff = board.coefficients[col];
            const int rem_me  = pc[t0_ci][col].empty_in_col;
            const int rem_opp = pc[t1_ci][col].empty_in_col;
            out[idx++] = static_cast<float>(rem_me)  / static_cast<float>(kNumRows);
            out[idx++] = static_cast<float>(rem_opp) / static_cast<float>(kNumRows);

            const int my_r_raw  = pc[t0_ci][col].raw_score;
            const int opp_r_raw = pc[t1_ci][col].raw_score;
            const int crush_my  = crush_multiplier(my_r_raw,  opp_r_raw);
            const int crush_opp = crush_multiplier(opp_r_raw, my_r_raw);
            const int active_crush = std::max(crush_my, crush_opp);
            // Apply per-pairing clean-column bonus the same way compute_duel does:
            // bonus value is 200 when no crush, 100 when crush, applied to each
            // clean player within the pairing.
            const int clean_bonus_val = (active_crush > 1) ? 100 : 200;
            const int my_r  = my_r_raw  + (pc[t0_ci][col].is_clean ? clean_bonus_val : 0);
            const int opp_r = opp_r_raw + (pc[t1_ci][col].is_clean ? clean_bonus_val : 0);
            const long long margin_now =
                static_cast<long long>(my_r - opp_r) * coeff * active_crush;
            out[idx++] = std::tanh(static_cast<float>(margin_now) / 15000.0f);
            total_duel_now += margin_now;

            const int my_eR  = static_cast<int>(pc[t0_ci][col].e_raw + 0.5f);
            const int opp_eR = static_cast<int>(pc[t1_ci][col].e_raw + 0.5f);
            const int crush_my_E  = crush_multiplier(my_eR,  opp_eR);
            const int crush_opp_E = crush_multiplier(opp_eR, my_eR);
            const int active_crush_E = std::max(crush_my_E, crush_opp_E);
            // Fold the expected clean-column bonus into the margin, weighted by
            // P_clean (T_min tier, grp_F[4]) so the EV side anticipates an
            // impending clean completion instead of stepping into it only when
            // is_clean flips. Mirrors the now-margin's clean handling above.
            const double margin_E = expected_duel_margin(
                pc[t0_ci][col].e_raw, pc[t1_ci][col].e_raw,
                pc[t0_ci][col].grp_F[4], pc[t1_ci][col].grp_F[4],
                coeff, active_crush_E);
            out[idx++] = std::tanh(static_cast<float>(margin_E / 15000.0));
            total_duel_E += margin_E;

            out[idx++] = static_cast<float>(crush_my   - crush_opp)   / 5.0f;
            out[idx++] = static_cast<float>(crush_my_E - crush_opp_E) / 5.0f;

            const float curr_scales[4] = {500.0f, 750.0f, 1000.0f, 1250.0f};
            for (int k = 0; k < 4; ++k) {
                // pts_to_Nx works against raw (pre-clean-bonus) scores because
                // crush thresholds are evaluated on raw_score, not adj_score.
                out[idx++] = pts_to_Nx(k + 2,
                                       static_cast<float>(my_r_raw),
                                       static_cast<float>(opp_r_raw),
                                       curr_scales[k]);
            }
            for (int k = 0; k < 4; ++k) {
                out[idx++] = pts_to_Nx(k + 2,
                                       pc[t0_ci][col].e_raw,
                                       pc[t1_ci][col].e_raw,
                                       curr_scales[k]);
            }
        }
    }

    // Group C — per-player × per-cell single-turn-non-scratch probability.
    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        for (int col = 0; col < kNumColumns; ++col) {
            for (int row = 0; row < kNumRows; ++row) {
                out[idx++] = pc[ci][col].p_one[row];
            }
        }
    }

    // Group D — global aggregates and phase flags.
    for (int col = 0; col < kNumColumns; ++col) {
        out[idx++] = static_cast<float>(board.coefficients[col]) / 18.0f;
    }
    out[idx++] = static_cast<float>(board.cells_filled) /
                 static_cast<float>(Traits::kTotalCells);
    // The margin-now / margin-E totals are now MY-TEAM margins (sum of all
    // canonical pairings, each defined as "my team's player − their player").
    // In 1v1 the single pairing reproduces the previous behaviour bit-for-bit.
    out[idx++] = std::tanh(static_cast<float>(total_duel_now) /
                           (80000.0f * Traits::kPlayersPerTeam));
    out[idx++] = std::tanh(static_cast<float>(total_duel_E)   /
                           (100000.0 * Traits::kPlayersPerTeam));

    // "Locked-in" margin flags: did MY TEAM already secure / lose the game
    // (against the opposing team's max / min potential)? In 1v1 this collapses
    // to the previous my-vs-opp comparison.
    //
    // For the upper bound used in opp_max / my_max we add the maximum possible
    // clean-column bonus (200, taken when there is no crush) on top of
    // pot_score whenever the column can still earn it. Without this the
    // "locked-in win" flag (my_min > opp_max) can fire prematurely, since
    // compute_duel can add up to 200 per-column to opp's adj score.
    int my_min = 0, my_max = 0, opp_min = 0, opp_max = 0;
    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        // Canonical seats: 0, 2, ... are "my team"; 1, 3, ... are opposing.
        const bool on_my_team = ((ci % 2) == 0);
        for (int col = 0; col < kNumColumns; ++col) {
            const int min_adj = pc[ci][col].raw_score +
                                (pc[ci][col].is_clean ? 100 : 0);  // crush ⇒ 100
            const int max_adj = pc[ci][col].pot_score +
                                (pc[ci][col].clean_possible ? 200 : 0);
            if (on_my_team) {
                my_min += min_adj;
                my_max += max_adj;
            } else {
                opp_min += min_adj;
                opp_max += max_adj;
            }
        }
    }
    out[idx++] = (my_min > opp_max) ? 1.0f : 0.0f;
    out[idx++] = (my_max < opp_min) ? 1.0f : 0.0f;

    // Phase flags — thresholds scaled to total cells. 1v1: 50/100/140 of 156.
    // 2v2: 100/200/280 of 312 (same proportions: 32%, 64%, 90%).
    const int filled = static_cast<int>(board.cells_filled);
    constexpr int kPhase1 = (Traits::kTotalCells * 50) / 156;
    constexpr int kPhase2 = (Traits::kTotalCells * 100) / 156;
    constexpr int kPhase3 = (Traits::kTotalCells * 140) / 156;
    out[idx++] = (filled > kPhase1) ? 1.0f : 0.0f;
    out[idx++] = (filled > kPhase2) ? 1.0f : 0.0f;
    out[idx++] = (filled > kPhase3) ? 1.0f : 0.0f;

    // Group E — per-player × per-column × 18 DP upper features.
    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        for (int col = 0; col < kNumColumns; ++col) {
            for (int i = 0; i < 18; ++i) {
                out[idx++] = pc[ci][col].grp_E[i];
            }
        }
    }

    // Group F — per-player × per-column × 15 DP mid/low/clean features.
    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        for (int col = 0; col < kNumColumns; ++col) {
            for (int i = 0; i < 15; ++i) {
                out[idx++] = pc[ci][col].grp_F[i];
            }
        }
    }

    // Group G (tensor V2) — SS/LS interlock poison features, per-player × column
    // (canonical order), 2 per cell. Appended last so the V1 layout above stays
    // a byte-exact prefix (the distillation teacher reads only that prefix).
    //
    //   G0 — DEFENSIVE exposure: this player has a committed positive LS (value
    //        L) with its SS still open and no lower scratch yet, and an OPPONENT
    //        could still place an SS >= L. That SS raises the shared golden_max
    //        floor to/above L, forcing this player's SS to scratch → mutual
    //        destruction (LS -> 0) + clean-bonus loss. Magnitude (30 - L)/10 is
    //        the poison window: L=20 -> 1.0 (wide), L=29 -> 0.1, L=30 -> safe.
    //   G1 — OFFENSIVE leverage: this player has an open SS it could push to >=
    //        an opponent's committed-low LS, poisoning that opponent's column.
    //        Magnitude is the best (30 - L_opp)/10 over poisonable opponents.
    //
    // Both restrict to true opponents via are_teammates (so 1v1 collapses to the
    // single opponent) and read only board cells + team structure, preserving
    // the canonical-view rotational invariance.
    static_assert(Traits::kTensorSize ==
                  Traits::kTensorSizeV1 + Traits::kGroupGSize,
                  "Group G size must account for the full V2 tensor tail");
    static_assert(Traits::kGroupGSize == 2 * Traits::kNumPlayers * kNumColumns,
                  "Group G is 2 features per (player, column)");
    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        const int p = canonical[ci];
        for (int col = 0; col < kNumColumns; ++col) {
            float g0 = 0.0f, g1 = 0.0f;

            // G0 — my committed LS is poisonable while my SS is still open.
            const int ls_p = board.cells[p][col][kRowLS];
            const bool my_ss_open = (board.cells[p][col][kRowSS] == kCellEmpty);
            if (ls_p != kCellEmpty && ls_p > 0 && my_ss_open &&
                !ctx.lower_has_scratch[p][col]) {
                const float window = (30 - std::max(20, ls_p)) / 10.0f;
                if (window > 0.0f) {
                    for (int q = 0; q < Traits::kNumPlayers; ++q) {
                        if (q == p || Traits::are_teammates(p, q)) continue;
                        if (max_placeable_ss<Traits>(q, col, board, ctx) >= ls_p) {
                            g0 = window;  // a capable opponent exists
                            break;
                        }
                    }
                }
            }

            // G1 — I can push an open SS to/above an opponent's committed-low LS.
            const int my_max_ss = max_placeable_ss<Traits>(p, col, board, ctx);
            if (my_max_ss > 0) {
                for (int q = 0; q < Traits::kNumPlayers; ++q) {
                    if (q == p || Traits::are_teammates(p, q)) continue;
                    const int ls_q = board.cells[q][col][kRowLS];
                    const bool opp_ss_open =
                        (board.cells[q][col][kRowSS] == kCellEmpty);
                    if (ls_q != kCellEmpty && ls_q > 0 && opp_ss_open &&
                        !ctx.lower_has_scratch[q][col] && my_max_ss >= ls_q) {
                        g1 = std::max(g1, (30 - std::max(20, ls_q)) / 10.0f);
                    }
                }
            }

            out[idx++] = g0;
            out[idx++] = g1;
        }
    }

    assert(idx == Traits::kTensorSize);
}

// ---------------------------------------------------------------------------
// generate_tensor — canonical-view templated implementation.
// ---------------------------------------------------------------------------
template <typename Traits>
void generate_tensor(const BoardStateT<Traits>& board,
                     const GameContextT<Traits>& ctx,
                     int player, const PrecomputedTables& tables,
                     float* out) {
    PCData pc[Traits::kNumPlayers][6];
    int canonical[Traits::kNumPlayers];
    make_canonical_order<Traits>(player, canonical);

    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        const int p = canonical[ci];
        const int player_filled = count_filled_cells<Traits>(board, p);
        for (int col = 0; col < kNumColumns; ++col) {
            compute_pc_data<Traits>(board, ctx, p, col, player_filled, tables, pc[ci][col]);
        }
    }

    write_tensor_from_pc<Traits>(board, player, ctx, pc, out);
}

// ---------------------------------------------------------------------------
// generate_tensor_batch — apply each candidate placement, regenerate tensor.
// Optimisation: pre-compute per-(canonical, col) base data once, then only
// recompute the (active_canonical_idx, placement.col) entry per request.
// ---------------------------------------------------------------------------
template <typename Traits>
void generate_tensor_batch(const BoardStateT<Traits>& board,
                           const GameContextT<Traits>& ctx,
                           int player,
                           const AfterstateRequest* requests, int request_count,
                           const PrecomputedTables& tables,
                           float* out) {
    int canonical[Traits::kNumPlayers];
    make_canonical_order<Traits>(player, canonical);

    PCData base_pc[Traits::kNumPlayers][6];
    int filled[Traits::kNumPlayers];
    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        const int p = canonical[ci];
        filled[ci] = count_filled_cells<Traits>(board, p);
    }
    // Active player gains one filled cell after the placement.
    filled[0] += 1;

    for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
        const int p = canonical[ci];
        for (int col = 0; col < kNumColumns; ++col) {
            compute_pc_data<Traits>(board, ctx, p, col, filled[ci], tables, base_pc[ci][col]);
        }
    }

    PCData saved_col[Traits::kNumPlayers];

    for (int i = 0; i < request_count; ++i) {
        BoardStateT<Traits> board_clone = board;
        GameContextT<Traits> ctx_clone;
        copy_tensor_context_fields<Traits>(ctx, ctx_clone);

        const AfterstateRequest& req = requests[i];
        const int col = req.placement.column;
        const int row = req.placement.row;
        const int score = req.score;

        // The clone is consumed solely by compute_pc_data, which never reads
        // the legal_all / legal_no_turbo caches.
        apply_placement<Traits>(player, col, row, score, board_clone, ctx_clone,
                                /*update_legal_cache=*/false);

        // Mutate base_pc in place for the affected column, write the tensor,
        // then restore. The placement updates ctx.golden_max[col][row], which
        // influences every player's expected EV in that column, so we have to
        // recompute the affected column for ALL players (not just active).
        for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
            saved_col[ci] = base_pc[ci][col];
        }
        for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
            const int p_other = canonical[ci];
            compute_pc_data<Traits>(board_clone, ctx_clone, p_other, col,
                                    filled[ci], tables, base_pc[ci][col]);
        }

        write_tensor_from_pc<Traits>(board_clone, player, ctx_clone, base_pc,
                                     out + static_cast<ptrdiff_t>(i) * Traits::kTensorSize);

        for (int ci = 0; ci < Traits::kNumPlayers; ++ci) {
            base_pc[ci][col] = saved_col[ci];
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template void generate_tensor<Yams1v1>(const BoardStateT<Yams1v1>&,
                                       const GameContextT<Yams1v1>&,
                                       int, const PrecomputedTables&, float*);
template void generate_tensor<Yams2v2>(const BoardStateT<Yams2v2>&,
                                       const GameContextT<Yams2v2>&,
                                       int, const PrecomputedTables&, float*);
template void generate_tensor_batch<Yams1v1>(const BoardStateT<Yams1v1>&,
                                             const GameContextT<Yams1v1>&,
                                             int,
                                             const AfterstateRequest*, int,
                                             const PrecomputedTables&, float*);
template void generate_tensor_batch<Yams2v2>(const BoardStateT<Yams2v2>&,
                                             const GameContextT<Yams2v2>&,
                                             int,
                                             const AfterstateRequest*, int,
                                             const PrecomputedTables&, float*);
