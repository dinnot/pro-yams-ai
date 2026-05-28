#include "solver/dp_tables.h"
#include "solver/dp_tables_internal.h"
#include "solver/precomputed_tables.h"
#include "engine/constants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace {

constexpr float kNegInf = -1.0e30f;

// Per-thread scratch buffers, sized for the largest inner dim (Sum=112).
// All buffers are 32-byte-aligned for AVX-256 SIMD.
struct UpperBuffers {
    alignas(32) float V0    [kNumDiceStates * kDPUpperSumPad];
    alignas(32) float V0_sq [kNumDiceStates * kDPUpperSumPad];
    alignas(32) float V_curr[kNumDiceStates * kDPUpperSumPad];
    alignas(32) float V_curr_sq[kNumDiceStates * kDPUpperSumPad];
    alignas(32) float V_next[kNumDiceStates * kDPUpperSumPad];
    alignas(32) float V_next_sq[kNumDiceStates * kDPUpperSumPad];
    alignas(32) float EV_held[462 * kDPUpperSumPad];   // 462 = num_held_configs
    alignas(32) float EV_held_sq[462 * kDPUpperSumPad];
    alignas(32) float skip   [kDPUpperSumPad];
    alignas(32) float skip_sq[kDPUpperSumPad];
};

// Cell-to-row maps.
constexpr int kUpperRow[6]  = {0, 1, 2, 3, 4, 5};                     // 1s..6s
constexpr int kMiddleRow[2] = {kRowSS, kRowLS};                       // 6, 7
constexpr int kLowerRow[5]  = {kRowFH, kRowK, kRowSTR, kRowU8, kRowY};// 8..12

}  // namespace

// ===========================================================================
// Upper Section DP (Tasks 1 & 4)
// ===========================================================================
void compute_upper_dp(DPTables& dp, const PrecomputedTables& tables) {
    // Pre-decode every upper sc — done once.
    std::vector<std::array<int8_t, 6>> Sc_decoded(kDPUpperStates);
    std::vector<int> ec_count(kDPUpperStates);
    for (int sc = 0; sc < kDPUpperStates; ++sc) {
        decode_upper(sc, Sc_decoded[sc].data());
        int n = 0;
        for (int i = 0; i < 6; ++i) if (Sc_decoded[sc][i] != -1) ++n;
        ec_count[sc] = n;
    }

    // ----------------------------------------------------------------------
    // T = 0 base cases
    //   Task 1: dp_t1[0][v][sc][R] = (R==0) ? 1 : 0   (R=0 trivially achieved)
    //   Task 4: dp_t4[0][v][sc][S] = (ec==0) ? S+bonus(S) : -inf  (S<=105)
    //           padded region (R>100, S>105) initialised to 0 for cleanliness
    // ----------------------------------------------------------------------
    for (int sc = 0; sc < kDPUpperStates; ++sc) {
        int ec = ec_count[sc];
        for (int v = 0; v < kDPNumVariants; ++v) {
            for (int R = 0; R < kDPUpperRPad; ++R) {
                float val = (R == 0) ? 1.0f : 0.0f;
                dp.dp_t1[dp_idx_t1(0, v, sc, R)] = val;
            }
            for (int S = 0; S < kDPUpperSumPad; ++S) {
                float val, val_sq;
                if (S > kDPUpperSumMax) {
                    val = 0.0f; val_sq = 0.0f;
                } else if (ec == 0) {
                    float pts = static_cast<float>(S + upper_bonus(S));
                    val = pts; val_sq = pts * pts;
                } else {
                    val = kNegInf; val_sq = 0.0f;
                }
                dp.dp_t4[dp_idx_t4(0, v, sc, S)] = val;
                dp.dp_t5[dp_idx_t4(0, v, sc, S)] = val_sq;
            }
        }
    }

    // ----------------------------------------------------------------------
    // Recurrence T = 1..78
    // ----------------------------------------------------------------------
    for (int T = 1; T < kDPNumTurns; ++T) {
        for (int v = 0; v < kDPNumVariants; ++v) {
            const int max_rolls =
                (static_cast<Variant>(v) == Variant::TURBO) ? 1 : 2;

            #pragma omp parallel
            {
                auto buf = std::make_unique<UpperBuffers>();

                #pragma omp for schedule(dynamic, 32)
                for (int sc = 0; sc < kDPUpperStates; ++sc) {
                    const int8_t* Sc = Sc_decoded[sc].data();
                    const int ec = ec_count[sc];

                    // Infeasible state: not enough turns to fill remaining cells.
                    if (T < ec) {
                        for (int R = 0; R < kDPUpperRPad; ++R)
                            dp.dp_t1[dp_idx_t1(T, v, sc, R)] = (R == 0) ? 1.0f : 0.0f;
                        for (int S = 0; S < kDPUpperSumPad; ++S) {
                            dp.dp_t4[dp_idx_t4(T, v, sc, S)] =
                                (S > kDPUpperSumMax) ? 0.0f : kNegInf;
                            dp.dp_t5[dp_idx_t4(T, v, sc, S)] = 0.0f;
                        }
                        continue;
                    }

                    std::vector<int> valid =
                        get_valid_cells(static_cast<Variant>(v), Sc, 6);
                    const int n_valid = static_cast<int>(valid.size());

                    int next_sc_per[6] = {0};
                    for (int c : valid) {
                        int8_t next_Sc[6];
                        std::memcpy(next_Sc, Sc, 6);
                        next_Sc[c] = -1;
                        next_sc_per[c] = encode_upper(next_Sc);
                    }

                    // ====================================================
                    // TASK 1 (R-dim, probability of reaching residual R)
                    // ====================================================
                    {
                        constexpr int RP = kDPUpperRPad;

                        // skip[R] = (T > ec) ? prev[same sc][R] : 0
                        const bool can_skip = (T > ec);
                        for (int R = 0; R < RP; ++R) {
                            buf->skip[R] = can_skip
                                ? dp.dp_t1[dp_idx_t1(T - 1, v, sc, R)]
                                : 0.0f;
                        }

                        // Build V0[d][R].
                        for (int d = 0; d < kNumDiceStates; ++d) {
                            float* V0d = &buf->V0[d * RP];
                            std::memcpy(V0d, buf->skip, RP * sizeof(float));

                            for (int idx = 0; idx < n_valid; ++idx) {
                                int c = valid[idx];
                                int row = kUpperRow[c];
                                int raw = tables.score_tables.dice_score[d][row];
                                int score = (raw >= Sc[c]) ? raw : 0;
                                int next_sc_c = next_sc_per[c];
                                const float* prev =
                                    &dp.dp_t1[dp_idx_t1(T - 1, v, next_sc_c, 0)];

                                #pragma omp simd
                                for (int R = 0; R < RP; ++R) {
                                    int next_R = R - score;
                                    if (next_R < 0) next_R = 0;
                                    float val = prev[next_R];
                                    if (val > V0d[R]) V0d[R] = val;
                                }
                            }
                        }

                        // V_curr := V0
                        std::memcpy(buf->V_curr, buf->V0,
                            kNumDiceStates * RP * sizeof(float));

                        // Rollback iterations.
                        for (int r = 0; r < max_rolls; ++r) {
                            std::memset(buf->EV_held, 0,
                                tables.num_held_configs * RP * sizeof(float));

                            for (int h = 0; h < tables.num_held_configs; ++h) {
                                int n_tr = tables.transition_count[h];
                                const Transition* tr = &tables.all_transitions[
                                    tables.transition_offset[h]];
                                float* EVh = &buf->EV_held[h * RP];
                                for (int k = 0; k < n_tr; ++k) {
                                    float p = static_cast<float>(tr[k].probability);
                                    int sid = tr[k].target_state_id;
                                    const float* Vc = &buf->V_curr[sid * RP];
                                    #pragma omp simd
                                    for (int R = 0; R < RP; ++R) {
                                        EVh[R] += p * Vc[R];
                                    }
                                }
                            }

                            for (int d = 0; d < kNumDiceStates; ++d) {
                                float* Vn = &buf->V_next[d * RP];
                                const float* V0d = &buf->V0[d * RP];
                                std::memcpy(Vn, V0d, RP * sizeof(float));
                                for (int m = 0; m < kNumHoldMasks; ++m) {
                                    int h = tables.moves[d][m];
                                    const float* EVh = &buf->EV_held[h * RP];
                                    #pragma omp simd
                                    for (int R = 0; R < RP; ++R) {
                                        if (EVh[R] > Vn[R]) Vn[R] = EVh[R];
                                    }
                                }
                            }

                            std::memcpy(buf->V_curr, buf->V_next,
                                kNumDiceStates * RP * sizeof(float));
                        }

                        // Final pre-roll EV from h_empty (= moves[0][0]).
                        const int h_empty = tables.moves[0][0];
                        const int n_tr = tables.transition_count[h_empty];
                        const Transition* tr = &tables.all_transitions[
                            tables.transition_offset[h_empty]];
                        float* dst = &dp.dp_t1[dp_idx_t1(T, v, sc, 0)];
                        std::memset(dst, 0, RP * sizeof(float));
                        for (int k = 0; k < n_tr; ++k) {
                            float p = static_cast<float>(tr[k].probability);
                            int sid = tr[k].target_state_id;
                            const float* Vc = &buf->V_curr[sid * RP];
                            #pragma omp simd
                            for (int R = 0; R < RP; ++R) {
                                dst[R] += p * Vc[R];
                            }
                        }
                    }

                    // ====================================================
                    // TASK 4 (Sum-dim, expected total points)
                    // ====================================================
                    {
                        constexpr int SP = kDPUpperSumPad;

                        const bool can_skip = (T > ec);
                        for (int S = 0; S < SP; ++S) {
                            if (can_skip) {
                                buf->skip[S]    = dp.dp_t4[dp_idx_t4(T - 1, v, sc, S)];
                                buf->skip_sq[S] = dp.dp_t5[dp_idx_t4(T - 1, v, sc, S)];
                            } else {
                                buf->skip[S]    = (S > kDPUpperSumMax) ? 0.0f : kNegInf;
                                buf->skip_sq[S] = 0.0f;
                            }
                        }

                        for (int d = 0; d < kNumDiceStates; ++d) {
                            float* V0d    = &buf->V0[d * SP];
                            float* V0d_sq = &buf->V0_sq[d * SP];
                            std::memcpy(V0d,    buf->skip,    SP * sizeof(float));
                            std::memcpy(V0d_sq, buf->skip_sq, SP * sizeof(float));

                            for (int idx = 0; idx < n_valid; ++idx) {
                                int c = valid[idx];
                                int row = kUpperRow[c];
                                int raw = tables.score_tables.dice_score[d][row];
                                int score = (raw >= Sc[c]) ? raw : 0;
                                int next_sc_c = next_sc_per[c];
                                const float* prev =
                                    &dp.dp_t4[dp_idx_t4(T - 1, v, next_sc_c, 0)];
                                const float* prev_sq =
                                    &dp.dp_t5[dp_idx_t4(T - 1, v, next_sc_c, 0)];

                                #pragma omp simd
                                for (int S = 0; S < SP; ++S) {
                                    int next_S = S + score;
                                    if (next_S > kDPUpperSumMax) next_S = kDPUpperSumMax;
                                    float val = prev[next_S];
                                    float val_sq = prev_sq[next_S];
                                    
                                    constexpr float kEps = 1e-4f;
                                    if (val > V0d[S] + kEps) {
                                        V0d[S] = val; V0d_sq[S] = val_sq;
                                    } else if (std::abs(val - V0d[S]) <= kEps) {
                                        if (val_sq < V0d_sq[S]) { V0d[S] = val; V0d_sq[S] = val_sq; }
                                    }
                                }
                            }
                        }

                        std::memcpy(buf->V_curr,    buf->V0,    kNumDiceStates * SP * sizeof(float));
                        std::memcpy(buf->V_curr_sq, buf->V0_sq, kNumDiceStates * SP * sizeof(float));

                        for (int r = 0; r < max_rolls; ++r) {
                            std::memset(buf->EV_held,    0, tables.num_held_configs * SP * sizeof(float));
                            std::memset(buf->EV_held_sq, 0, tables.num_held_configs * SP * sizeof(float));

                            for (int h = 0; h < tables.num_held_configs; ++h) {
                                int n_tr = tables.transition_count[h];
                                const Transition* tr = &tables.all_transitions[
                                    tables.transition_offset[h]];
                                float* EVh    = &buf->EV_held[h * SP];
                                float* EVh_sq = &buf->EV_held_sq[h * SP];
                                for (int k = 0; k < n_tr; ++k) {
                                    float p = static_cast<float>(tr[k].probability);
                                    int sid = tr[k].target_state_id;
                                    const float* Vc    = &buf->V_curr[sid * SP];
                                    const float* Vc_sq = &buf->V_curr_sq[sid * SP];
                                    #pragma omp simd
                                    for (int S = 0; S < SP; ++S) {
                                        EVh[S]    += p * Vc[S];
                                        EVh_sq[S] += p * Vc_sq[S];
                                    }
                                }
                            }

                            for (int d = 0; d < kNumDiceStates; ++d) {
                                float* Vn     = &buf->V_next[d * SP];
                                float* Vn_sq  = &buf->V_next_sq[d * SP];
                                const float* V0d    = &buf->V0[d * SP];
                                const float* V0d_sq = &buf->V0_sq[d * SP];
                                std::memcpy(Vn,    V0d,    SP * sizeof(float));
                                std::memcpy(Vn_sq, V0d_sq, SP * sizeof(float));
                                for (int m = 0; m < kNumHoldMasks; ++m) {
                                    int h = tables.moves[d][m];
                                    const float* EVh    = &buf->EV_held[h * SP];
                                    const float* EVh_sq = &buf->EV_held_sq[h * SP];
                                    #pragma omp simd
                                    for (int S = 0; S < SP; ++S) {
                                        constexpr float kEps = 1e-5f;
                                        if (EVh[S] > Vn[S] + kEps) {
                                            Vn[S] = EVh[S]; Vn_sq[S] = EVh_sq[S];
                                        } else if (std::abs(EVh[S] - Vn[S]) <= kEps) {
                                            if (EVh_sq[S] < Vn_sq[S]) { Vn[S] = EVh[S]; Vn_sq[S] = EVh_sq[S]; }
                                        }
                                    }
                                }
                            }

                            std::memcpy(buf->V_curr,    buf->V_next,    kNumDiceStates * SP * sizeof(float));
                            std::memcpy(buf->V_curr_sq, buf->V_next_sq, kNumDiceStates * SP * sizeof(float));
                        }

                        const int h_empty = tables.moves[0][0];
                        const int n_tr = tables.transition_count[h_empty];
                        const Transition* tr = &tables.all_transitions[
                            tables.transition_offset[h_empty]];
                        float* dst    = &dp.dp_t4[dp_idx_t4(T, v, sc, 0)];
                        float* dst_sq = &dp.dp_t5[dp_idx_t4(T, v, sc, 0)];
                        std::memset(dst,    0, SP * sizeof(float));
                        std::memset(dst_sq, 0, SP * sizeof(float));
                        for (int k = 0; k < n_tr; ++k) {
                            float p = static_cast<float>(tr[k].probability);
                            int sid = tr[k].target_state_id;
                            const float* Vc    = &buf->V_curr[sid * SP];
                            const float* Vc_sq = &buf->V_curr_sq[sid * SP];
                            #pragma omp simd
                            for (int S = 0; S < SP; ++S) {
                                dst[S]    += p * Vc[S];
                                dst_sq[S] += p * Vc_sq[S];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Scalar rollback helper used by Middle / Lower DPs.
// V_curr/V_next/EV_held are 1-D arrays sized by dice/held count.
// ===========================================================================
namespace {

struct ScalarBuffers {
    alignas(32) float V0_prob   [kNumDiceStates];
    alignas(32) float V_curr_prob[kNumDiceStates];
    alignas(32) float V_next_prob[kNumDiceStates];
    alignas(32) float V0_ev     [kNumDiceStates];
    alignas(32) float V_curr_ev [kNumDiceStates];
    alignas(32) float V_next_ev [kNumDiceStates];
    alignas(32) float V0_ev_sq  [kNumDiceStates];
    alignas(32) float V_curr_ev_sq[kNumDiceStates];
    alignas(32) float V_next_ev_sq[kNumDiceStates];
    alignas(32) float EV_held_prob[462];
    alignas(32) float EV_held_ev  [462];
    alignas(32) float EV_held_ev_sq[462];
};

inline void scalar_rollback(ScalarBuffers& buf,
                            const PrecomputedTables& tables,
                            int max_rolls,
                            float& out_prob, float& out_ev, float& out_ev_sq) {
    std::memcpy(buf.V_curr_prob, buf.V0_prob, kNumDiceStates * sizeof(float));
    std::memcpy(buf.V_curr_ev,   buf.V0_ev,   kNumDiceStates * sizeof(float));
    std::memcpy(buf.V_curr_ev_sq, buf.V0_ev_sq, kNumDiceStates * sizeof(float));

    for (int r = 0; r < max_rolls; ++r) {
        // EV_held for each held config: weighted sum over transitions.
        std::memset(buf.EV_held_prob, 0, tables.num_held_configs * sizeof(float));
        std::memset(buf.EV_held_ev,   0, tables.num_held_configs * sizeof(float));
        std::memset(buf.EV_held_ev_sq, 0, tables.num_held_configs * sizeof(float));
        for (int h = 0; h < tables.num_held_configs; ++h) {
            int n_tr = tables.transition_count[h];
            const Transition* tr = &tables.all_transitions[
                tables.transition_offset[h]];
            float sum_p = 0.0f, sum_e = 0.0f, sum_esq = 0.0f;
            for (int k = 0; k < n_tr; ++k) {
                float p = static_cast<float>(tr[k].probability);
                int sid = tr[k].target_state_id;
                sum_p   += p * buf.V_curr_prob[sid];
                sum_e   += p * buf.V_curr_ev  [sid];
                sum_esq += p * buf.V_curr_ev_sq[sid];
            }
            buf.EV_held_prob[h]  = sum_p;
            buf.EV_held_ev  [h]  = sum_e;
            buf.EV_held_ev_sq[h] = sum_esq;
        }

        // V_next[d] = max(V0[d], max over m of EV_held[moves[d][m]])
        for (int d = 0; d < kNumDiceStates; ++d) {
            float bp = buf.V0_prob[d];
            float be = buf.V0_ev  [d];
            float besq = buf.V0_ev_sq[d];
            for (int m = 0; m < kNumHoldMasks; ++m) {
                int h = tables.moves[d][m];
                float p = buf.EV_held_prob[h];
                float e = buf.EV_held_ev  [h];
                float esq = buf.EV_held_ev_sq[h];
                if (p > bp) bp = p;

                constexpr float kEps = 1e-5f;
                if (e > be + kEps) {
                    be = e; besq = esq;
                } else if (std::abs(e - be) <= kEps) {
                    if (esq < besq) { be = e; besq = esq; } // Tie-breaker!
                }
            }
            buf.V_next_prob[d] = bp;
            buf.V_next_ev  [d] = be;
            buf.V_next_ev_sq[d] = besq;
        }

        std::memcpy(buf.V_curr_prob, buf.V_next_prob, kNumDiceStates * sizeof(float));
        std::memcpy(buf.V_curr_ev,   buf.V_next_ev,   kNumDiceStates * sizeof(float));
        std::memcpy(buf.V_curr_ev_sq, buf.V_next_ev_sq, kNumDiceStates * sizeof(float));
    }

    // Final pre-roll EV from h_empty (= moves[0][0]).
    const int h_empty = tables.moves[0][0];
    const int n_tr = tables.transition_count[h_empty];
    const Transition* tr = &tables.all_transitions[
        tables.transition_offset[h_empty]];
    float sum_p = 0.0f, sum_e = 0.0f, sum_esq = 0.0f;
    for (int k = 0; k < n_tr; ++k) {
        float p = static_cast<float>(tr[k].probability);
        int sid = tr[k].target_state_id;
        sum_p   += p * buf.V_curr_prob[sid];
        sum_e   += p * buf.V_curr_ev  [sid];
        sum_esq += p * buf.V_curr_ev_sq[sid];
    }
    out_prob = sum_p;
    out_ev   = sum_e;
    out_ev_sq = sum_esq;
}

}  // namespace

// ===========================================================================
// Middle Section DP (Tasks 2 & 5)
// ===========================================================================
void compute_middle_dp(DPTables& dp, const PrecomputedTables& tables) {
    std::vector<std::array<int8_t, 3>> Sc_decoded(kDPMiddleStates);
    std::vector<int> ec_count(kDPMiddleStates);
    for (int sc = 0; sc < kDPMiddleStates; ++sc) {
        decode_middle(sc, Sc_decoded[sc].data());
        int n = 0;
        for (int i = 0; i < 2; ++i) if (Sc_decoded[sc][i] != -1) ++n;
        ec_count[sc] = n;  // ss_cap (Sc[2]) is metadata, not a playable cell
    }

    // T = 0 base.
    for (int sc = 0; sc < kDPMiddleStates; ++sc) {
        int ec = ec_count[sc];
        for (int v = 0; v < kDPNumVariants; ++v) {
            DPVal& cell = dp.dp_mid[dp_idx_mid(0, v, sc)];
            if (ec == 0) cell = DPVal{1.0f, 0.0f, 0.0f};
            else         cell = DPVal{0.0f, kNegInf, 0.0f};
        }
    }

    for (int T = 1; T < kDPNumTurns; ++T) {
        for (int v = 0; v < kDPNumVariants; ++v) {
            const int max_rolls =
                (static_cast<Variant>(v) == Variant::TURBO) ? 1 : 2;

            #pragma omp parallel
            {
                ScalarBuffers buf;

                #pragma omp for schedule(dynamic, 16)
                for (int sc = 0; sc < kDPMiddleStates; ++sc) {
                    const int8_t* Sc = Sc_decoded[sc].data();
                    const int ec = ec_count[sc];

                    if (T < ec) {
                        dp.dp_mid[dp_idx_mid(T, v, sc)] = DPVal{0.0f, kNegInf, 0.0f};
                        continue;
                    }

                    std::vector<int> valid =
                        get_valid_cells(static_cast<Variant>(v), Sc, 2);
                    const int n_valid = static_cast<int>(valid.size());

                    const bool can_skip = (T > ec);
                    const DPVal skip_val = can_skip
                        ? dp.dp_mid[dp_idx_mid(T - 1, v, sc)]
                        : DPVal{0.0f, kNegInf, 0.0f};

                    for (int d = 0; d < kNumDiceStates; ++d) {
                        float bp = skip_val.prob_no_scratch;
                        float be = skip_val.expected_pts;
                        float besq = skip_val.expected_pts_sq;

                        for (int idx = 0; idx < n_valid; ++idx) {
                            int c = valid[idx];
                            int row = kMiddleRow[c];
                            int raw = tables.score_tables.dice_score[d][row];
                            int score;
                            if (c == 0) {
                                // SS: clear the golden min AND stay strictly
                                // below the LS cap (Sc[2]; 0 = no cap).
                                const int8_t cap = Sc[2];
                                bool ok = (raw >= Sc[0]) &&
                                          (cap == 0 || raw < cap);
                                score = ok ? raw : 0;
                            } else {
                                score = (raw >= Sc[1]) ? raw : 0;
                            }

                            int8_t next_ss, next_ls, next_cap;
                            apply_middle_destruction(c, score, Sc[0], Sc[1],
                                                     Sc[2], next_ss, next_ls,
                                                     next_cap);
                            int next_sc_dc =
                                encode_middle(next_ss, next_ls, next_cap);
                            const DPVal& prev =
                                dp.dp_mid[dp_idx_mid(T - 1, v, next_sc_dc)];

                            float p_contrib = (score == 0) ? 0.0f
                                                           : prev.prob_no_scratch;
                            float e_contrib = static_cast<float>(score)
                                              + prev.expected_pts;
                            float esq_contrib = static_cast<float>(score * score)
                                                + 2.0f * static_cast<float>(score) * prev.expected_pts
                                                + prev.expected_pts_sq;

                            if (p_contrib > bp) bp = p_contrib;
                            
                            constexpr float kEps = 1e-5f;
                            if (e_contrib > be + kEps) {
                                be = e_contrib; besq = esq_contrib;
                            } else if (std::abs(e_contrib - be) <= kEps) {
                                if (esq_contrib < besq) { be = e_contrib; besq = esq_contrib; }
                            }
                        }

                        buf.V0_prob[d]   = bp;
                        buf.V0_ev  [d]   = be;
                        buf.V0_ev_sq [d] = besq;
                    }

                    float final_p, final_e, final_esq;
                    scalar_rollback(buf, tables, max_rolls, final_p, final_e, final_esq);
                    dp.dp_mid[dp_idx_mid(T, v, sc)] = DPVal{final_p, final_e, final_esq};
                }
            }
        }
    }
}

// ===========================================================================
// Lower Section DP (Tasks 3 & 6)
// No mutual destruction; standard Golden Rule + clear placement.
// ===========================================================================
void compute_lower_dp(DPTables& dp, const PrecomputedTables& tables) {
    std::vector<std::array<int8_t, 5>> Sc_decoded(kDPLowerStates);
    std::vector<int> ec_count(kDPLowerStates);
    for (int sc = 0; sc < kDPLowerStates; ++sc) {
        decode_lower(sc, Sc_decoded[sc].data());
        int n = 0;
        for (int i = 0; i < 5; ++i) if (Sc_decoded[sc][i] != -1) ++n;
        ec_count[sc] = n;
    }

    // T = 0 base.
    for (int sc = 0; sc < kDPLowerStates; ++sc) {
        int ec = ec_count[sc];
        for (int v = 0; v < kDPNumVariants; ++v) {
            DPVal& cell = dp.dp_low[dp_idx_low(0, v, sc)];
            if (ec == 0) cell = DPVal{1.0f, 0.0f, 0.0f};
            else         cell = DPVal{0.0f, kNegInf, 0.0f};
        }
    }

    for (int T = 1; T < kDPNumTurns; ++T) {
        for (int v = 0; v < kDPNumVariants; ++v) {
            const int max_rolls =
                (static_cast<Variant>(v) == Variant::TURBO) ? 1 : 2;

            #pragma omp parallel
            {
                ScalarBuffers buf;

                #pragma omp for schedule(dynamic, 64)
                for (int sc = 0; sc < kDPLowerStates; ++sc) {
                    const int8_t* Sc = Sc_decoded[sc].data();
                    const int ec = ec_count[sc];

                    if (T < ec) {
                        dp.dp_low[dp_idx_low(T, v, sc)] = DPVal{0.0f, kNegInf, 0.0f};
                        continue;
                    }

                    std::vector<int> valid =
                        get_valid_cells(static_cast<Variant>(v), Sc, 5);
                    const int n_valid = static_cast<int>(valid.size());

                    int next_sc_per[5] = {0};
                    int8_t next_Sc[5];
                    for (int c : valid) {
                        std::memcpy(next_Sc, Sc, 5);
                        next_Sc[c] = -1;
                        next_sc_per[c] = encode_lower(next_Sc);
                    }

                    const bool can_skip = (T > ec);
                    const DPVal skip_val = can_skip
                        ? dp.dp_low[dp_idx_low(T - 1, v, sc)]
                        : DPVal{0.0f, kNegInf, 0.0f};

                    for (int d = 0; d < kNumDiceStates; ++d) {
                        float bp = skip_val.prob_no_scratch;
                        float be = skip_val.expected_pts;
                        float besq = skip_val.expected_pts_sq;

                        for (int idx = 0; idx < n_valid; ++idx) {
                            int c = valid[idx];
                            int row = kLowerRow[c];
                            int raw = tables.score_tables.dice_score[d][row];
                            int score = (raw >= Sc[c]) ? raw : 0;
                            int next_sc_c = next_sc_per[c];
                            const DPVal& prev =
                                dp.dp_low[dp_idx_low(T - 1, v, next_sc_c)];

                            float p_contrib = (score == 0) ? 0.0f
                                                           : prev.prob_no_scratch;
                            float e_contrib = static_cast<float>(score)
                                              + prev.expected_pts;
                            float esq_contrib = static_cast<float>(score * score)
                                                + 2.0f * static_cast<float>(score) * prev.expected_pts
                                                + prev.expected_pts_sq;

                            if (p_contrib > bp) bp = p_contrib;
                            
                            constexpr float kEps = 1e-5f;
                            if (e_contrib > be + kEps) {
                                be = e_contrib; besq = esq_contrib;
                            } else if (std::abs(e_contrib - be) <= kEps) {
                                if (esq_contrib < besq) { be = e_contrib; besq = esq_contrib; }
                            }
                        }

                        buf.V0_prob[d]   = bp;
                        buf.V0_ev  [d]   = be;
                        buf.V0_ev_sq [d] = besq;
                    }

                    float final_p, final_e, final_esq;
                    scalar_rollback(buf, tables, max_rolls, final_p, final_e, final_esq);
                    dp.dp_low[dp_idx_low(T, v, sc)] = DPVal{final_p, final_e, final_esq};
                }
            }
        }
    }
}

