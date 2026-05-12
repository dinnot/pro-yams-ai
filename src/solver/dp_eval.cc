#include "solver/dp_eval.h"

#include <algorithm>

#include "engine/constants.h"

namespace {

constexpr int8_t kVals1s[]  = {-1, 0, 3, 4, 5};
constexpr int8_t kVals2s[]  = {-1, 0, 6, 8, 10};
constexpr int8_t kVals3s[]  = {-1, 0, 12, 15};
constexpr int8_t kVals4s[]  = {-1, 0, 16, 20};
constexpr int8_t kVals5s[]  = {-1, 0, 20, 25};
constexpr int8_t kVals6s[]  = {-1, 0, 24, 30};
constexpr int8_t kValsMid[] = {-1, 0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
constexpr int8_t kValsFH[]  = {-1, 0, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37,
                               38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50};
constexpr int8_t kValsK[]   = {-1, 0, 38, 42, 46, 50, 54};
constexpr int8_t kValsQ[]   = {-1, 0, 50};
constexpr int8_t kValsU8[]  = {-1, 0, 65, 70, 75};
constexpr int8_t kValsY[]   = {-1, 0, 80, 85, 90, 95, 100};

}  // namespace

int8_t snap_gmax(int gmax, const int8_t* vals, int count) {
    if (gmax <= 0) return 0;
    for (int i = 2; i < count; ++i) {
        if (vals[i] >= gmax) return vals[i];
    }
    return vals[count - 1];
}

Variant get_variant(int col) {
    if (col == kColDown)    return Variant::DOWN;
    if (col == kColUp)      return Variant::UP;
    if (col == kColMid)     return Variant::UPDOWN;
    if (col == kColUpDown)  return Variant::UPDOWN;
    if (col == kColTurbo)   return Variant::TURBO;
    return Variant::FREE;
}

void build_Sc(int p, int col, const BoardState& board, const GameContext& ctx,
              int8_t Sc_U[6], int8_t Sc_M[2], int8_t Sc_L[5],
              int& EU, int& EM, int& EL) {
    EU = 0; EM = 0; EL = 0;

    const int8_t* u_vals[] = {kVals1s, kVals2s, kVals3s, kVals4s, kVals5s, kVals6s};
    const int u_counts[] = {5, 5, 4, 4, 4, 4};
    for (int i = 0; i < 6; ++i) {
        if (board.cells[p][col][i] != kCellEmpty) {
            Sc_U[i] = -1;
        } else {
            Sc_U[i] = snap_gmax(ctx.golden_max[col][i], u_vals[i], u_counts[i]);
            ++EU;
        }
    }

    if (board.cells[p][col][kRowSS] != kCellEmpty) {
        Sc_M[0] = -1;
    } else {
        Sc_M[0] = snap_gmax(ctx.golden_max[col][kRowSS], kValsMid, 13);
        if (ctx.ls_scratched[p][col]) Sc_M[0] = 31;
        ++EM;
    }
    if (board.cells[p][col][kRowLS] != kCellEmpty) {
        Sc_M[1] = -1;
    } else {
        Sc_M[1] = snap_gmax(ctx.golden_max[col][kRowLS], kValsMid, 13);
        if (ctx.ss_scratched[p][col]) Sc_M[1] = 31;
        ++EM;
    }

    const int8_t* l_vals[] = {kValsFH, kValsK, kValsQ, kValsU8, kValsY};
    const int l_counts[] = {24, 7, 3, 5, 7};
    for (int i = 0; i < 5; ++i) {
        if (board.cells[p][col][8 + i] != kCellEmpty) {
            Sc_L[i] = -1;
        } else {
            Sc_L[i] = snap_gmax(ctx.golden_max[col][8 + i], l_vals[i], l_counts[i]);
            ++EL;
        }
    }
}

void apportion_turns(int T, int EU, int EM, int EL,
                     int& TU, int& TM, int& TL) {
    int Ecol = EU + EM + EL;
    if (Ecol <= 0) {
        TU = TM = TL = 0;
        return;
    }
    TU = (T * EU) / Ecol;
    TM = (T * EM) / Ecol;
    TL = T - TU - TM;
    if (TL < 0) TL = 0;
}

float get_E_raw(int p, int col, int T, const BoardState& board,
                const GameContext& ctx, const DPTables& dp) {
    if (dp.dp_t1.empty()) return 0.0f;

    int8_t Sc_U[6], Sc_M[2], Sc_L[5];
    int EU, EM, EL;
    build_Sc(p, col, board, ctx, Sc_U, Sc_M, Sc_L, EU, EM, EL);
    int TU, TM, TL;
    apportion_turns(T, EU, EM, EL, TU, TM, TL);

    Variant v = get_variant(col);
    float eu = get_upper_ev(dp, v, Sc_U, TU, ctx.upper_sum[p][col]);
    float em = get_middle_ev(dp, v, Sc_M, TM);
    float el = get_lower_ev(dp, v, Sc_L, TL);

    int filled_score = 0;
    for (int r = 6; r <= 12; ++r) {
        int8_t cv = board.cells[p][col][r];
        if (cv > 0) filled_score += cv;
    }
    return eu + em + el + static_cast<float>(filled_score);
}

float get_P_clean(int p, int col, int T, const BoardState& board,
                  const GameContext& ctx, const DPTables& dp) {
    if (dp.dp_t1.empty()) return 0.0f;
    if (ctx.lower_has_scratch[p][col]) return 0.0f;

    int8_t Sc_U[6], Sc_M[2], Sc_L[5];
    int EU, EM, EL;
    build_Sc(p, col, board, ctx, Sc_U, Sc_M, Sc_L, EU, EM, EL);
    int TU, TM, TL;
    apportion_turns(T, EU, EM, EL, TU, TM, TL);

    Variant v = get_variant(col);
    int cur_upper = ctx.upper_sum[p][col];
    int R_clean = std::max(0, 60 - cur_upper);

    float p_up_60 = get_upper_prob(dp, v, Sc_U, TU, R_clean);
    float p_mid   = get_middle_prob(dp, v, Sc_M, TM);
    float p_low   = get_lower_prob(dp, v, Sc_L, TL);

    float pc = p_up_60 * p_mid * p_low;
    if (pc < 0.0f) pc = 0.0f;
    if (pc > 1.0f) pc = 1.0f;
    return pc;
}

float get_E_raw_var(int p, int col, int T, const BoardState& board,
                    const GameContext& ctx, const DPTables& dp) {
    if (dp.dp_t1.empty()) return 0.0f;

    int8_t Sc_U[6], Sc_M[2], Sc_L[5];
    int EU, EM, EL;
    build_Sc(p, col, board, ctx, Sc_U, Sc_M, Sc_L, EU, EM, EL);

    int TU, TM, TL;
    apportion_turns(T, EU, EM, EL, TU, TM, TL);
    Variant v = get_variant(col);
    
    float eu = get_upper_ev(dp, v, Sc_U, TU, ctx.upper_sum[p][col]);
    float eu_sq = get_upper_ev_sq(dp, v, Sc_U, TU, ctx.upper_sum[p][col]);
    float var_u = (eu > -1e10f) ? std::max(0.0f, eu_sq - eu * eu) : 0.0f; 

    float em = get_middle_ev(dp, v, Sc_M, TM);
    float em_sq = get_middle_ev_sq(dp, v, Sc_M, TM);
    float var_m = (em > -1e10f) ? std::max(0.0f, em_sq - em * em) : 0.0f;

    float el = get_lower_ev(dp, v, Sc_L, TL);
    float el_sq = get_lower_ev_sq(dp, v, Sc_L, TL);
    float var_l = (el > -1e10f) ? std::max(0.0f, el_sq - el * el) : 0.0f;

    return var_u + var_m + var_l;
}
