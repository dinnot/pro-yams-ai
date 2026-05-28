#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "solver/dp_tables.h"

// ---------------------------------------------------------------------------
// Internal index helpers — shared between dp_tables.cc and DP compute units.
// ---------------------------------------------------------------------------

inline std::size_t dp_idx_t1(int T, int v, int sc, int R) {
    return ((static_cast<std::size_t>(T) * kDPNumVariants + v)
                * kDPUpperStates + sc) * kDPUpperRPad + R;
}

inline std::size_t dp_idx_t4(int T, int v, int sc, int S) {
    return ((static_cast<std::size_t>(T) * kDPNumVariants + v)
                * kDPUpperStates + sc) * kDPUpperSumPad + S;
}

inline std::size_t dp_idx_mid(int T, int v, int sc) {
    return (static_cast<std::size_t>(T) * kDPNumVariants + v)
                * kDPMiddleStates + sc;
}

inline std::size_t dp_idx_low(int T, int v, int sc) {
    return (static_cast<std::size_t>(T) * kDPNumVariants + v)
                * kDPLowerStates + sc;
}

// ---------------------------------------------------------------------------
// Middle Section mutual destruction logic — applied when placing in cell c
// (0 = SS, 1 = LS) given the resulting clamped score.
//
// State carries {ss_min, ls_min, ss_cap}; this transitions it when placing
// cell c (0 = SS, 1 = LS) with the resulting clamped score.
//
// Placing SS (c=0): SS becomes filled (-1) and its cap is now irrelevant
//   (cap=0). A scratch forces the partner LS to a forced-scratch (31, since
//   LS raw <= 30 always); a positive SS raises the LS threshold to
//   max(prev_ls, score+1), capped at 31 (impossible-to-beat → forced scratch).
// Placing LS (c=1): LS becomes filled (-1). It bounds SS strictly from above
//   (SS < score). A scratch, or any LS <= 20, makes SS impossible →
//   ss_min=31. A binding 21..29 sets ss_cap=score. An LS >= 30 is
//   non-binding (SS raw maxes at 29) → cap stays 0.
// ---------------------------------------------------------------------------
inline void apply_middle_destruction(int c, int score,
                                     int8_t prev_ss, int8_t prev_ls,
                                     int8_t prev_cap,
                                     int8_t& next_ss, int8_t& next_ls,
                                     int8_t& next_cap) {
    next_ss = prev_ss;
    next_ls = prev_ls;
    next_cap = prev_cap;
    if (c == 0) {
        next_ss = -1;
        next_cap = 0;  // SS filled → its upper-bound cap no longer applies
        if (score == 0 && next_ls != -1) {
            next_ls = 31;  // forced scratch
        } else if (score > 0 && next_ls != -1) {
            int new_ls = std::max(static_cast<int>(next_ls), score + 1);
            next_ls = (new_ls > 30) ? static_cast<int8_t>(31)
                                    : static_cast<int8_t>(new_ls);
        }
    } else if (c == 1) {
        next_ls = -1;
        if (next_ss != -1) {
            if (score == 0 || score <= 20) {
                next_ss = 31;  // LS scratch / LS<=20 ⇒ SS impossible
                next_cap = 0;
            } else if (score <= 29) {
                next_cap = static_cast<int8_t>(score);  // SS must be < score
            } else {
                next_cap = 0;  // LS >= 30: non-binding (SS raw <= 29)
            }
        } else {
            next_cap = 0;  // SS already filled → cap irrelevant
        }
    }
}
