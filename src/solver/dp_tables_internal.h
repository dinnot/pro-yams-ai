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
// SS scratch makes the partner LS a forced-scratch (constraint = 31, since
// raw <= 30 always); LS scratch likewise marks SS forced-scratch. The cell
// stays empty (must still be played as its own turn).
// SS positive forces LS threshold to max(prev_ls, score+1), capped at 31
// (impossible-to-beat → forced scratch).
// LS positive forces SS to a forced-scratch when SS's lower bound already
// reaches the placed LS value (no legal SS sum < LS remains). A still-open
// band below LS cannot be represented in the min-threshold state, so SS EV is
// mildly overestimated there.
// ---------------------------------------------------------------------------
inline void apply_middle_destruction(int c, int score,
                                     int8_t prev_ss, int8_t prev_ls,
                                     int8_t& next_ss, int8_t& next_ls) {
    next_ss = prev_ss;
    next_ls = prev_ls;
    if (c == 0) {
        next_ss = -1;
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
            if (score == 0) {
                next_ss = 31;  // LS scratch forces SS scratch
            } else {
                // SS must stay strictly below the LS just placed. With SS's
                // natural floor of 20, no legal SS sum remains once the SS
                // lower bound reaches that value → forced scratch.
                int ss_floor = (next_ss == 0) ? 20 : static_cast<int>(next_ss);
                if (ss_floor >= score) next_ss = 31;
            }
        }
    }
}
