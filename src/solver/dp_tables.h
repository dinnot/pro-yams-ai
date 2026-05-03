#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "engine/constants.h"

struct PrecomputedTables;

// ---------------------------------------------------------------------------
// DP Tables — six precomputed Dynamic Programming tables for Pro Yams.
//
// Tasks:
//   1. Upper section: probability of reaching cumulative target R
//   4. Upper section: expected total points (incl. progressive bonus)
//   2. Middle section: probability of no-scratch
//   5. Middle section: expected points
//   3. Lower section: probability of no-scratch
//   6. Lower section: expected points
//
// Indexed by (T, variant, section_state[, R or Sum]).
//   T:        0..78 (turns remaining)
//   variant:  one of FREE, TURBO, DOWN, UP, UPDOWN  (MID column is queried as
//             UPDOWN at runtime)
//
// Tables are persisted to disk after first computation (~2 GB on disk) and
// loaded thereafter.
// ---------------------------------------------------------------------------

constexpr int kDPNumTurns        = 79;     // T = 0..78
constexpr int kDPNumVariants     = 5;
constexpr int kDPUpperStates     = 6400;
constexpr int kDPMiddleStates    = 169;    // 13×13 (incl. 31 = forced scratch)
constexpr int kDPLowerStates     = 17640;
constexpr int kDPUpperRMax       = 100;
constexpr int kDPUpperSumMax     = 105;
constexpr int kDPUpperRPad       = 104;    // R dim padded for SIMD (mult of 8)
constexpr int kDPUpperSumPad     = 112;    // Sum dim padded for SIMD

constexpr int kDPUpperCells      = 6;
constexpr int kDPMiddleCells     = 2;
constexpr int kDPLowerCells      = 5;

enum class Variant : int {
    FREE   = 0,
    TURBO  = 1,
    DOWN   = 2,
    UP     = 3,
    UPDOWN = 4,
};

struct DPVal {
    float prob_no_scratch;
    float expected_pts;
};

struct DPTables {
    // dp_t1[T][var][sc][R]   layout, R dim padded to kDPUpperRPad
    std::vector<float> dp_t1;
    // dp_t4[T][var][sc][S]   layout, S dim padded to kDPUpperSumPad
    std::vector<float> dp_t4;
    // dp_mid[T][var][sc]
    std::vector<DPVal> dp_mid;
    // dp_low[T][var][sc]
    std::vector<DPVal> dp_low;
};

// =========================================================================
// Encoders / Decoders
// Invalid (un-mapped) values in Sc clamp to index 1 (== a "0" constraint).
//
// Middle Sc convention: SS / LS each take values in
//   {-1 (filled), 0 (no constraint), 21..30 (golden threshold),
//    31 (forced scratch — mutual destruction)}.
// At runtime, callers must pass Sc[LS]=31 if ctx.ss_scratched is true and
// LS is still empty (and likewise Sc[SS]=31 if ctx.ls_scratched is true).
// =========================================================================
int  encode_upper (const int8_t Sc[6]);
void decode_upper (int id, int8_t Sc[6]);
int  encode_middle(int8_t ss, int8_t ls);
void decode_middle(int id, int8_t Sc[2]);
int  encode_lower (const int8_t Sc[5]);
void decode_lower (int id, int8_t Sc[5]);

// =========================================================================
// Placement variant — return cell indices that are legal placements under v.
// N must equal kDPUpperCells / kDPMiddleCells / kDPLowerCells appropriately.
// =========================================================================
std::vector<int> get_valid_cells(Variant v, const int8_t Sc[], int N);

// =========================================================================
// Initialise tables. Loads from cache_path if it exists; else computes and
// saves. Pass empty cache_path to compute fresh without persistence.
// =========================================================================
void init_dp_tables(DPTables& dp,
                    const PrecomputedTables& tables,
                    const std::string& cache_path = "cache/dp_tables/dp_v1.bin");

// =========================================================================
// Query API — O(1) lookups after initialisation.
// =========================================================================
float get_upper_prob (const DPTables& dp, Variant v, const int8_t Sc[6], int T, int R);
float get_upper_ev   (const DPTables& dp, Variant v, const int8_t Sc[6], int T, int current_sum);
float get_middle_prob(const DPTables& dp, Variant v, const int8_t Sc[2], int T);
float get_middle_ev  (const DPTables& dp, Variant v, const int8_t Sc[2], int T);
float get_lower_prob (const DPTables& dp, Variant v, const int8_t Sc[5], int T);
float get_lower_ev   (const DPTables& dp, Variant v, const int8_t Sc[5], int T);

// =========================================================================
// Helpers exposed for tests.
// =========================================================================
inline int upper_bonus(int sum) {
    return sum >= 100 ? 500
         : sum >=  90 ? 200
         : sum >=  80 ? 100
         : sum >=  70 ?  50
         : sum >=  60 ?  30
         : 0;
}
