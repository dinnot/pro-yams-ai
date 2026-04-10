#include "engine/solver_tables.h"

#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Compute the flat index into state_id[] for 5 SORTED 0-based dice values.
static int dice_flat_index(const int8_t d[5]) {
    // d[i] is 0-based (0-5), sorted ascending
    return (int)d[0] * 1296 + (int)d[1] * 216 + (int)d[2] * 36 + (int)d[3] * 6 + (int)d[4];
}

// Enumerate all 252 sorted 5-dice combinations (with repetition) in
// lexicographic order. Fills dice_states and state_id.
static void enumerate_dice_states(SolverTables& t) {
    std::memset(t.state_id, -1, sizeof(t.state_id));
    int id = 0;
    // Iterate all combinations with repetition of {0..5} choose 5 (sorted)
    for (int a = 0; a <= 5; ++a)
      for (int b = a; b <= 5; ++b)
        for (int c = b; c <= 5; ++c)
          for (int d = c; d <= 5; ++d)
            for (int e = d; e <= 5; ++e) {
                t.dice_states[id][0] = (int8_t)(a + 1);
                t.dice_states[id][1] = (int8_t)(b + 1);
                t.dice_states[id][2] = (int8_t)(c + 1);
                t.dice_states[id][3] = (int8_t)(d + 1);
                t.dice_states[id][4] = (int8_t)(e + 1);
                int8_t zero_based[5] = {(int8_t)a,(int8_t)b,(int8_t)c,(int8_t)d,(int8_t)e};
                t.state_id[dice_flat_index(zero_based)] = (int16_t)id;
                ++id;
            }
    // Should be exactly 252
}

// ---------------------------------------------------------------------------
// compute_raw_score — pure dice math, no game-state dependencies.
// dice must be SORTED ascending (1-6).
// ---------------------------------------------------------------------------
int compute_raw_score(const int8_t dice[kNumDice], int row) {
    // Frequency counts: counts[f] = number of dice showing face f (1-6)
    int counts[7] = {};
    int total = 0;
    for (int i = 0; i < kNumDice; ++i) {
        counts[dice[i]]++;
        total += dice[i];
    }

    if (row <= 5) {
        // Number rows 0-5: sum of dice matching face (row+1)
        return counts[row + 1] * (row + 1);
    }

    switch (row) {
    case kRowSS: {
        // Small Sum: sum >= 20, capped at 29
        if (total >= 20) return std::min(total, 29);
        return 0;
    }
    case kRowLS: {
        // Large Sum: sum >= 20 (no upper cap in raw table)
        if (total >= 20) return total;
        return 0;
    }
    case kRowFH: {
        // Full House: exactly 3-of-a-kind + 2-of-a-kind (or 5-of-a-kind counts)
        int twos = 0, threes = 0;
        for (int f = 1; f <= 6; ++f) {
            if (counts[f] == 2) twos++;
            if (counts[f] == 3) threes++;
            if (counts[f] == 5) { threes++; twos++; }  // Yams qualifies as FH
        }
        // Exactly one triple and one pair (Yams: triple+pair both from same face)
        if (threes == 1 && twos == 1) return 20 + total;
        return 0;
    }
    case kRowK: {
        // Four of a Kind: 4+ identical dice; score = 30 + (face × 4)
        for (int f = 1; f <= 6; ++f) {
            if (counts[f] >= 4) return 30 + f * 4;
        }
        return 0;
    }
    case kRowSTR: {
        // Straight: 1-2-3-4-5 (score 45) or 2-3-4-5-6 (score 50)
        if (counts[1] >= 1 && counts[2] >= 1 && counts[3] >= 1 &&
            counts[4] >= 1 && counts[5] >= 1) return 45;
        if (counts[2] >= 1 && counts[3] >= 1 && counts[4] >= 1 &&
            counts[5] >= 1 && counts[6] >= 1) return 50;
        return 0;
    }
    case kRowU8: {
        // Under Eight: sum <= 8; score = 60 + 5*(8-sum)
        if (total <= 8) return 60 + 5 * (8 - total);
        return 0;
    }
    case kRowY: {
        // Yams (Five of a Kind): score = 75 + 5*(face-1)
        for (int f = 1; f <= 6; ++f) {
            if (counts[f] == 5) return 75 + 5 * (f - 1);
        }
        return 0;
    }
    default:
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Build possible_scores and filtered_scores tables
// ---------------------------------------------------------------------------
static void build_score_tables(SolverTables& t) {
    // Possible non-zero scores per row (fixed by rules, listed in design doc)
    // Row 0 (1s): 1..5
    // Row 1 (2s): 2,4,6,8,10
    // Row 2 (3s): 3,6,9,12,15
    // Row 3 (4s): 4,8,12,16,20
    // Row 4 (5s): 5,10,15,20,25
    // Row 5 (6s): 6,12,18,24,30
    // Row 6 (SS): 20..29
    // Row 7 (LS): 20..30
    // Row 8 (FH): 25..50
    // Row 9 (K):  34,38,42,46,50,54
    // Row 10(STR): 45,50
    // Row 11(U8): 60,65,70,75
    // Row 12(Y):  75,80,85,90,95,100
    //
    // We generate these by enumerating all 252 dice states rather than
    // hardcoding, to ensure consistency with dice_score.

    // Collect all distinct non-zero scores from the dice_score table
    for (int row = 0; row < kNumRows; ++row) {
        bool seen[101] = {};
        for (int sid = 0; sid < 252; ++sid) {
            int s = t.dice_score[sid][row];
            if (s > 0 && s <= 100 && !seen[s]) {
                seen[s] = true;
            }
        }
        // For SS, also cap at 29 (the raw table may not produce all values)
        // Actually: we generate possible_scores from the table, so they match.
        // Collect and sort
        int cnt = 0;
        for (int s = 1; s <= 100; ++s) {
            if (seen[s]) {
                t.possible_scores[row][cnt++] = (int8_t)s;
            }
        }
        t.possible_count[row] = (int8_t)cnt;

        // Build filtered_scores[row][threshold]
        for (int thresh = 0; thresh <= 100; ++thresh) {
            int fcnt = 0;
            for (int i = 0; i < cnt; ++i) {
                if (t.possible_scores[row][i] >= thresh) {
                    t.filtered_scores[row][thresh][fcnt++] = t.possible_scores[row][i];
                }
            }
            t.filtered_count[row][thresh] = (int8_t)fcnt;
        }
    }
}

// ---------------------------------------------------------------------------
// init_solver_tables — public entry point
// ---------------------------------------------------------------------------
void init_solver_tables(SolverTables& t) {
    // Step 1: enumerate the 252 sorted dice states
    enumerate_dice_states(t);

    // Step 2: for each state, compute raw scores for all rows
    for (int sid = 0; sid < 252; ++sid) {
        for (int row = 0; row < kNumRows; ++row) {
            int raw = compute_raw_score(t.dice_states[sid], row);
            // SS special: cap at 29 (raw score from compute_raw_score returns 0 if >29)
            // LS: no cap in raw table (per design doc)
            t.dice_score[sid][row] = (int8_t)raw;
        }
    }

    // Step 3: build possible_scores and filtered_scores from dice_score
    build_score_tables(t);
}
