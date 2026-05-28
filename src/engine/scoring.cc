#include "engine/scoring.h"
#include "engine/game_traits.h"
#include "engine/solver_tables.h"

#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
// Dice utilities (variant-independent)
// ---------------------------------------------------------------------------

void sort_dice(int8_t dice[kNumDice]) {
    // Insertion sort — array is tiny (5 elements)
    for (int i = 1; i < kNumDice; ++i) {
        int8_t key = dice[i];
        int j = i - 1;
        while (j >= 0 && dice[j] > key) {
            dice[j + 1] = dice[j];
            --j;
        }
        dice[j + 1] = key;
    }
}

void dice_counts(const int8_t dice[kNumDice], int counts[7]) {
    for (int f = 0; f <= 6; ++f) counts[f] = 0;
    for (int i = 0; i < kNumDice; ++i) counts[dice[i]]++;
}

int dice_sum(const int8_t dice[kNumDice]) {
    int s = 0;
    for (int i = 0; i < kNumDice; ++i) s += dice[i];
    return s;
}

// ---------------------------------------------------------------------------
// calculate_score
// ---------------------------------------------------------------------------

template <typename Traits>
int calculate_score(int row, const int8_t dice[kNumDice],
                    int player, int column,
                    const BoardStateT<Traits>& board,
                    const GameContextT<Traits>& ctx) {
    // Sort a local copy so we can use compute_raw_score
    int8_t sorted[kNumDice];
    for (int i = 0; i < kNumDice; ++i) sorted[i] = dice[i];
    sort_dice(sorted);

    // 1. Compute raw dice score
    int raw = compute_raw_score(sorted, row);

    // 2. If raw == 0, the dice don't qualify → forced scratch
    if (raw == 0) return 0;

    // 3. Golden Rule: score must be >= golden_max for this (col, row).
    //    In 2v2 this includes the teammate's score — golden_max is populated
    //    by every player, by design.
    int gmax = ctx.golden_max[column][row];
    if (raw < gmax) return 0;

    // 4. SS-specific checks (row 6)
    if (row == kRowSS) {
        // If LS is already scratched for this player, SS must also be scratched
        if (ctx.ls_scratched[player][column]) return 0;

        // If LS is already filled for this player, SS must be strictly less than it
        int8_t ls_val = board.cells[player][column][kRowLS];
        if (ls_val != kCellEmpty && ls_val != 0 && raw >= ls_val) return 0;
    }

    // 5. LS-specific checks (row 7)
    if (row == kRowLS) {
        // If SS is already scratched for this player, LS must also be scratched
        if (ctx.ss_scratched[player][column]) return 0;

        // LS must be strictly greater than the highest SS recorded by anyone
        // (teammate included in 2v2 — golden_max is the global max).
        int max_ss = ctx.golden_max[column][kRowSS];
        if (max_ss > 0 && raw <= max_ss) return 0;
    }

    return raw;
}

// ---------------------------------------------------------------------------
// Board queries
// ---------------------------------------------------------------------------

template <typename Traits>
bool is_terminal(const BoardStateT<Traits>& board) {
    return board.cells_filled >= Traits::kTotalCells;
}

template <typename Traits>
int cells_remaining(const BoardStateT<Traits>& board, int player) {
    int count = 0;
    for (int c = 0; c < kNumColumns; ++c)
        for (int r = 0; r < kNumRows; ++r)
            if (board.cells[player][c][r] == kCellEmpty) count++;
    return count;
}

template <typename Traits>
int column_cells_remaining(const BoardStateT<Traits>& board, int player, int column) {
    int count = 0;
    for (int r = 0; r < kNumRows; ++r)
        if (board.cells[player][column][r] == kCellEmpty) count++;
    return count;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template int  calculate_score<Yams1v1>(int, const int8_t[], int, int,
                                       const BoardStateT<Yams1v1>&,
                                       const GameContextT<Yams1v1>&);
template int  calculate_score<Yams2v2>(int, const int8_t[], int, int,
                                       const BoardStateT<Yams2v2>&,
                                       const GameContextT<Yams2v2>&);
template bool is_terminal<Yams1v1>(const BoardStateT<Yams1v1>&);
template bool is_terminal<Yams2v2>(const BoardStateT<Yams2v2>&);
template int  cells_remaining<Yams1v1>(const BoardStateT<Yams1v1>&, int);
template int  cells_remaining<Yams2v2>(const BoardStateT<Yams2v2>&, int);
template int  column_cells_remaining<Yams1v1>(const BoardStateT<Yams1v1>&, int, int);
template int  column_cells_remaining<Yams2v2>(const BoardStateT<Yams2v2>&, int, int);
