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

bool is_five_of_a_kind(const int8_t dice[kNumDice]) {
    for (int i = 1; i < kNumDice; ++i)
        if (dice[i] != dice[0]) return false;
    return true;
}

// ---------------------------------------------------------------------------
// validate_score — Golden Rule + SS/LS interlock for a candidate raw score.
// ---------------------------------------------------------------------------

template <typename Traits>
int validate_score(int row, int raw, int player, int column,
                   const BoardStateT<Traits>& board,
                   const GameContextT<Traits>& ctx) {
    // If raw == 0, the dice don't qualify → forced scratch
    if (raw == 0) return 0;

    // Golden Rule: score must be >= golden_max for this (col, row).
    // In 2v2 this includes the teammate's score — golden_max is populated
    // by every player, by design.
    int gmax = ctx.golden_max[column][row];
    if (raw < gmax) return 0;

    // SS-specific checks (row 6)
    if (row == kRowSS) {
        // If LS is already scratched for this player, SS must also be scratched
        if (ctx.ls_scratched[player][column]) return 0;

        // If LS is already filled for this player, SS must be strictly less than it
        int8_t ls_val = board.cells[player][column][kRowLS];
        if (ls_val != kCellEmpty && ls_val != 0 && raw >= ls_val) return 0;
    }

    // LS-specific checks (row 7)
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

    int raw = compute_raw_score(sorted, row);
    return validate_score<Traits>(row, raw, player, column, board, ctx);
}

// ---------------------------------------------------------------------------
// calculate_yams_bonus_score — "Lucky Yams" wildcard placement.
//
// Iterates the row's achievable scores from the highest downward and returns
// the first that passes validate_score (Golden Rule + SS/LS interlock). For
// every row except SS/LS the row maximum always satisfies the Golden Rule
// (the global max can never exceed the row max), so the loop returns the row
// max immediately. SS (20..29) and LS (20..30) ranges are contiguous and fully
// achievable as dice sums, so scanning them downward yields exactly the
// maximum legal value.
// ---------------------------------------------------------------------------

template <typename Traits>
int calculate_yams_bonus_score(int row, int player, int column,
                               const BoardStateT<Traits>& board,
                               const GameContextT<Traits>& ctx) {
    // Highest achievable raw score per row (mirrors compute_raw_score maxima).
    auto try_range = [&](int hi, int lo) -> int {
        for (int s = hi; s >= lo; --s) {
            int v = validate_score<Traits>(row, s, player, column, board, ctx);
            if (v > 0) return v;
        }
        return 0;
    };

    if (row >= kRow1s && row <= kRow6s) {
        // Number rows: only the row max (5 of that face) is meaningful here.
        return try_range(5 * (row + 1), 5 * (row + 1));
    }
    switch (row) {
    case kRowSS:  return try_range(29, 20);
    case kRowLS:  return try_range(30, 20);
    case kRowFH:  return try_range(50, 50);   // 20 + 30 (five 6s as full house)
    case kRowK:   return try_range(54, 54);   // 30 + 6*4
    case kRowSTR: return try_range(50, 50);   // large straight
    case kRowU8:  return try_range(75, 75);   // 60 + 5*(8-5)
    case kRowY:   return try_range(100, 100); // 75 + 5*(6-1)
    default:      return 0;
    }
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
template int  validate_score<Yams1v1>(int, int, int, int,
                                      const BoardStateT<Yams1v1>&,
                                      const GameContextT<Yams1v1>&);
template int  validate_score<Yams2v2>(int, int, int, int,
                                      const BoardStateT<Yams2v2>&,
                                      const GameContextT<Yams2v2>&);
template int  calculate_yams_bonus_score<Yams1v1>(int, int, int,
                                                  const BoardStateT<Yams1v1>&,
                                                  const GameContextT<Yams1v1>&);
template int  calculate_yams_bonus_score<Yams2v2>(int, int, int,
                                                  const BoardStateT<Yams2v2>&,
                                                  const GameContextT<Yams2v2>&);
template bool is_terminal<Yams1v1>(const BoardStateT<Yams1v1>&);
template bool is_terminal<Yams2v2>(const BoardStateT<Yams2v2>&);
template int  cells_remaining<Yams1v1>(const BoardStateT<Yams1v1>&, int);
template int  cells_remaining<Yams2v2>(const BoardStateT<Yams2v2>&, int);
template int  column_cells_remaining<Yams1v1>(const BoardStateT<Yams1v1>&, int, int);
template int  column_cells_remaining<Yams2v2>(const BoardStateT<Yams2v2>&, int, int);
