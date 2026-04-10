#include "engine/placement.h"
#include "engine/legal_moves.h"

#include <algorithm>
#include <cassert>

// Recompute golden_max for a (column, row) from the actual cell values.
// Called after mutual SS/LS destruction overwrites a previously non-zero cell.
static void recalc_golden_max(int column, int row, const BoardState& board, GameContext& ctx) {
    int8_t best = 0;
    for (int p = 0; p < kNumPlayers; ++p) {
        int8_t v = board.cells[p][column][row];
        if (v > best) best = v;
    }
    ctx.golden_max[column][row] = best;
}

void apply_placement(int player, int column, int row, int score,
                     BoardState& board, GameContext& ctx) {
    assert(board.cells[player][column][row] == kCellEmpty &&
           "Cannot place in an already filled cell!");

    // 1. Write the cell
    board.cells[player][column][row] = static_cast<int8_t>(score);

    // 2. Increment cells_filled
    board.cells_filled++;

    // 3. Update Golden Rule cache
    if (score > ctx.golden_max[column][row]) {
        ctx.golden_max[column][row] = static_cast<int8_t>(score);
    }

    // 4. Update upper section sum (rows 0-5 only)
    if (row <= kRow6s) {
        ctx.upper_sum[player][column] += static_cast<int16_t>(score);
    }

    // 5. SS/LS scratch tracking and mutual destruction
    if (row == kRowSS && score == 0) {
        ctx.ss_scratched[player][column] = true;
        // If LS is already filled with a non-zero score, scratch it too
        int8_t ls_val = board.cells[player][column][kRowLS];
        if (ls_val != kCellEmpty && ls_val != 0) {
            board.cells[player][column][kRowLS] = 0;
            ctx.ls_scratched[player][column] = true;
            ctx.lower_has_scratch[player][column] = true;
            recalc_golden_max(column, kRowLS, board, ctx);
        }
    } else if (row == kRowLS && score == 0) {
        ctx.ls_scratched[player][column] = true;
        // If SS is already filled with a non-zero score, scratch it too
        int8_t ss_val = board.cells[player][column][kRowSS];
        if (ss_val != kCellEmpty && ss_val != 0) {
            board.cells[player][column][kRowSS] = 0;
            ctx.ss_scratched[player][column] = true;
            ctx.lower_has_scratch[player][column] = true;
            recalc_golden_max(column, kRowSS, board, ctx);
        }
    }

    // 6. Update clean column tracking (lower section rows 6-12)
    if (row >= kRowSS && score == 0) {
        ctx.lower_has_scratch[player][column] = true;
    }

    // 7. Update non-Turbo counter
    if (column != kColTurbo) {
        ctx.non_turbo_cells_remaining[player]--;
    }

    // 8. Update legal placements (incremental)
    update_legal_placements_after_move(player, column, row, board, ctx);
}
