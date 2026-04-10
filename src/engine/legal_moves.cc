#include "engine/legal_moves.h"

#include <cassert>

// ---------------------------------------------------------------------------
// Circular adjacency helper
// ---------------------------------------------------------------------------

bool has_filled_neighbor(int player, int column, int row, const BoardState& board) {
    int prev = (row - 1 + kNumRows) % kNumRows;
    int next = (row + 1) % kNumRows;
    return board.cells[player][column][prev] != kCellEmpty ||
           board.cells[player][column][next] != kCellEmpty;
}

// ---------------------------------------------------------------------------
// Check whether a single (column, row) cell is a legal next placement for
// a player, given the current board state (column constraint only).
// Does NOT check if the cell is empty — caller must do that.
// ---------------------------------------------------------------------------
static bool is_column_legal(int player, int column, int row, const BoardState& board) {
    switch (column) {
    case kColDown:
        // Legal row = lowest-indexed empty row
        for (int r = 0; r < kNumRows; ++r) {
            if (board.cells[player][column][r] == kCellEmpty)
                return r == row;
        }
        return false;

    case kColFree:
        // Any empty cell is legal
        return true;

    case kColUp:
        // Legal row = highest-indexed empty row
        for (int r = kNumRows - 1; r >= 0; --r) {
            if (board.cells[player][column][r] == kCellEmpty)
                return r == row;
        }
        return false;

    case kColMid: {
        // If no cells filled: only rows 5 (6s) and 6 (SS) are legal
        bool any_filled = false;
        for (int r = 0; r < kNumRows; ++r)
            if (board.cells[player][column][r] != kCellEmpty) { any_filled = true; break; }
        if (!any_filled) return (row == kRow6s || row == kRowSS);
        // Otherwise: any empty cell with at least one filled neighbour (wraps)
        return has_filled_neighbor(player, column, row, board);
    }

    case kColTurbo:
        // Same as Free for placement purposes (roll restriction is turn-level)
        return true;

    case kColUpDown: {
        // If no cells filled: any row is legal
        bool any_filled = false;
        for (int r = 0; r < kNumRows; ++r)
            if (board.cells[player][column][r] != kCellEmpty) { any_filled = true; break; }
        if (!any_filled) return true;
        // Otherwise: any empty cell with at least one filled neighbour (wraps)
        return has_filled_neighbor(player, column, row, board);
    }

    default:
        return false;
    }
}

// ---------------------------------------------------------------------------
// rebuild_legal_placements — full rebuild from scratch
// ---------------------------------------------------------------------------

void rebuild_legal_placements(int player, const BoardState& board, GameContext& ctx) {
    ctx.legal_all[player].clear();
    ctx.legal_no_turbo[player].clear();

    for (int col = 0; col < kNumColumns; ++col) {
        for (int row = 0; row < kNumRows; ++row) {
            if (board.cells[player][col][row] != kCellEmpty) continue;
            if (!is_column_legal(player, col, row, board)) continue;
            ctx.legal_all[player].add(col, row);
            if (col != kColTurbo)
                ctx.legal_no_turbo[player].add(col, row);
        }
    }
}

// ---------------------------------------------------------------------------
// update_legal_placements_after_move — incremental update
// ---------------------------------------------------------------------------

void update_legal_placements_after_move(int player, int column, int row,
                                         const BoardState& board, GameContext& ctx) {
    // Remove the placed cell from both caches
    ctx.legal_all[player].remove(column, row);
    if (column != kColTurbo)
        ctx.legal_no_turbo[player].remove(column, row);

    // Determine newly legal cells based on column type
    switch (column) {
    case kColDown:
        // Next row in Down column becomes legal (if it exists and is empty)
        if (row + 1 < kNumRows && board.cells[player][column][row + 1] == kCellEmpty) {
            ctx.legal_all[player].add(column, row + 1);
            ctx.legal_no_turbo[player].add(column, row + 1);
        }
        break;

    case kColUp:
        // Previous row in Up column becomes legal (if it exists and is empty)
        if (row - 1 >= 0 && board.cells[player][column][row - 1] == kCellEmpty) {
            ctx.legal_all[player].add(column, row - 1);
            ctx.legal_no_turbo[player].add(column, row - 1);
        }
        break;

    case kColFree:
    case kColTurbo:
        // No new cells become legal — all empty cells were already in the cache
        break;

    case kColMid:
    case kColUpDown: {
        // For Mid and UpDown, legality is adjacency-based. After a placement the
        // full valid set may change (e.g., the first UpDown placement invalidates all
        // non-adjacent cells). Rebuild the affected column from scratch — only 13
        // rows to check, so this is O(13) and negligible.

        // Remove all current entries for this column
        for (int r = 0; r < kNumRows; ++r) {
            ctx.legal_all[player].remove(column, r);
            ctx.legal_no_turbo[player].remove(column, r);
        }
        // Re-add empty cells that have at least one filled neighbour
        for (int r = 0; r < kNumRows; ++r) {
            if (board.cells[player][column][r] == kCellEmpty &&
                has_filled_neighbor(player, column, r, board)) {
                ctx.legal_all[player].add(column, r);
                ctx.legal_no_turbo[player].add(column, r);
            }
        }
        break;
    }
    }
}
