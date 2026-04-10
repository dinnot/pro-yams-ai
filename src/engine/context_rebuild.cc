#include "engine/context_rebuild.h"

#include <cstring>

#include "engine/constants.h"
#include "engine/legal_moves.h"

void rebuild_context_from_board(const BoardState& board, GameContext& ctx) {
    std::memset(&ctx, 0, sizeof(GameContext));

    for (int p = 0; p < kNumPlayers; ++p) {
        for (int c = 0; c < kNumColumns; ++c) {
            for (int r = 0; r < kNumRows; ++r) {
                int8_t cell = board.cells[p][c][r];
                if (cell == kCellEmpty) continue;

                // Golden max (both players contribute).
                if (cell > ctx.golden_max[c][r])
                    ctx.golden_max[c][r] = cell;

                // Upper section sum (rows 0-5).
                if (r < 6 && cell > 0)
                    ctx.upper_sum[p][c] += cell;

                // SS/LS scratch tracking.
                if (r == kRowSS && cell == 0)
                    ctx.ss_scratched[p][c] = true;
                if (r == kRowLS && cell == 0)
                    ctx.ls_scratched[p][c] = true;

                // Clean column tracking: any scratch in lower section (rows 6-12).
                if (r >= 6 && cell == 0)
                    ctx.lower_has_scratch[p][c] = true;
            }
        }

        // Non-turbo cells remaining.
        int count = 0;
        for (int c = 0; c < kNumColumns; ++c) {
            if (c == kColTurbo) continue;
            for (int r = 0; r < kNumRows; ++r) {
                if (board.cells[p][c][r] == kCellEmpty) count++;
            }
        }
        ctx.non_turbo_cells_remaining[p] = static_cast<int8_t>(count);

        // Rebuild legal placements.
        rebuild_legal_placements(p, board, ctx);
    }
}
