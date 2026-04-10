#include "engine/board_init.h"
#include "engine/rng.h"
#include "engine/legal_moves.h"

#include <array>
#include <cstring>

void init_board(BoardState& board, RNG& rng) {
    // Zero/empty all cells
    for (int p = 0; p < kNumPlayers; ++p)
        for (int c = 0; c < kNumColumns; ++c)
            for (int r = 0; r < kNumRows; ++r)
                board.cells[p][c][r] = kCellEmpty;

    board.cells_filled = 0;

    // Shuffle coefficients {8, 10, 12, 14, 16, 18}
    std::array<int8_t, kNumColumns> coeffs = {8, 10, 12, 14, 16, 18};
    rng.shuffle(coeffs);
    for (int c = 0; c < kNumColumns; ++c)
        board.coefficients[c] = coeffs[c];

    // Randomly select starting player
    board.current_player = static_cast<int8_t>(rng.uniform_int(0, 1));
}

void init_context(GameContext& ctx, const BoardState& board) {
    // Zero all cached values
    std::memset(&ctx, 0, sizeof(GameContext));

    // non_turbo_cells_remaining: 13 rows × 5 non-Turbo columns = 65 per player
    ctx.non_turbo_cells_remaining[0] = 65;
    ctx.non_turbo_cells_remaining[1] = 65;

    // Build initial legal placements for both players
    for (int p = 0; p < kNumPlayers; ++p) {
        rebuild_legal_placements(p, board, ctx);
    }
}
