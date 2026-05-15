#include "engine/board_init.h"
#include "engine/game_traits.h"
#include "engine/rng.h"
#include "engine/legal_moves.h"

#include <array>
#include <cstring>

template <typename Traits>
void init_board(BoardStateT<Traits>& board, RNG& rng) {
    // Zero/empty all cells
    for (int p = 0; p < Traits::kNumPlayers; ++p)
        for (int c = 0; c < kNumColumns; ++c)
            for (int r = 0; r < kNumRows; ++r)
                board.cells[p][c][r] = kCellEmpty;

    board.cells_filled = 0;

    // Shuffle coefficients {8, 10, 12, 14, 16, 18}
    std::array<int8_t, kNumColumns> coeffs = {8, 10, 12, 14, 16, 18};
    rng.shuffle(coeffs);
    for (int c = 0; c < kNumColumns; ++c)
        board.coefficients[c] = coeffs[c];

    // Randomly select starting player (A in 1v1, any seat in 2v2).
    board.current_player = static_cast<int8_t>(rng.uniform_int(0, Traits::kNumPlayers - 1));
}

template <typename Traits>
void init_context(GameContextT<Traits>& ctx, const BoardStateT<Traits>& board) {
    // Zero all cached values
    std::memset(&ctx, 0, sizeof(GameContextT<Traits>));

    // non_turbo_cells_remaining: 13 rows × 5 non-Turbo columns = 65 per player
    for (int p = 0; p < Traits::kNumPlayers; ++p) {
        ctx.non_turbo_cells_remaining[p] = 65;
    }

    // Build initial legal placements for all players
    for (int p = 0; p < Traits::kNumPlayers; ++p) {
        rebuild_legal_placements<Traits>(p, board, ctx);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template void init_board<Yams1v1>(BoardStateT<Yams1v1>&, RNG&);
template void init_board<Yams2v2>(BoardStateT<Yams2v2>&, RNG&);
template void init_context<Yams1v1>(GameContextT<Yams1v1>&, const BoardStateT<Yams1v1>&);
template void init_context<Yams2v2>(GameContextT<Yams2v2>&, const BoardStateT<Yams2v2>&);
