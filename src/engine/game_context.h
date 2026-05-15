#pragma once

#include <cstdint>
#include "engine/constants.h"
#include "engine/board_state.h"
#include "engine/game_traits.h"

// ---------------------------------------------------------------------------
// Placement — a (column, row) pair identifying a cell on the board.
// ---------------------------------------------------------------------------
struct Placement {
    int8_t column;
    int8_t row;
};

// ---------------------------------------------------------------------------
// LegalPlacementCache — O(1) membership query + fast dense iteration.
//
// Size is variant-independent (78 = kNumColumns * kNumRows), so this struct
// is not templated.
//
// Two representations are kept in sync:
//   placements[] — dense array for fast iteration (solver's hot path)
//   is_legal[][] — 2D boolean for O(1) membership queries
//
// Zero heap allocation; entirely stack/struct allocated.
// ---------------------------------------------------------------------------
struct LegalPlacementCache {
    Placement placements[78];    // Dense array — max 78 legal cells per player
    int8_t    count;             // Number of valid entries in placements[]
    bool      is_legal[kNumColumns][kNumRows];  // O(1) lookup

    // Add an entry (no-op if already present)
    void add(int col, int row) {
        if (is_legal[col][row]) return;
        is_legal[col][row] = true;
        placements[count++] = {static_cast<int8_t>(col), static_cast<int8_t>(row)};
    }

    // Remove an entry (linear scan of the dense array, swap-with-last)
    void remove(int col, int row) {
        if (!is_legal[col][row]) return;
        is_legal[col][row] = false;
        for (int i = 0; i < count; ++i) {
            if (placements[i].column == col && placements[i].row == row) {
                placements[i] = placements[--count];
                return;
            }
        }
    }

    void clear() {
        count = 0;
        for (int c = 0; c < kNumColumns; ++c)
            for (int r = 0; r < kNumRows; ++r)
                is_legal[c][r] = false;
    }
};

// ---------------------------------------------------------------------------
// GameContextT<Traits> — cached derived data maintained during actual gameplay.
//
// Not cloned by the solver (too large / not needed for afterstate evaluation).
// Updated incrementally on each placement via apply_placement().
// ---------------------------------------------------------------------------
template <typename Traits>
struct GameContextT {
    // === Golden Rule cache ===
    // Maximum score placed in each (column, row) across all players (including
    // teammates in 2v2). Any new score must be >= this to be valid (or 0 to
    // scratch).
    int8_t golden_max[kNumColumns][kNumRows];

    // === Upper section sums (rows 0-5) per player per column ===
    // Used for upper-section bonus calculation in compute_duel().
    int16_t upper_sum[Traits::kNumPlayers][kNumColumns];

    // === SS/LS scratch status per player per column ===
    // When SS is scratched, LS is forced to scratch (and vice versa).
    bool ss_scratched[Traits::kNumPlayers][kNumColumns];
    bool ls_scratched[Traits::kNumPlayers][kNumColumns];

    // === Clean column tracking ===
    // True if any lower-section cell (rows 6-12) has been scratched.
    // A clean column requires upper_sum >= 60 AND no lower scratches.
    bool lower_has_scratch[Traits::kNumPlayers][kNumColumns];

    // === Cached legal placements per player ===
    // legal_all     — all placements including Turbo column
    // legal_no_turbo — non-Turbo placements only (used when rolls_left == 0)
    LegalPlacementCache legal_all[Traits::kNumPlayers];
    LegalPlacementCache legal_no_turbo[Traits::kNumPlayers];

    // === Non-Turbo cells remaining per player ===
    // Decremented on each non-Turbo placement.
    // When 0, the player can no longer reroll after their first roll
    // (because they'd be forced to place in Turbo which requires > 0 rolls).
    int8_t non_turbo_cells_remaining[Traits::kNumPlayers];
};

// Variant-specific instantiations and backward-compat alias.
using GameContext1v1 = GameContextT<Yams1v1>;
using GameContext2v2 = GameContextT<Yams2v2>;
using GameContext    = GameContext1v1;  // removed at end of migration
