#pragma once

#include <cstdint>
#include "engine/board_state.h"
#include "engine/constants.h"
#include "engine/game_traits.h"

// ---------------------------------------------------------------------------
// GameStateT<Traits> — extends BoardStateT with dice and roll information.
//
// Used during active gameplay. The solver works with BoardStateT clones;
// GameStateT is only needed by the game loop and game flow functions.
// ---------------------------------------------------------------------------
template <typename Traits>
struct GameStateT {
    BoardStateT<Traits> board;

    // Current dice values (1-6), valid during an active turn
    int8_t dice[kNumDice];

    // Rerolls remaining in the current turn (0, 1, or 2).
    // After the initial roll: rolls_left = 2.
    // After each reroll: rolls_left decremented.
    // At rolls_left == 0: player must place.
    int8_t rolls_left;
};

// Variant-specific instantiations and backward-compat alias.
using GameState1v1 = GameStateT<Yams1v1>;
using GameState2v2 = GameStateT<Yams2v2>;
using GameState    = GameState1v1;  // removed at end of migration
