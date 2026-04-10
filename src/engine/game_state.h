#pragma once

#include <cstdint>
#include "engine/board_state.h"
#include "engine/constants.h"

// ---------------------------------------------------------------------------
// GameState — extends BoardState with dice and roll information.
//
// Used during active gameplay. The solver works with BoardState clones;
// GameState is only needed by the game loop and game flow functions.
// ---------------------------------------------------------------------------
struct GameState {
    BoardState board;

    // Current dice values (1-6), valid during an active turn
    int8_t dice[kNumDice];

    // Rerolls remaining in the current turn (0, 1, or 2).
    // After the initial roll: rolls_left = 2.
    // After each reroll: rolls_left decremented.
    // At rolls_left == 0: player must place.
    int8_t rolls_left;

    // ~170 bytes total
};
