#pragma once

#include <cstdint>
#include <type_traits>
#include "engine/constants.h"

// ---------------------------------------------------------------------------
// BoardState — the minimal, trivially-copyable board representation.
//
// This is the structure the solver clones when evaluating afterstates, and
// what the neural network evaluates. It must be small, cache-friendly, and
// trivially copyable so the solver can memcpy it freely.
//
// Layout:
//   cells[player][column][row]
//   Player 0 = current evaluating player, Player 1 = opponent
//   Columns: 0=Down, 1=Free, 2=Up, 3=Mid, 4=Turbo, 5=UpDown
//   Rows: 0=1s, 1=2s, 2=3s, 3=4s, 4=5s, 5=6s,
//          6=SS, 7=LS, 8=FH, 9=K, 10=STR, 11=U8, 12=Y
// ---------------------------------------------------------------------------
struct BoardState {
    // Cell values: kCellEmpty(-1) = empty, 0 = scratched, 1-100 = score
    int8_t cells[kNumPlayers][kNumColumns][kNumRows];

    // Column coefficients (shuffled {8,10,12,14,16,18}), same for both players
    int8_t coefficients[kNumColumns];

    // Index of the player whose turn it is next (0 or 1)
    int8_t current_player;

    // Total cells filled across both players (0-156)
    // Used for progress tracking and terminal detection
    uint8_t cells_filled;

    // ~164 bytes total — fits in 3 cache lines
};

static_assert(std::is_trivially_copyable_v<BoardState>,
              "BoardState must be trivially copyable for fast solver cloning");
static_assert(sizeof(BoardState) <= 168,
              "BoardState must fit in 3 cache lines (168 bytes)");
