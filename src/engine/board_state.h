#pragma once

#include <cstdint>
#include <type_traits>
#include "engine/constants.h"
#include "engine/game_traits.h"

// ---------------------------------------------------------------------------
// BoardStateT<Traits> — the minimal, trivially-copyable board representation.
//
// This is the structure the solver clones when evaluating afterstates, and
// what the neural network evaluates. It must be small, cache-friendly, and
// trivially copyable so the solver can memcpy it freely.
//
// Layout:
//   cells[player][column][row]
//   Columns: 0=Down, 1=Free, 2=Up, 3=Mid, 4=Turbo, 5=UpDown
//   Rows: 0=1s, 1=2s, 2=3s, 3=4s, 4=5s, 5=6s,
//          6=SS, 7=LS, 8=FH, 9=K, 10=STR, 11=U8, 12=Y
// ---------------------------------------------------------------------------
template <typename Traits>
struct BoardStateT {
    // Cell values: kCellEmpty(-1) = empty, 0 = scratched, 1-100 = score
    int8_t cells[Traits::kNumPlayers][kNumColumns][kNumRows];

    // Column coefficients (shuffled {8,10,12,14,16,18}), same for all players
    int8_t coefficients[kNumColumns];

    // Index of the player whose turn it is next
    int8_t current_player;

    // Total cells filled across all players (0..Traits::kTotalCells).
    // uint16_t (not uint8_t): 2v2 has 312 cells, which overflows uint8_t and
    // would silently prevent terminal detection.
    uint16_t cells_filled;
};

// Variant-specific instantiations and backward-compat alias.
using BoardState1v1 = BoardStateT<Yams1v1>;
using BoardState2v2 = BoardStateT<Yams2v2>;
using BoardState    = BoardState1v1;  // removed at end of migration

static_assert(std::is_trivially_copyable_v<BoardState1v1>,
              "BoardState1v1 must be trivially copyable for fast solver cloning");
static_assert(std::is_trivially_copyable_v<BoardState2v2>,
              "BoardState2v2 must be trivially copyable for fast solver cloning");
static_assert(sizeof(BoardState1v1) <= 168,
              "1v1 BoardState must fit in 3 cache lines (168 bytes)");
static_assert(sizeof(BoardState2v2) <= 336,
              "2v2 BoardState must fit in 6 cache lines (336 bytes)");
