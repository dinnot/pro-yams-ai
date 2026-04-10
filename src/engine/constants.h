#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Global constants and enumerations for Pro Yams
// ---------------------------------------------------------------------------

constexpr int kNumPlayers  = 2;
constexpr int kNumColumns  = 6;
constexpr int kNumRows     = 13;
constexpr int kNumDice     = 5;
constexpr int kNumDieSides = 6;
// Total cells on both players' boards
constexpr int kTotalCells  = kNumPlayers * kNumColumns * kNumRows;  // 156

// Column indices (fixed assignment per design doc)
enum Column : int8_t {
    kColDown   = 0,  // Must fill top-to-bottom
    kColFree   = 1,  // Any order
    kColUp     = 2,  // Must fill bottom-to-top
    kColMid    = 3,  // Start from row 5 or 6, expand outward (wraps)
    kColTurbo  = 4,  // Any order, but limited to 2 rolls per turn
    kColUpDown = 5   // First anywhere, then must be adjacent to an existing cell (wraps)
};

// Row indices
enum Row : int8_t {
    kRow1s  =  0,
    kRow2s  =  1,
    kRow3s  =  2,
    kRow4s  =  3,
    kRow5s  =  4,
    kRow6s  =  5,
    kRowSS  =  6,   // Small Sum
    kRowLS  =  7,   // Large Sum
    kRowFH  =  8,   // Full House
    kRowK   =  9,   // Four of a Kind
    kRowSTR = 10,   // Straight
    kRowU8  = 11,   // Under Eight
    kRowY   = 12    // Yams (Five of a Kind)
};

// Sentinel value for an empty cell
constexpr int8_t kCellEmpty = -1;

// The six column coefficients, shuffled at game start
constexpr int8_t kCoefficients[] = {8, 10, 12, 14, 16, 18};
