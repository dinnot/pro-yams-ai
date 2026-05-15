#pragma once

// ---------------------------------------------------------------------------
// Game variant traits — compile-time configuration of player count and team
// structure. Use as template parameters to make engine code variant-agnostic.
//
// Contract: are_teammates(p, p) must return true for all valid p. TD-bootstrap
// correctness in the multi-agent setting depends on this — when a player's
// future trajectory step is themselves, the value is taken with no sign flip.
// ---------------------------------------------------------------------------

struct Yams1v1 {
    static constexpr int kNumPlayers      = 2;
    static constexpr int kNumTeams        = 2;
    static constexpr int kPlayersPerTeam  = 1;
    static constexpr int kCellsPerSheet   = 78;                            // 6 cols * 13 rows
    static constexpr int kTotalCells      = kNumPlayers * kCellsPerSheet;  // 156

    // Team rosters. Duels run as the cross product kTeam0 × kTeam1.
    static constexpr int kTeam0[kPlayersPerTeam] = {0};
    static constexpr int kTeam1[kPlayersPerTeam] = {1};

    static constexpr bool are_teammates(int p1, int p2) {
        // Every player is on their own (singleton) team.
        return p1 == p2;
    }
};

struct Yams2v2 {
    static constexpr int kNumPlayers      = 4;
    static constexpr int kNumTeams        = 2;
    static constexpr int kPlayersPerTeam  = 2;
    static constexpr int kCellsPerSheet   = 78;
    static constexpr int kTotalCells      = kNumPlayers * kCellsPerSheet;  // 312

    // Seating clockwise: A=P0, B=P1, C=P2, D=P3. Teams: {P0, P2} and {P1, P3}.
    // The four cross-team pairings (0,1), (0,3), (2,1), (2,3) cover exactly
    // each player's two neighbor duels — see pro_yams_2v2.md §5.
    static constexpr int kTeam0[kPlayersPerTeam] = {0, 2};
    static constexpr int kTeam1[kPlayersPerTeam] = {1, 3};

    static constexpr bool are_teammates(int p1, int p2) {
        // Players with the same parity are teammates.
        return (p1 & 1) == (p2 & 1);
    }
};

// Contract validation
static_assert(Yams1v1::are_teammates(0, 0) && Yams1v1::are_teammates(1, 1),
              "Yams1v1: self must be teammate of self");
static_assert(!Yams1v1::are_teammates(0, 1),
              "Yams1v1: distinct players are not teammates");

static_assert(Yams2v2::are_teammates(0, 0) && Yams2v2::are_teammates(1, 1) &&
              Yams2v2::are_teammates(2, 2) && Yams2v2::are_teammates(3, 3),
              "Yams2v2: self must be teammate of self");
static_assert(Yams2v2::are_teammates(0, 2) && Yams2v2::are_teammates(2, 0),
              "Yams2v2: P0 and P2 are teammates");
static_assert(Yams2v2::are_teammates(1, 3) && Yams2v2::are_teammates(3, 1),
              "Yams2v2: P1 and P3 are teammates");
static_assert(!Yams2v2::are_teammates(0, 1) && !Yams2v2::are_teammates(0, 3) &&
              !Yams2v2::are_teammates(1, 2) && !Yams2v2::are_teammates(2, 3),
              "Yams2v2: cross-team pairs are not teammates");
