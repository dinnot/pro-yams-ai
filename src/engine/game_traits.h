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

    // Tensor (NN observation) shape.
    //   Group A: kNumPlayers * 6 * 13 * 2     = 312
    //   Group B.1 (per-player × col):  N*6*2  =  24
    //   Group B.2 (per-pairing × col): P*6*14 =  84
    //   Group C: kNumPlayers * 6 * 13         = 156
    //   Group D (global aggregates):          =  14
    //   Group E: kNumPlayers * 6 * 18         = 216
    //   Group F: kNumPlayers * 6 * 15         = 180
    static constexpr int kTensorSize = 986;
    static constexpr int kNumPairings = 1;

    // Canonical pairings — index into canonical[ci] = (active + ci) % kNumPlayers.
    // For 1v1: a single pairing (Active vs Opp).
    static constexpr int kCanonicalPairingT0[kNumPairings] = {0};
    static constexpr int kCanonicalPairingT1[kNumPairings] = {1};
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

    // Tensor (NN observation) shape — 2× per-player groups, 4× per-pairing.
    //   Group A:   624,  B.1:  48,  B.2: 336,  C: 312,  D: 14,  E: 432,  F: 360
    static constexpr int kTensorSize = 2126;
    static constexpr int kNumPairings = 4;

    // Canonical view (relative to active player): canonical[0]=Active,
    // [1]=NextOpp, [2]=Teammate, [3]=PrevOpp. The four pairings are the
    // cross product {Active, Teammate} × {NextOpp, PrevOpp}, in this order:
    //   (Active, NextOpp), (Active, PrevOpp),
    //   (Teammate, NextOpp), (Teammate, PrevOpp).
    static constexpr int kCanonicalPairingT0[kNumPairings] = {0, 0, 2, 2};
    static constexpr int kCanonicalPairingT1[kNumPairings] = {1, 3, 1, 3};
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
