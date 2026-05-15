#include "engine/game_traits.h"

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Yams1v1 trait
// ---------------------------------------------------------------------------
TEST(GameTraits1v1, BasicConstants) {
    EXPECT_EQ(Yams1v1::kNumPlayers, 2);
    EXPECT_EQ(Yams1v1::kNumTeams, 2);
    EXPECT_EQ(Yams1v1::kCellsPerSheet, 78);
    EXPECT_EQ(Yams1v1::kTotalCells, 156);
}

TEST(GameTraits1v1, AreTeammates_SelfIsTeammate) {
    EXPECT_TRUE(Yams1v1::are_teammates(0, 0));
    EXPECT_TRUE(Yams1v1::are_teammates(1, 1));
}

TEST(GameTraits1v1, AreTeammates_DistinctPlayersAreNot) {
    EXPECT_FALSE(Yams1v1::are_teammates(0, 1));
    EXPECT_FALSE(Yams1v1::are_teammates(1, 0));
}

// ---------------------------------------------------------------------------
// Yams2v2 trait
// ---------------------------------------------------------------------------
TEST(GameTraits2v2, BasicConstants) {
    EXPECT_EQ(Yams2v2::kNumPlayers, 4);
    EXPECT_EQ(Yams2v2::kNumTeams, 2);
    EXPECT_EQ(Yams2v2::kCellsPerSheet, 78);
    EXPECT_EQ(Yams2v2::kTotalCells, 312);
}

TEST(GameTraits2v2, AreTeammates_SelfIsTeammate) {
    for (int p = 0; p < Yams2v2::kNumPlayers; ++p) {
        EXPECT_TRUE(Yams2v2::are_teammates(p, p)) << "p=" << p;
    }
}

TEST(GameTraits2v2, AreTeammates_Team0_IsP0AndP2) {
    EXPECT_TRUE(Yams2v2::are_teammates(0, 2));
    EXPECT_TRUE(Yams2v2::are_teammates(2, 0));
}

TEST(GameTraits2v2, AreTeammates_Team1_IsP1AndP3) {
    EXPECT_TRUE(Yams2v2::are_teammates(1, 3));
    EXPECT_TRUE(Yams2v2::are_teammates(3, 1));
}

TEST(GameTraits2v2, AreTeammates_CrossTeamIsNot) {
    EXPECT_FALSE(Yams2v2::are_teammates(0, 1));
    EXPECT_FALSE(Yams2v2::are_teammates(0, 3));
    EXPECT_FALSE(Yams2v2::are_teammates(1, 2));
    EXPECT_FALSE(Yams2v2::are_teammates(2, 3));
    EXPECT_FALSE(Yams2v2::are_teammates(1, 0));
    EXPECT_FALSE(Yams2v2::are_teammates(3, 0));
    EXPECT_FALSE(Yams2v2::are_teammates(2, 1));
    EXPECT_FALSE(Yams2v2::are_teammates(3, 2));
}

TEST(GameTraits2v2, AreTeammates_SymmetricAcrossAllPairs) {
    for (int p1 = 0; p1 < Yams2v2::kNumPlayers; ++p1) {
        for (int p2 = 0; p2 < Yams2v2::kNumPlayers; ++p2) {
            EXPECT_EQ(Yams2v2::are_teammates(p1, p2),
                      Yams2v2::are_teammates(p2, p1))
                << "asymmetric for (" << p1 << ", " << p2 << ")";
        }
    }
}
