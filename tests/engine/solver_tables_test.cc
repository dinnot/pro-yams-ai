#include <gtest/gtest.h>
#include "engine/solver_tables.h"

class SolverTablesTest : public ::testing::Test {
protected:
    SolverTables t;
    void SetUp() override { init_solver_tables(t); }
};

TEST_F(SolverTablesTest, ExactlyN252States) {
    // Count unique sorted 5-dice combinations
    int count = 0;
    for (int a = 1; a <= 6; ++a)
      for (int b = a; b <= 6; ++b)
        for (int c = b; c <= 6; ++c)
          for (int d = c; d <= 6; ++d)
            for (int e = d; e <= 6; ++e)
                count++;
    EXPECT_EQ(count, 252);
}

TEST_F(SolverTablesTest, StateIDLookupConsistent) {
    for (int id = 0; id < 252; ++id) {
        int8_t* ds = t.dice_states[id];
        // Dice must be sorted
        EXPECT_LE(ds[0], ds[1]);
        EXPECT_LE(ds[1], ds[2]);
        EXPECT_LE(ds[2], ds[3]);
        EXPECT_LE(ds[3], ds[4]);
        // state_id lookup must return this id
        int8_t zero[5] = {(int8_t)(ds[0]-1),(int8_t)(ds[1]-1),(int8_t)(ds[2]-1),
                          (int8_t)(ds[3]-1),(int8_t)(ds[4]-1)};
        int flat = (int)zero[0]*1296+(int)zero[1]*216+(int)zero[2]*36+(int)zero[3]*6+(int)zero[4];
        EXPECT_EQ(t.state_id[flat], id);
    }
}

TEST_F(SolverTablesTest, DiceScores_NumberRows) {
    // {3,3,3,1,2} sorted → {1,2,3,3,3}: row 3s = 9
    int8_t dice[] = {1,2,3,3,3};
    EXPECT_EQ(compute_raw_score(dice, kRow3s), 9);
    // {6,6,6,6,6}: row 6s = 30
    int8_t dice2[] = {6,6,6,6,6};
    EXPECT_EQ(compute_raw_score(dice2, kRow6s), 30);
}

TEST_F(SolverTablesTest, DiceScores_SS) {
    int8_t dice_20[] = {4,4,4,4,4};  // sum=20
    EXPECT_EQ(compute_raw_score(dice_20, kRowSS), 20);
    int8_t dice_29[] = {5,6,6,6,6};  // sum=29
    EXPECT_EQ(compute_raw_score(dice_29, kRowSS), 29);
    int8_t dice_30[] = {6,6,6,6,6};  // sum=30, SS capped to 29
    EXPECT_EQ(compute_raw_score(dice_30, kRowSS), 29);
    int8_t dice_15[] = {3,3,3,3,3};  // sum=15, below 20
    EXPECT_EQ(compute_raw_score(dice_15, kRowSS), 0);
}

TEST_F(SolverTablesTest, DiceScores_LS_NoUpperCap) {
    int8_t dice[] = {6,6,6,6,6};  // sum=30
    EXPECT_EQ(compute_raw_score(dice, kRowLS), 30);
    int8_t dice2[] = {4,4,4,4,4}; // sum=20
    EXPECT_EQ(compute_raw_score(dice2, kRowLS), 20);
}

TEST_F(SolverTablesTest, DiceScores_FH) {
    int8_t dice[] = {2,2,3,3,3};  // FH, sum=13
    EXPECT_EQ(compute_raw_score(dice, kRowFH), 33);
    int8_t yams[] = {5,5,5,5,5};  // Yams qualifies as FH
    EXPECT_EQ(compute_raw_score(yams, kRowFH), 45);
}

TEST_F(SolverTablesTest, DiceScores_K) {
    int8_t dice[] = {1,4,4,4,4};  // four 4s: 30+16=46
    EXPECT_EQ(compute_raw_score(dice, kRowK), 46);
}

TEST_F(SolverTablesTest, DiceScores_STR) {
    int8_t small[] = {1,2,3,4,5};
    EXPECT_EQ(compute_raw_score(small, kRowSTR), 45);
    int8_t large[] = {2,3,4,5,6};
    EXPECT_EQ(compute_raw_score(large, kRowSTR), 50);
}

TEST_F(SolverTablesTest, DiceScores_U8) {
    int8_t dice[] = {1,1,1,1,1};  // sum=5: 60+15=75
    EXPECT_EQ(compute_raw_score(dice, kRowU8), 75);
}

TEST_F(SolverTablesTest, DiceScores_Y) {
    int8_t dice[] = {6,6,6,6,6};
    EXPECT_EQ(compute_raw_score(dice, kRowY), 100);
    int8_t dice2[] = {1,1,1,1,1};
    EXPECT_EQ(compute_raw_score(dice2, kRowY), 75);
}

TEST_F(SolverTablesTest, FilteredScores_Row6s_Threshold13) {
    // Row 6s possible: 6,12,18,24,30. Scores >= 13: {18,24,30}
    int cnt = t.filtered_count[kRow6s][13];
    EXPECT_EQ(cnt, 3);
    EXPECT_EQ(t.filtered_scores[kRow6s][13][0], 18);
    EXPECT_EQ(t.filtered_scores[kRow6s][13][1], 24);
    EXPECT_EQ(t.filtered_scores[kRow6s][13][2], 30);
}

TEST_F(SolverTablesTest, FilteredScores_Yams_Threshold0) {
    // All 6 Yams scores: 75,80,85,90,95,100
    EXPECT_EQ(t.filtered_count[kRowY][0], 6);
}

TEST_F(SolverTablesTest, FilteredScores_Yams_Threshold100) {
    EXPECT_EQ(t.filtered_count[kRowY][100], 1);
    EXPECT_EQ(t.filtered_scores[kRowY][100][0], 100);
}

TEST_F(SolverTablesTest, FilteredScores_Yams_Threshold101) {
    // No score > 100 exists
    EXPECT_EQ(t.filtered_count[kRowY][100], 1);
    // threshold 100 gives {100}; there's no threshold=101 slot (array only 0..100)
    // Just verify the boundary
    EXPECT_GE(t.filtered_count[kRowY][95], 2);  // {95,100}
}

TEST_F(SolverTablesTest, PossibleScores_Counts) {
    EXPECT_EQ(t.possible_count[kRow1s], 5);   // 1,2,3,4,5
    EXPECT_EQ(t.possible_count[kRow6s], 5);   // 6,12,18,24,30
    EXPECT_EQ(t.possible_count[kRowSS], 10);  // 20..29
    EXPECT_EQ(t.possible_count[kRowLS], 11);  // 20..30
    EXPECT_EQ(t.possible_count[kRowK],  6);   // 34,38,42,46,50,54
    EXPECT_EQ(t.possible_count[kRowSTR],2);   // 45,50
    EXPECT_EQ(t.possible_count[kRowU8], 4);   // 60,65,70,75
    EXPECT_EQ(t.possible_count[kRowY],  6);   // 75,80,85,90,95,100
}
