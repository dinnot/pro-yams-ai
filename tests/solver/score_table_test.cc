#include <gtest/gtest.h>
#include "solver/precomputed_tables.h"
#include "engine/constants.h"

// Shared fixture
class ScoreTableTest : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) { init_precomputed_tables(tables); initialised = true; }
    }
    static int state_of(int a, int b, int c, int d, int e) {
        int8_t dice[5] = {(int8_t)a,(int8_t)b,(int8_t)c,(int8_t)d,(int8_t)e};
        return get_dice_state_id(dice, tables);
    }
};
PrecomputedTables ScoreTableTest::tables;
bool ScoreTableTest::initialised = false;

// ---------------------------------------------------------------------------
// Spot checks: dice_score[][]
// ---------------------------------------------------------------------------
TEST_F(ScoreTableTest, DiceScore_ThreeThreesAndTwoTwos) {
    int sid = state_of(2,2,3,3,3);  // sorted: {2,2,3,3,3}
    EXPECT_EQ(get_dice_score(sid, kRow3s, tables), 9);     // 3×3
    EXPECT_EQ(get_dice_score(sid, kRowFH, tables), 33);    // 20 + (2+2+3+3+3) = 33
    EXPECT_EQ(get_dice_score(sid, kRowK, tables), 0);      // no four-of-a-kind
    EXPECT_EQ(get_dice_score(sid, kRowY, tables), 0);      // no five-of-a-kind
}

TEST_F(ScoreTableTest, DiceScore_Straight12345) {
    int sid = state_of(1,2,3,4,5);
    EXPECT_EQ(get_dice_score(sid, kRowSTR, tables), 45);   // 1-2-3-4-5 straight
    EXPECT_EQ(get_dice_score(sid, kRowFH, tables), 0);     // not a FH
    EXPECT_EQ(get_dice_score(sid, kRowSS, tables), 0);     // sum=15 < 20
}

TEST_F(ScoreTableTest, DiceScore_AllSixes) {
    int sid = state_of(6,6,6,6,6);
    EXPECT_EQ(get_dice_score(sid, kRowY, tables), 100);    // 75 + 5*(6-1) = 100
    EXPECT_EQ(get_dice_score(sid, kRowK, tables), 54);     // 30 + 6*4 = 54
    EXPECT_EQ(get_dice_score(sid, kRowFH, tables), 50);    // 20 + 30 = 50 (Yams qualifies)
    EXPECT_EQ(get_dice_score(sid, kRow6s, tables), 30);    // 5×6 = 30
    EXPECT_EQ(get_dice_score(sid, kRowU8, tables), 0);     // sum=30 > 8
}

TEST_F(ScoreTableTest, DiceScore_AllOnes) {
    int sid = state_of(1,1,1,1,1);
    EXPECT_EQ(get_dice_score(sid, kRowY, tables), 75);     // 75 + 5*(1-1) = 75
    EXPECT_EQ(get_dice_score(sid, kRowU8, tables), 75);    // 60 + 5*(8-5) = 75
    EXPECT_EQ(get_dice_score(sid, kRow1s, tables), 5);     // 5×1 = 5
}

TEST_F(ScoreTableTest, DiceScore_FourOfAKind) {
    int sid = state_of(4,4,4,4,2);
    EXPECT_EQ(get_dice_score(sid, kRowK, tables), 46);     // 30 + 4*4 = 46
    EXPECT_EQ(get_dice_score(sid, kRowY, tables), 0);      // not Yams
}

TEST_F(ScoreTableTest, DiceScore_Straight23456) {
    int sid = state_of(2,3,4,5,6);
    EXPECT_EQ(get_dice_score(sid, kRowSTR, tables), 50);   // 2-3-4-5-6
}

TEST_F(ScoreTableTest, DiceScore_UnderEight) {
    // sum=8: 60 + 5*(8-8) = 60
    int sid = state_of(1,1,1,2,3);  // sum=8
    EXPECT_EQ(get_dice_score(sid, kRowU8, tables), 60);
    // sum=5: 60 + 5*(8-5) = 75
    int sid2 = state_of(1,1,1,1,1);
    EXPECT_EQ(get_dice_score(sid2, kRowU8, tables), 75);
}

// ---------------------------------------------------------------------------
// Filtered scores
// ---------------------------------------------------------------------------
TEST_F(ScoreTableTest, FilteredScores_6s_Threshold0) {
    int count = 0;
    const int8_t* scores = get_filtered_scores(kRow6s, 0, count, tables);
    ASSERT_EQ(count, 5);  // {6,12,18,24,30}
    EXPECT_EQ(scores[0], 6);
    EXPECT_EQ(scores[1], 12);
    EXPECT_EQ(scores[2], 18);
    EXPECT_EQ(scores[3], 24);
    EXPECT_EQ(scores[4], 30);
}

TEST_F(ScoreTableTest, FilteredScores_6s_Threshold13) {
    int count = 0;
    const int8_t* scores = get_filtered_scores(kRow6s, 13, count, tables);
    ASSERT_EQ(count, 3);  // {18,24,30}
    EXPECT_EQ(scores[0], 18);
    EXPECT_EQ(scores[1], 24);
    EXPECT_EQ(scores[2], 30);
}

TEST_F(ScoreTableTest, FilteredScores_6s_Threshold31) {
    int count = 0;
    get_filtered_scores(kRow6s, 31, count, tables);
    EXPECT_EQ(count, 0);  // no score >= 31 for 6s
}

TEST_F(ScoreTableTest, FilteredScores_Y_Threshold100) {
    int count = 0;
    const int8_t* scores = get_filtered_scores(kRowY, 100, count, tables);
    ASSERT_EQ(count, 1);
    EXPECT_EQ(scores[0], 100);
}

TEST_F(ScoreTableTest, FilteredScores_Y_Threshold0) {
    int count = 0;
    get_filtered_scores(kRowY, 0, count, tables);
    EXPECT_EQ(count, 6);  // {75,80,85,90,95,100}
}

// ---------------------------------------------------------------------------
// Completeness: every non-zero dice_score appears in possible_scores
// ---------------------------------------------------------------------------
TEST_F(ScoreTableTest, DiceScoreInPossibleScores) {
    for (int row = 0; row < kNumRows; ++row) {
        const SolverTables& st = tables.score_tables;
        for (int sid = 0; sid < kNumDiceStates; ++sid) {
            int s = st.dice_score[sid][row];
            if (s == 0) continue;
            // Check it's in possible_scores[row]
            bool found = false;
            for (int i = 0; i < st.possible_count[row]; ++i)
                if (st.possible_scores[row][i] == s) { found = true; break; }
            EXPECT_TRUE(found) << "score=" << s << " not in possible_scores for row=" << row;
        }
    }
}

// ---------------------------------------------------------------------------
// Accessor: get_dice_score consistent with direct table
// ---------------------------------------------------------------------------
TEST_F(ScoreTableTest, Accessor_GetDiceScore_Consistent) {
    for (int sid = 0; sid < kNumDiceStates; sid += 20)
        for (int row = 0; row < kNumRows; ++row)
            EXPECT_EQ(get_dice_score(sid, row, tables),
                      tables.score_tables.dice_score[sid][row]);
}
