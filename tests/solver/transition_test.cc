#include <gtest/gtest.h>
#include "solver/precomputed_tables.h"

#include <cmath>

// Shared fixture
class TransitionTest : public ::testing::Test {
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
PrecomputedTables TransitionTest::tables;
bool TransitionTest::initialised = false;

// ---------------------------------------------------------------------------
// Probability sums to 1.0 for every held configuration
// ---------------------------------------------------------------------------
TEST_F(TransitionTest, AllProbabilitiesSumToOne) {
    for (int hid = 0; hid < tables.num_held_configs; ++hid) {
        int count = 0;
        const Transition* tr = get_transitions(hid, count, tables);
        double sum = 0.0;
        for (int i = 0; i < count; ++i) sum += tr[i].probability;
        EXPECT_NEAR(sum, 1.0, 1e-10)
            << "Held config " << hid << " probs sum to " << sum;
    }
}

// ---------------------------------------------------------------------------
// All target_state_id values are in [0, 251]
// ---------------------------------------------------------------------------
TEST_F(TransitionTest, AllTargetStateIdsInRange) {
    for (int hid = 0; hid < tables.num_held_configs; ++hid) {
        int count = 0;
        const Transition* tr = get_transitions(hid, count, tables);
        for (int i = 0; i < count; ++i) {
            EXPECT_GE(tr[i].target_state_id, 0)
                << "Held config " << hid << " entry " << i << " has negative state_id";
            EXPECT_LT(tr[i].target_state_id, kNumDiceStates)
                << "Held config " << hid << " entry " << i << " state_id out of range";
        }
    }
}

// ---------------------------------------------------------------------------
// Hold all dice (mask 0b11111) → exactly one transition, probability 1.0,
// target = the same state.
// ---------------------------------------------------------------------------
TEST_F(TransitionTest, HoldAllDice_DeterministicSelfTransition) {
    for (int sid = 0; sid < kNumDiceStates; ++sid) {
        int held_id = tables.moves[sid][31];  // mask = 0b11111
        int count = 0;
        const Transition* tr = get_transitions(held_id, count, tables);
        ASSERT_EQ(count, 1) << "State " << sid << " hold-all has " << count << " transitions";
        EXPECT_EQ(tr[0].target_state_id, sid)
            << "State " << sid << " hold-all transitions to different state";
        EXPECT_NEAR(tr[0].probability, 1.0, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Hold none (mask 0b00000) — reroll all 5 dice → 7776 total outcomes.
// Verify the most common state ({1,2,3,4,5} = all distinct) has prob 120/7776.
// Verify any Yams state ({x,x,x,x,x}) has prob 1/7776.
// ---------------------------------------------------------------------------
TEST_F(TransitionTest, HoldNone_AllDistinctStateProb) {
    int held_id = tables.moves[0][0];  // any state, mask=0 → same empty held config
    int count = 0;
    const Transition* tr = get_transitions(held_id, count, tables);

    // Find state {1,2,3,4,5} — should have prob 120/7776
    int sid_12345 = state_of(1,2,3,4,5);
    double expected = 120.0 / 7776.0;
    bool found = false;
    for (int i = 0; i < count; ++i) {
        if (tr[i].target_state_id == sid_12345) {
            EXPECT_NEAR(tr[i].probability, expected, 1e-12);
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "State {1,2,3,4,5} not found in hold-none transitions";
}

TEST_F(TransitionTest, HoldNone_YamsStateProb) {
    int held_id = tables.moves[0][0];
    int count = 0;
    const Transition* tr = get_transitions(held_id, count, tables);

    // State {1,1,1,1,1} should have prob 1/7776
    int sid_yams = state_of(1,1,1,1,1);  // state id 0
    double expected = 1.0 / 7776.0;
    bool found = false;
    for (int i = 0; i < count; ++i) {
        if (tr[i].target_state_id == sid_yams) {
            EXPECT_NEAR(tr[i].probability, expected, 1e-14);
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "State {1,1,1,1,1} not found in hold-none transitions";
}

// ---------------------------------------------------------------------------
// Hold four sixes — one die rerolled → exactly 6 transitions, each prob 1/6.
// ---------------------------------------------------------------------------
TEST_F(TransitionTest, HoldFourSixes_SixTransitions) {
    // Use state {2,6,6,6,6}, mask 0b11110 (hold dice 1-4 = sixes).
    int sid = state_of(2,6,6,6,6);
    ASSERT_GE(sid, 0);
    // Dice in sorted order: [2,6,6,6,6]. Hold bits 1-4 → hold dice[1..4] = {6,6,6,6}.
    int held_id = tables.moves[sid][0b11110];

    int count = 0;
    const Transition* tr = get_transitions(held_id, count, tables);

    EXPECT_EQ(count, 6) << "Holding 4 sixes should give 6 transitions (1 per rerolled face)";

    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(tr[i].probability, 1.0/6.0, 1e-12);
        sum += tr[i].probability;
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);

    // Expected target states: {1,6,6,6,6},{2,6,6,6,6},{3,6,6,6,6},
    //                         {4,6,6,6,6},{5,6,6,6,6},{6,6,6,6,6}
    int expected_sids[6] = {
        state_of(1,6,6,6,6), state_of(2,6,6,6,6), state_of(3,6,6,6,6),
        state_of(4,6,6,6,6), state_of(5,6,6,6,6), state_of(6,6,6,6,6)
    };
    for (int expected : expected_sids) {
        bool found = false;
        for (int i = 0; i < count; ++i)
            if (tr[i].target_state_id == expected) { found = true; break; }
        EXPECT_TRUE(found) << "Expected state " << expected << " not in 4-sixes transitions";
    }
}

// ---------------------------------------------------------------------------
// Hold-all transitions point back to the correct state (spot check)
// ---------------------------------------------------------------------------
TEST_F(TransitionTest, HoldAllSixes_PointsToSixesState) {
    int sid = state_of(6,6,6,6,6);
    int held_id = tables.moves[sid][31];
    int count = 0;
    const Transition* tr = get_transitions(held_id, count, tables);
    ASSERT_EQ(count, 1);
    EXPECT_EQ(tr[0].target_state_id, sid);
    EXPECT_NEAR(tr[0].probability, 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Accessor: get_transitions returns pointer consistent with direct table access
// ---------------------------------------------------------------------------
TEST_F(TransitionTest, Accessor_GetTransitions_ConsistentWithTable) {
    for (int hid = 0; hid < tables.num_held_configs; hid += 50) {
        int count_acc = 0;
        const Transition* tr_acc = get_transitions(hid, count_acc, tables);
        int expected_count = tables.transition_count[hid];
        int expected_offset = tables.transition_offset[hid];
        EXPECT_EQ(count_acc, expected_count);
        EXPECT_EQ(tr_acc, tables.all_transitions.data() + expected_offset);
    }
}
