#include <gtest/gtest.h>
#include "solver/precomputed_tables.h"

#include <algorithm>
#include <unordered_set>

// Shared fixture: initialise tables once per test-suite run.
class DiceStateTest : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) { init_precomputed_tables(tables); initialised = true; }
    }
};
PrecomputedTables DiceStateTest::tables;
bool DiceStateTest::initialised = false;

// ---------------------------------------------------------------------------
// Enumeration
// ---------------------------------------------------------------------------
TEST_F(DiceStateTest, Exactly252States) {
    // Count valid entries in linear_to_id
    int valid = 0;
    for (int i = 0; i < kMaxLinearIndex; ++i)
        if (tables.linear_to_id[i] >= 0) ++valid;
    EXPECT_EQ(valid, kNumDiceStates);
}

TEST_F(DiceStateTest, FirstStateIsAllOnes) {
    // State 0 = {1,1,1,1,1} (lowest sorted dice)
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(tables.id_to_state[0][i], 1);
}

TEST_F(DiceStateTest, LastStateIsAllSixes) {
    // State 251 = {6,6,6,6,6}
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(tables.id_to_state[kNumDiceStates - 1][i], 6);
}

TEST_F(DiceStateTest, AllStatesSortedAscending) {
    for (int id = 0; id < kNumDiceStates; ++id) {
        const auto& s = tables.id_to_state[id];
        for (int i = 1; i < 5; ++i)
            EXPECT_LE(s[i-1], s[i]) << "State " << id << " not sorted at index " << i;
    }
}

TEST_F(DiceStateTest, AllStatesInRange1To6) {
    for (int id = 0; id < kNumDiceStates; ++id)
        for (int i = 0; i < 5; ++i) {
            EXPECT_GE(tables.id_to_state[id][i], 1);
            EXPECT_LE(tables.id_to_state[id][i], 6);
        }
}

TEST_F(DiceStateTest, NoDuplicateStates) {
    std::unordered_set<std::string> seen;
    for (int id = 0; id < kNumDiceStates; ++id) {
        const auto& s = tables.id_to_state[id];
        std::string key;
        for (int x : s) key += (char)x;
        EXPECT_TRUE(seen.insert(key).second) << "Duplicate state at id=" << id;
    }
}

// ---------------------------------------------------------------------------
// Linearisation round-trip
// ---------------------------------------------------------------------------
TEST_F(DiceStateTest, LinearisationRoundTrip) {
    for (int id = 0; id < kNumDiceStates; ++id) {
        const auto& s = tables.id_to_state[id];
        int lin = (s[0]-1)*1296 + (s[1]-1)*216 + (s[2]-1)*36 + (s[3]-1)*6 + (s[4]-1);
        EXPECT_EQ(tables.linear_to_id[lin], id)
            << "Round-trip failed for state id=" << id;
    }
}

TEST_F(DiceStateTest, InvalidLinearIndicesAreMinusOne) {
    // Non-sorted dice should map to -1
    // e.g., {2,1,1,1,1} is not sorted → linear index should be -1
    int lin = (2-1)*1296 + (1-1)*216 + (1-1)*36 + (1-1)*6 + (1-1);
    // This is index 1296 which corresponds to {2,1,1,1,1} — not sorted
    EXPECT_EQ(tables.linear_to_id[lin], -1);
}

// ---------------------------------------------------------------------------
// Unsorted dice lookup
// ---------------------------------------------------------------------------
TEST_F(DiceStateTest, UnsortedLookup_SameAsSorted) {
    // {3,1,4,1,5} sorted → {1,1,3,4,5} — same state ID
    int8_t unsorted[5] = {3,1,4,1,5};
    int8_t sorted[5]   = {1,1,3,4,5};
    int id_u = get_dice_state_id(unsorted, tables);
    int id_s = get_dice_state_id(sorted, tables);
    EXPECT_EQ(id_u, id_s);
    EXPECT_GE(id_u, 0);
    EXPECT_LT(id_u, kNumDiceStates);
}

TEST_F(DiceStateTest, UnsortedLookup_AllOnes) {
    int8_t d[5] = {1,1,1,1,1};
    EXPECT_EQ(get_dice_state_id(d, tables), 0);
}

TEST_F(DiceStateTest, UnsortedLookup_AllSixes) {
    int8_t d[5] = {6,6,6,6,6};
    EXPECT_EQ(get_dice_state_id(d, tables), kNumDiceStates - 1);
}

TEST_F(DiceStateTest, UnsortedLookup_KnownPermutation) {
    // {6,5,4,3,2} sorted → {2,3,4,5,6}
    int8_t unsorted[5] = {6,5,4,3,2};
    int8_t sorted[5]   = {2,3,4,5,6};
    EXPECT_EQ(get_dice_state_id(unsorted, tables), get_dice_state_id(sorted, tables));
}

TEST_F(DiceStateTest, UnsortedLookup_ResultMatchesIdToState) {
    // For each id, reverse-permute the dice and verify lookup gives back the same id.
    // Just test a sample of states.
    for (int id = 0; id < kNumDiceStates; id += 25) {
        const auto& s = tables.id_to_state[id];
        // Create a reversed array (still valid since it's a permutation of sorted)
        int8_t dice[5] = {
            (int8_t)s[4], (int8_t)s[3], (int8_t)s[2], (int8_t)s[1], (int8_t)s[0]
        };
        EXPECT_EQ(get_dice_state_id(dice, tables), id);
    }
}
