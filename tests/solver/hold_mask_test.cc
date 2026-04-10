#include <gtest/gtest.h>
#include "solver/precomputed_tables.h"

// Shared fixture
class HoldMaskTest : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) { init_precomputed_tables(tables); initialised = true; }
    }
    // Helper: find the state_id for a sorted dice array.
    static int state_of(int a, int b, int c, int d, int e) {
        int8_t dice[5] = {(int8_t)a,(int8_t)b,(int8_t)c,(int8_t)d,(int8_t)e};
        return get_dice_state_id(dice, tables);
    }
};
PrecomputedTables HoldMaskTest::tables;
bool HoldMaskTest::initialised = false;

// ---------------------------------------------------------------------------
// Mask 0b00000 — hold nothing; all states share the same held config (empty set)
// ---------------------------------------------------------------------------
TEST_F(HoldMaskTest, MaskZero_AllStatesSameConfig) {
    int id0 = tables.moves[0][0];
    for (int s = 1; s < kNumDiceStates; ++s)
        EXPECT_EQ(tables.moves[s][0], id0)
            << "State " << s << " hold-none config differs from state 0";
}

// ---------------------------------------------------------------------------
// Mask 0b11111 — hold all; each state maps to its own unique held config
// (because holding different sorted dice gives different held vectors).
// ---------------------------------------------------------------------------
TEST_F(HoldMaskTest, MaskAllSet_UniquePerState) {
    // Each full-hold produces a held set equal to the state itself.
    // Different states have different dice → different held configs.
    for (int a = 0; a < kNumDiceStates; ++a)
        for (int b = a + 1; b < kNumDiceStates; ++b) {
            // Only check a sample to keep test fast
            if ((a * 17 + b) % 500 != 0) continue;
            EXPECT_NE(tables.moves[a][31], tables.moves[b][31])
                << "States " << a << " and " << b << " share full-hold config";
        }
}

// ---------------------------------------------------------------------------
// Holding same subset from two different states → same held config ID
// ---------------------------------------------------------------------------
TEST_F(HoldMaskTest, SameHeldValues_SameConfigId) {
    // State {1,2,3,4,5}: hold dice 0,1 (mask 0b00011) → held = {1,2}
    // State {1,2,4,5,6}: hold dice 0,1 (mask 0b00011) → held = {1,2}
    int sid_a = state_of(1,2,3,4,5);
    int sid_b = state_of(1,2,4,5,6);
    ASSERT_GE(sid_a, 0); ASSERT_GE(sid_b, 0);
    EXPECT_EQ(tables.moves[sid_a][0b00011], tables.moves[sid_b][0b00011]);
}

// ---------------------------------------------------------------------------
// Specific known held configs
// ---------------------------------------------------------------------------
TEST_F(HoldMaskTest, HoldDice01_State12345_GivesCorrectTransitions) {
    // State {1,2,3,4,5}, mask 0b00011 → held {1,2}, reroll 3 dice.
    int sid = state_of(1,2,3,4,5);
    ASSERT_GE(sid, 0);
    int held_id = tables.moves[sid][0b00011];
    int count = 0;
    const Transition* tr = get_transitions(held_id, count, tables);
    // Rerolling 3 dice → many possible outcomes, all probs > 0, sum = 1
    EXPECT_GT(count, 0);
    double sum = 0.0;
    for (int i = 0; i < count; ++i) sum += tr[i].probability;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Held config count — 462 unique sorted multisubsets of sizes 0-5 from {1-6}
// (Note: the doc says 792, but the mathematically correct value is 462.)
// ---------------------------------------------------------------------------
TEST_F(HoldMaskTest, NumHeldConfigs_Is462) {
    EXPECT_EQ(tables.num_held_configs, 462);
}

// ---------------------------------------------------------------------------
// All held config IDs are in range
// ---------------------------------------------------------------------------
TEST_F(HoldMaskTest, AllMoveEntriesInRange) {
    for (int s = 0; s < kNumDiceStates; ++s)
        for (int m = 0; m < kNumHoldMasks; ++m) {
            EXPECT_GE(tables.moves[s][m], 0);
            EXPECT_LT(tables.moves[s][m], tables.num_held_configs);
        }
}

// ---------------------------------------------------------------------------
// Accessor: get_held_config returns same value as direct table lookup
// ---------------------------------------------------------------------------
TEST_F(HoldMaskTest, Accessor_GetHeldConfig) {
    for (int s = 0; s < kNumDiceStates; s += 10)
        for (int m = 0; m < kNumHoldMasks; m += 4) {
            EXPECT_EQ(get_held_config(s, m, tables), tables.moves[s][m]);
        }
}
