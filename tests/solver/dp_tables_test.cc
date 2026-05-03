#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "solver/dp_tables.h"
#include "solver/dp_tables_internal.h"

// ===========================================================================
// Suite 1: DPEncoderTest — encoder/decoder round-trip + clamping
// ===========================================================================

TEST(DPEncoderTest, ClampInvalidUpper) {
    // Sc[0]=7 (invalid for 1s row) must clamp to index 1 (constraint 0).
    int8_t sc_invalid[6] = {7, 0, 0, 0, 0, 0};
    int8_t sc_zero   [6] = {0, 0, 0, 0, 0, 0};
    EXPECT_EQ(encode_upper(sc_invalid), encode_upper(sc_zero));
}

TEST(DPEncoderTest, ClampInvalidLowerQ) {
    // Q (cell 2) value 45 must clamp to index 1 (constraint 0).
    int8_t sc_invalid[5] = {0, 0, 45, 0, 0};
    int8_t sc_zero   [5] = {0, 0,  0, 0, 0};
    EXPECT_EQ(encode_lower(sc_invalid), encode_lower(sc_zero));
}

TEST(DPEncoderTest, RoundTripUpper) {
    for (int id = 0; id < kDPUpperStates; ++id) {
        int8_t Sc[6];
        decode_upper(id, Sc);
        EXPECT_EQ(encode_upper(Sc), id) << "id=" << id;
    }
}

TEST(DPEncoderTest, RoundTripMiddle) {
    for (int id = 0; id < kDPMiddleStates; ++id) {
        int8_t Sc[2];
        decode_middle(id, Sc);
        EXPECT_EQ(encode_middle(Sc[0], Sc[1]), id) << "id=" << id;
    }
}

TEST(DPEncoderTest, RoundTripLower) {
    for (int id = 0; id < kDPLowerStates; ++id) {
        int8_t Sc[5];
        decode_lower(id, Sc);
        EXPECT_EQ(encode_lower(Sc), id) << "id=" << id;
    }
}

TEST(DPEncoderTest, EmptyStatesAtZero) {
    // The "all empty, no constraints" state should be at id where every cell
    // index is 1 (the "0" constraint bucket).
    int8_t Sc6[6] = {0, 0, 0, 0, 0, 0};
    int id6 = encode_upper(Sc6);
    int8_t back6[6];
    decode_upper(id6, back6);
    for (int i = 0; i < 6; ++i) EXPECT_EQ(back6[i], 0);

    int id_mid = encode_middle(0, 0);
    int8_t back2[2];
    decode_middle(id_mid, back2);
    EXPECT_EQ(back2[0], 0);
    EXPECT_EQ(back2[1], 0);

    int8_t Sc5[5] = {0, 0, 0, 0, 0};
    int id5 = encode_lower(Sc5);
    int8_t back5[5];
    decode_lower(id5, back5);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(back5[i], 0);
}

// ===========================================================================
// Suite 2: VariantLogicTest — get_valid_cells per variant
// ===========================================================================

TEST(VariantLogicTest, FreeReturnsAllUnfilled) {
    int8_t Sc[6] = {-1, 0, -1, 4, 5, -1};
    auto cells = get_valid_cells(Variant::FREE, Sc, 6);
    ASSERT_EQ(cells.size(), 3u);
    EXPECT_EQ(cells[0], 1);
    EXPECT_EQ(cells[1], 3);
    EXPECT_EQ(cells[2], 4);
}

TEST(VariantLogicTest, DownReturnsLowestUnfilled) {
    int8_t Sc[6] = {-1, -1, 12, 15, -1, -1};
    auto cells = get_valid_cells(Variant::DOWN, Sc, 6);
    ASSERT_EQ(cells.size(), 1u);
    EXPECT_EQ(cells[0], 2);
}

TEST(VariantLogicTest, UpReturnsHighestUnfilled) {
    int8_t Sc[6] = {-1, -1, 12, 15, -1, -1};
    auto cells = get_valid_cells(Variant::UP, Sc, 6);
    ASSERT_EQ(cells.size(), 1u);
    EXPECT_EQ(cells[0], 3);
}

TEST(VariantLogicTest, UpDownAdjacency) {
    // Sc = {-1, 0, -1, 4, 5, -1}: cells 0, 2, 5 filled; cells 1, 3, 4 unfilled.
    // Cell 1 adj to 0 (filled) → valid
    // Cell 3 adj to 2 (filled) → valid (and adj to 4 unfilled, doesn't matter)
    // Cell 4 adj to 5 (filled) → valid
    // Plan example expects {1, 4} only. Let me re-check: with Sc={-1,0,-1,4,5,-1},
    // cell 3 is unfilled (Sc[3]=4 != -1), cell 4 unfilled (Sc[4]=5 != -1).
    // Cell 3 adj to filled cells? Sc[2]=-1 yes. So cell 3 is also valid.
    int8_t Sc[6] = {-1, 0, -1, 4, 5, -1};
    auto cells = get_valid_cells(Variant::UPDOWN, Sc, 6);
    // Per UPDOWN spec: unfilled cells adjacent to filled cells.
    // cell 1: adj 0 (-1, filled) → valid
    // cell 3: adj 2 (-1, filled) → valid
    // cell 4: adj 5 (-1, filled) → valid
    ASSERT_EQ(cells.size(), 3u);
    EXPECT_EQ(cells[0], 1);
    EXPECT_EQ(cells[1], 3);
    EXPECT_EQ(cells[2], 4);
}

TEST(VariantLogicTest, UpDownEmptySection) {
    // No cell filled → can place anywhere (start position).
    int8_t Sc[6] = {0, 0, 0, 0, 0, 0};
    auto cells = get_valid_cells(Variant::UPDOWN, Sc, 6);
    ASSERT_EQ(cells.size(), 6u);
    for (int i = 0; i < 6; ++i) EXPECT_EQ(cells[i], i);
}

TEST(VariantLogicTest, UpDownAllFilled) {
    // All filled → nothing to place.
    int8_t Sc[6] = {-1, -1, -1, -1, -1, -1};
    auto cells = get_valid_cells(Variant::UPDOWN, Sc, 6);
    EXPECT_EQ(cells.size(), 0u);
}

TEST(VariantLogicTest, TurboMatchesFree) {
    int8_t Sc[6] = {-1, 0, -1, 4, 5, -1};
    auto cf = get_valid_cells(Variant::FREE,  Sc, 6);
    auto ct = get_valid_cells(Variant::TURBO, Sc, 6);
    EXPECT_EQ(cf, ct);
}

// ===========================================================================
// Suite 3: MutualDestructionTest — middle-section destruction logic
// ===========================================================================

TEST(MutualDestructionTest, SS_Scratch_Forces_Empty_LS) {
    // c=0 (SS), score=0, prev_ss=0, prev_ls=0 (both empty).
    // Expected: next_ss=-1 (placed), next_ls=31 (forced scratch — still empty).
    int8_t next_ss, next_ls;
    apply_middle_destruction(0, 0, 0, 0, next_ss, next_ls);
    EXPECT_EQ(next_ss, -1);
    EXPECT_EQ(next_ls, 31);
}

TEST(MutualDestructionTest, LS_Scratch_Forces_Empty_SS) {
    int8_t next_ss, next_ls;
    apply_middle_destruction(1, 0, 0, 0, next_ss, next_ls);
    EXPECT_EQ(next_ss, 31);
    EXPECT_EQ(next_ls, -1);
}

TEST(MutualDestructionTest, SS_Scratch_Does_Not_Touch_Filled_LS) {
    // LS already filled (-1). Don't change it.
    int8_t next_ss, next_ls;
    apply_middle_destruction(0, 0, 0, -1, next_ss, next_ls);
    EXPECT_EQ(next_ss, -1);
    EXPECT_EQ(next_ls, -1);  // stays filled
}

TEST(MutualDestructionTest, SS_Score_Sets_LS_Threshold) {
    // SS scores 25 → LS threshold becomes max(prev_ls, 26) = 26.
    int8_t next_ss, next_ls;
    apply_middle_destruction(0, 25, 0, 0, next_ss, next_ls);
    EXPECT_EQ(next_ss, -1);
    EXPECT_EQ(next_ls, 26);
}

TEST(MutualDestructionTest, SS_Score_Respects_Existing_LS_Threshold) {
    int8_t next_ss, next_ls;
    apply_middle_destruction(0, 22, 0, 28, next_ss, next_ls);
    EXPECT_EQ(next_ss, -1);
    EXPECT_EQ(next_ls, 28);  // existing 28 wins over 23
}

TEST(MutualDestructionTest, SS_Score_29_Forces_LS_30) {
    // SS=29 → LS threshold 30 (encoder supports it).
    int8_t next_ss, next_ls;
    apply_middle_destruction(0, 29, 0, 0, next_ss, next_ls);
    EXPECT_EQ(next_ss, -1);
    EXPECT_EQ(next_ls, 30);
}

TEST(MutualDestructionTest, SS_Score_Above_30_Forces_LS_Scratch) {
    // SS=30 (theoretical) → score+1 = 31 > 30 → LS forced scratch.
    int8_t next_ss, next_ls;
    apply_middle_destruction(0, 30, 0, 0, next_ss, next_ls);
    EXPECT_EQ(next_ss, -1);
    EXPECT_EQ(next_ls, 31);
}

TEST(MutualDestructionTest, LS_Place_Does_Not_Update_SS) {
    int8_t next_ss, next_ls;
    apply_middle_destruction(1, 25, 0, 0, next_ss, next_ls);
    EXPECT_EQ(next_ss, 0);   // SS untouched
    EXPECT_EQ(next_ls, -1);
}

// ===========================================================================
// Suite 4: UpperBonusTest
// ===========================================================================

TEST(UpperBonusTest, Tiers) {
    EXPECT_EQ(upper_bonus(0),    0);
    EXPECT_EQ(upper_bonus(59),   0);
    EXPECT_EQ(upper_bonus(60),  30);
    EXPECT_EQ(upper_bonus(69),  30);
    EXPECT_EQ(upper_bonus(70),  50);
    EXPECT_EQ(upper_bonus(80), 100);
    EXPECT_EQ(upper_bonus(90), 200);
    EXPECT_EQ(upper_bonus(100), 500);
    EXPECT_EQ(upper_bonus(105), 500);
}
