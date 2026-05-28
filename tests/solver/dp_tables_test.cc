#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "solver/dp_eval.h"
#include "solver/dp_tables.h"
#include "solver/dp_tables_internal.h"

#include "engine/board_init.h"
#include "engine/board_state.h"
#include "engine/constants.h"
#include "engine/game_context.h"
#include "engine/rng.h"

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
        int8_t Sc[3];
        decode_middle(id, Sc);
        EXPECT_EQ(encode_middle(Sc[0], Sc[1], Sc[2]), id) << "id=" << id;
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

    int id_mid = encode_middle(0, 0, 0);
    int8_t back3[3];
    decode_middle(id_mid, back3);
    EXPECT_EQ(back3[0], 0);
    EXPECT_EQ(back3[1], 0);
    EXPECT_EQ(back3[2], 0);  // no SS cap

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
// Suite 3: MutualDestructionTest — middle-section destruction logic.
// State is {ss_min, ls_min, ss_cap}; the cap is SS's strict upper bound from a
// filled LS (0 = none, 21..29 = SS must be < cap).
// ===========================================================================
namespace {
struct Mid3 { int8_t ss, ls, cap; };
Mid3 destruct(int c, int score, int8_t pss, int8_t pls, int8_t pcap = 0) {
    Mid3 r;
    apply_middle_destruction(c, score, pss, pls, pcap, r.ss, r.ls, r.cap);
    return r;
}
}  // namespace

TEST(MutualDestructionTest, SS_Scratch_Forces_Empty_LS) {
    // c=0 (SS), score=0, both empty → SS placed (-1), LS forced scratch (31).
    auto r = destruct(0, 0, 0, 0);
    EXPECT_EQ(r.ss, -1);
    EXPECT_EQ(r.ls, 31);
    EXPECT_EQ(r.cap, 0);
}

TEST(MutualDestructionTest, LS_Scratch_Forces_Empty_SS) {
    auto r = destruct(1, 0, 0, 0);
    EXPECT_EQ(r.ss, 31);
    EXPECT_EQ(r.ls, -1);
    EXPECT_EQ(r.cap, 0);
}

TEST(MutualDestructionTest, SS_Scratch_Does_Not_Touch_Filled_LS) {
    auto r = destruct(0, 0, 0, -1);
    EXPECT_EQ(r.ss, -1);
    EXPECT_EQ(r.ls, -1);  // stays filled
}

TEST(MutualDestructionTest, SS_Score_Sets_LS_Threshold) {
    // SS scores 25 → LS threshold becomes max(prev_ls, 26) = 26.
    auto r = destruct(0, 25, 0, 0);
    EXPECT_EQ(r.ss, -1);
    EXPECT_EQ(r.ls, 26);
}

TEST(MutualDestructionTest, SS_Score_Respects_Existing_LS_Threshold) {
    auto r = destruct(0, 22, 0, 28);
    EXPECT_EQ(r.ss, -1);
    EXPECT_EQ(r.ls, 28);  // existing 28 wins over 23
}

TEST(MutualDestructionTest, SS_Score_29_Forces_LS_30) {
    auto r = destruct(0, 29, 0, 0);
    EXPECT_EQ(r.ss, -1);
    EXPECT_EQ(r.ls, 30);
}

TEST(MutualDestructionTest, SS_Score_Above_30_Forces_LS_Scratch) {
    auto r = destruct(0, 30, 0, 0);
    EXPECT_EQ(r.ss, -1);
    EXPECT_EQ(r.ls, 31);
}

TEST(MutualDestructionTest, SS_Place_Clears_Existing_Cap) {
    // SS placed (legal under an existing LS cap of 24): SS filled, cap cleared.
    auto r = destruct(0, 22, /*pss=*/0, /*pls=*/-1, /*pcap=*/24);
    EXPECT_EQ(r.ss, -1);
    EXPECT_EQ(r.ls, -1);
    EXPECT_EQ(r.cap, 0);
}

TEST(MutualDestructionTest, LS_Place_25_Caps_SS_Band) {
    // LS=25 placed, SS no_min: SS must be < 25 → cap=25 (band [20,24]).
    auto r = destruct(1, 25, 0, 0);
    EXPECT_EQ(r.ss, 0);    // min unchanged
    EXPECT_EQ(r.ls, -1);
    EXPECT_EQ(r.cap, 25);
}

TEST(MutualDestructionTest, LS_Place_21_Caps_SS_To_Just_20) {
    auto r = destruct(1, 21, 0, 0);
    EXPECT_EQ(r.cap, 21);  // SS band is exactly {20}
    EXPECT_EQ(r.ls, -1);
}

TEST(MutualDestructionTest, LS_Place_20_Forces_SS_Scratch) {
    // LS=20 placed: SS must be < 20, but its floor is 20 → impossible.
    auto r = destruct(1, 20, 0, 0);
    EXPECT_EQ(r.ss, 31);
    EXPECT_EQ(r.ls, -1);
    EXPECT_EQ(r.cap, 0);
}

TEST(MutualDestructionTest, LS_Place_30_NonBinding) {
    // LS=30 placed: SS raw maxes at 29, so SS < 30 is always true → no cap.
    auto r = destruct(1, 30, 0, 0);
    EXPECT_EQ(r.ss, 0);
    EXPECT_EQ(r.ls, -1);
    EXPECT_EQ(r.cap, 0);
}

TEST(MutualDestructionTest, LS_Place_Caps_SS_Even_With_Golden_Min) {
    // SS already needs >= 25 (golden); LS=25 placed → cap=25. Band [25,25) is
    // empty, but that is enforced by scoring (min vs cap), not by a state hack.
    auto r = destruct(1, 25, /*pss=*/25, /*pls=*/0);
    EXPECT_EQ(r.ss, 25);   // min preserved
    EXPECT_EQ(r.ls, -1);
    EXPECT_EQ(r.cap, 25);
}

TEST(MutualDestructionTest, LS_Place_Does_Not_Touch_Filled_SS) {
    auto r = destruct(1, 22, /*pss=*/-1, /*pls=*/0);
    EXPECT_EQ(r.ss, -1);
    EXPECT_EQ(r.ls, -1);
    EXPECT_EQ(r.cap, 0);
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

// ===========================================================================
// Suite 5: SnapGmaxTest — Golden Rule threshold must round DOWN to the
// largest representable Sc bucket <= gmax (no_min when none qualifies).
//
// The Sc value is the minimum a roll must beat to place in an open cell. The
// representable buckets per row are coarse, so a real threshold that falls
// between buckets must snap to the bucket *below* it: a threshold the column
// already clears (e.g. "beat a 5" on the 5s row, whose smallest non-zero
// score is itself 5) becomes no_min, rather than being inflated up to the
// next bucket (the old behaviour, which turned "beat 5" into "need 20").
// ===========================================================================

// Bucket sets mirror the per-row constraint values in dp_eval.cc. Index 0 is
// the "filled" sentinel (-1), index 1 is no_min (0), the rest are ascending.
namespace {
constexpr int8_t kSnap5s[] = {-1, 0, 20, 25};
constexpr int8_t kSnap4s[] = {-1, 0, 16, 20};
constexpr int8_t kSnap3s[] = {-1, 0, 12, 15};
constexpr int8_t kSnap6s[] = {-1, 0, 24, 30};
}  // namespace

TEST(SnapGmaxTest, FivesRoundDown) {
    // No constraint, or any threshold the row trivially clears, -> no_min.
    EXPECT_EQ(snap_gmax(0,  kSnap5s, 4), 0);
    EXPECT_EQ(snap_gmax(5,  kSnap5s, 4), 0);
    EXPECT_EQ(snap_gmax(10, kSnap5s, 4), 0);
    EXPECT_EQ(snap_gmax(15, kSnap5s, 4), 0);
    // Only a genuine 20/25 threshold should impose a real minimum.
    EXPECT_EQ(snap_gmax(20, kSnap5s, 4), 20);
    EXPECT_EQ(snap_gmax(25, kSnap5s, 4), 25);
}

TEST(SnapGmaxTest, OtherUpperRowsRoundDown) {
    // 4s: {0,16,20}
    EXPECT_EQ(snap_gmax(4,  kSnap4s, 4), 0);
    EXPECT_EQ(snap_gmax(12, kSnap4s, 4), 0);
    EXPECT_EQ(snap_gmax(16, kSnap4s, 4), 16);
    EXPECT_EQ(snap_gmax(20, kSnap4s, 4), 20);
    // 3s: {0,12,15}
    EXPECT_EQ(snap_gmax(9,  kSnap3s, 4), 0);
    EXPECT_EQ(snap_gmax(12, kSnap3s, 4), 12);
    EXPECT_EQ(snap_gmax(15, kSnap3s, 4), 15);
    // 6s: {0,24,30}
    EXPECT_EQ(snap_gmax(18, kSnap6s, 4), 0);
    EXPECT_EQ(snap_gmax(24, kSnap6s, 4), 24);
    EXPECT_EQ(snap_gmax(30, kSnap6s, 4), 30);
}

TEST(SnapGmaxTest, BetweenBucketsRoundsToLower) {
    // A threshold strictly between two buckets snaps to the lower one.
    EXPECT_EQ(snap_gmax(22, kSnap5s, 4), 20);
    // Above the top bucket clamps to the top bucket.
    EXPECT_EQ(snap_gmax(99, kSnap5s, 4), 25);
}

// ===========================================================================
// Suite 6: BuildScMiddleTest — the SS/LS middle state {ss_min, ls_min, ss_cap}
// produced by build_Sc must honour the cross-row Golden Rule constraints:
//   * LS must beat the highest SS recorded by anyone (LS > max_SS).
//   * SS must stay strictly below this player's filled LS (SS < LS): expressed
//     as ss_cap (21..29), or ss_min=31 when even SS=20 is impossible (LS<=20).
// ===========================================================================
namespace {

// Fresh 1v1 board+context with all middle cells empty and no golden maxima.
void make_clean_middle(BoardState& board, GameContext& ctx) {
    RNG rng(7);
    init_board(board, rng);
    init_context(ctx, board);
}

// Run build_Sc for player 0 / column and return the middle state slots.
void middle_sc(const BoardState& board, const GameContext& ctx, int col,
               int8_t& ss, int8_t& ls, int8_t& cap) {
    int8_t Sc_U[6], Sc_M[3], Sc_L[5];
    int EU, EM, EL;
    build_Sc(0, col, board, ctx, Sc_U, Sc_M, Sc_L, EU, EM, EL);
    ss = Sc_M[0];
    ls = Sc_M[1];
    cap = Sc_M[2];
}

}  // namespace

TEST(BuildScMiddleTest, LS_Threshold_Beats_Opponent_SS) {
    // Opponent recorded SS=25 (global golden_max[SS]=25). Our LS is still open
    // and must beat it: min threshold = 26 (bug #1 regression guard).
    BoardState board; GameContext ctx; make_clean_middle(board, ctx);
    const int col = kColFree;
    ctx.golden_max[col][kRowSS] = 25;
    int8_t ss, ls, cap; middle_sc(board, ctx, col, ss, ls, cap);
    EXPECT_EQ(ls, 26);
}

TEST(BuildScMiddleTest, LS_Threshold_Uses_Max_Of_LS_And_SS_Gmax) {
    // golden_max[LS]=28 dominates max_ss+1=26 → threshold stays 28.
    BoardState board; GameContext ctx; make_clean_middle(board, ctx);
    const int col = kColFree;
    ctx.golden_max[col][kRowSS] = 25;
    ctx.golden_max[col][kRowLS] = 28;
    int8_t ss, ls, cap; middle_sc(board, ctx, col, ss, ls, cap);
    EXPECT_EQ(ls, 28);
}

TEST(BuildScMiddleTest, SS_ForcedBand_Under_Filled_LS_And_Equal_Golden) {
    // We placed LS=25; opponent then placed SS=25 (golden_max[SS]=25). Our SS
    // must be >= 25 (golden) and < 25 (cap) → the band is empty. ss_min snaps
    // to 25 and ss_cap=25 (min >= cap ⇒ no legal SS sum).
    BoardState board; GameContext ctx; make_clean_middle(board, ctx);
    const int col = kColFree;
    board.cells[0][col][kRowLS] = 25;
    ctx.golden_max[col][kRowLS] = 25;
    ctx.golden_max[col][kRowSS] = 25;
    int8_t ss, ls, cap; middle_sc(board, ctx, col, ss, ls, cap);
    EXPECT_EQ(ss, 25);
    EXPECT_EQ(cap, 25);  // band [25,25) empty → forced scratch
    EXPECT_EQ(ls, -1);   // LS filled
}

TEST(BuildScMiddleTest, SS_Forced_Scratch_Under_Filled_LS_20_Floor) {
    // LS=20 filled, no SS golden: SS must be < 20 but its natural floor is 20
    // → forced scratch via ss_min=31 (the floor=20 edge, cap range starts 21).
    BoardState board; GameContext ctx; make_clean_middle(board, ctx);
    const int col = kColFree;
    board.cells[0][col][kRowLS] = 20;
    ctx.golden_max[col][kRowLS] = 20;
    int8_t ss, ls, cap; middle_sc(board, ctx, col, ss, ls, cap);
    EXPECT_EQ(ss, 31);
    EXPECT_EQ(cap, 0);
}

TEST(BuildScMiddleTest, SS_Cap_Set_Under_Filled_LS) {
    // LS=25 filled, no SS golden: SS band [20,24] is now represented as
    // ss_min=0 (no_min) with ss_cap=25.
    BoardState board; GameContext ctx; make_clean_middle(board, ctx);
    const int col = kColFree;
    board.cells[0][col][kRowLS] = 25;
    ctx.golden_max[col][kRowLS] = 25;
    int8_t ss, ls, cap; middle_sc(board, ctx, col, ss, ls, cap);
    EXPECT_EQ(ss, 0);
    EXPECT_EQ(cap, 25);
    EXPECT_EQ(ls, -1);
}

TEST(BuildScMiddleTest, SS_NonBinding_Cap_Under_Filled_LS_30) {
    // LS=30 filled: SS raw maxes at 29, so SS < 30 is always satisfied → cap=0.
    BoardState board; GameContext ctx; make_clean_middle(board, ctx);
    const int col = kColFree;
    board.cells[0][col][kRowLS] = 30;
    ctx.golden_max[col][kRowLS] = 30;
    int8_t ss, ls, cap; middle_sc(board, ctx, col, ss, ls, cap);
    EXPECT_EQ(ss, 0);
    EXPECT_EQ(cap, 0);
}
