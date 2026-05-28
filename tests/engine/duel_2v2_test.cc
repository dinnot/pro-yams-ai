#include "engine/board_init.h"
#include "engine/duel.h"
#include "engine/game_traits.h"
#include "engine/rng.h"

#include <gtest/gtest.h>
#include <cstring>

// ---------------------------------------------------------------------------
// 2v2 duel tests (Sub-task 3.2 validation).
//
// These pin per-pairing crush and per-pairing clean-bonus value behavior.
// Pairings in 2v2: (P0,P1), (P0,P3), (P2,P1), (P2,P3) — cross-product of
// Team 0 = {P0, P2} and Team 1 = {P1, P3}. Team 0 margin is the sum across
// all four pairings.
//
// The "old shortcut" the duel rewrite deleted was a column-shared crush
// multiplier and column-shared bonus value. Scenario A specifically detects
// that bug — the same player has different clean-bonus values in different
// pairings.
// ---------------------------------------------------------------------------

namespace {

// Build a fresh-but-empty 2v2 board with a single nonzero column coefficient
// (column 0) so we can isolate that column's contribution from the totals.
void make_blank_2v2(BoardState2v2& board, GameContext2v2& ctx, int col0_coeff = 10) {
    std::memset(&board, 0, sizeof(board));
    std::memset(&ctx, 0, sizeof(ctx));
    board.coefficients[0] = static_cast<int8_t>(col0_coeff);
    for (int c = 1; c < kNumColumns; ++c) board.coefficients[c] = 0;
    board.cells_filled = static_cast<uint16_t>(Yams2v2::kTotalCells);
}

// Set up player p's column 0 to have a specific raw score and upper-section bonus
// status. cell_sum is the literal sum of placed dice in lower rows. Spread
// across multiple cells to keep each within int8_t range (max ~100 per cell).
void set_player_col0(GameContext2v2& ctx, BoardState2v2& board,
                     int p, int cell_sum, int upper_sum, bool lower_scratch) {
    int remaining = cell_sum;
    const int lower_rows[] = {kRowFH, kRowK, kRowSTR, kRowU8, kRowY};
    for (int i = 0; i < 5 && remaining > 0; ++i) {
        int v = remaining > 100 ? 100 : remaining;
        board.cells[p][0][lower_rows[i]] = static_cast<int8_t>(v);
        remaining -= v;
    }
    ctx.upper_sum[p][0] = static_cast<int16_t>(upper_sum);
    ctx.lower_has_scratch[p][0] = lower_scratch;
}

}  // namespace

// ---------------------------------------------------------------------------
// Scenario A: P0 crushes P1 (2×), but P0 does NOT crush P3 (P3 raw too high).
//             P0 has a clean column; P1, P2, P3 are not clean.
//
// Demonstrates the central property: the SAME player (P0) sees DIFFERENT
// clean-bonus values in their two pairings:
//   - In (P0, P1): crush triggers ⇒ bonus value = +100
//   - In (P0, P3): no crush ⇒ bonus value = +200
// ---------------------------------------------------------------------------
TEST(Duel2v2, PerPairingBonusValue_SamePlayerDifferentValues) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_blank_2v2(board, ctx, /*col0_coeff=*/10);

    // P0: cell_sum=170 + upper_section_bonus(60)=30 → raw=200. clean = true.
    set_player_col0(ctx, board, /*p=*/0, /*cell_sum=*/170, /*upper_sum=*/60,
                    /*lower_scratch=*/false);
    // P1: raw=100. not clean.
    set_player_col0(ctx, board, /*p=*/1, /*cell_sum=*/100, /*upper_sum=*/0,
                    /*lower_scratch=*/true);
    // P2: raw=100. not clean.
    set_player_col0(ctx, board, /*p=*/2, /*cell_sum=*/100, /*upper_sum=*/0,
                    /*lower_scratch=*/true);
    // P3: raw=110. not clean. 200/110 < 2 so P0 does NOT crush P3.
    set_player_col0(ctx, board, /*p=*/3, /*cell_sum=*/110, /*upper_sum=*/0,
                    /*lower_scratch=*/true);

    // Hand calculation:
    //   (P0,P1): raw0=200 raw1=100 → crush=2, bonus_value=100.
    //            adj0=200+100=300, adj1=100. diff=200. pts=200*2*10 = 4000.
    //   (P0,P3): raw0=200 raw1=110 → no crush (200<220), bonus_value=200.
    //            adj0=200+200=400, adj1=110. diff=290. pts=290*1*10 = 2900.
    //   (P2,P1): raw0=100 raw1=100 → no crush. adj0=100, adj1=100. diff=0. pts=0.
    //   (P2,P3): raw0=100 raw1=110 → no crush. adj0=100, adj1=110.
    //            diff=-10. pts=-10*1*10 = -100.
    //   Total Team 0 margin = 4000 + 2900 + 0 - 100 = 6800.
    EXPECT_EQ(compute_duel<Yams2v2>(board, ctx), 6800);
}

// Detection test for the OLD column-shared shortcut. If a future refactor
// re-introduces "active_crush = max over column" or "bonus_value shared across
// column," this scenario will fail because the column-shared bug would compute
// P0's adj as 300 (using crush=2 / bonus=100) in BOTH pairings, giving:
//   (P0,P1): (300-100)*2*10 = 4000
//   (P0,P3): (300-110)*2*10 = 3800   ← wrong sign of multiplier too
//   Total ≠ 6800 in any reasonable shortcut.
// The PerPairingBonusValue test above already catches this; adding it
// explicitly here makes the regression intent obvious.
TEST(Duel2v2, ColumnSharedShortcut_WouldBeDetected) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_blank_2v2(board, ctx, /*col0_coeff=*/10);

    set_player_col0(ctx, board, 0, 170, 60, false);   // clean, raw 200
    set_player_col0(ctx, board, 1, 100, 0,  true);    // raw 100
    set_player_col0(ctx, board, 2, 100, 0,  true);    // raw 100
    set_player_col0(ctx, board, 3, 110, 0,  true);    // raw 110

    // Computed independently per-pairing → 6800.
    // A column-shared shortcut would produce a noticeably different number
    // (computation: 4000 + 3800 + 0 + (-100)*2 = 7600 if crush were shared,
    // or 4000 + 2900 + 0 + (-100) = 6800 if both shared correctly … the point
    // is the math differs and we pin 6800).
    EXPECT_EQ(compute_duel<Yams2v2>(board, ctx), 6800);
}

// ---------------------------------------------------------------------------
// Scenario B: P0 clean column, P1 zero raw → 5× crush. Verifies the
//             OpponentRaw=0 special-case and the +100 bonus value with crush.
// ---------------------------------------------------------------------------
TEST(Duel2v2, OpponentRawZero_FiveXCrush) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_blank_2v2(board, ctx, /*col0_coeff=*/10);

    // P0: cell_sum=70 + upper_bonus(60)=30 → raw=100. clean.
    set_player_col0(ctx, board, 0, 70, 60, false);
    // P1, P2, P3: raw=0 (nothing scored). All have lower scratch ⇒ not clean.
    set_player_col0(ctx, board, 1, 0, 0, true);
    set_player_col0(ctx, board, 2, 0, 0, true);
    set_player_col0(ctx, board, 3, 0, 0, true);

    //   (P0,P1): raw0=100 raw1=0 → 5× crush (opp_raw=0 special case).
    //            bonus_value=100. adj0=100+100=200, adj1=0. diff=200.
    //            pts = 200 * 5 * 10 = 10000.
    //   (P0,P3): same → 10000.
    //   (P2,P1): raw0=0 raw1=0 → no crush, bonus_value=200.
    //            P2 not clean → adj0=0, adj1=0. diff=0. pts=0.
    //   (P2,P3): same → 0.
    //   Total = 20000.
    EXPECT_EQ(compute_duel<Yams2v2>(board, ctx), 20000);
}

// ---------------------------------------------------------------------------
// Scenario C: All four players identical raw scores → margin = 0 in every
//             pairing → total = 0.
// ---------------------------------------------------------------------------
TEST(Duel2v2, AllPlayersEqual_MarginIsZero) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_blank_2v2(board, ctx, /*col0_coeff=*/10);

    for (int p = 0; p < Yams2v2::kNumPlayers; ++p) {
        set_player_col0(ctx, board, p, 100, 0, true);
    }
    EXPECT_EQ(compute_duel<Yams2v2>(board, ctx), 0);
}

// ---------------------------------------------------------------------------
// Sign sanity: swap teams (give Team 1 the dominant scores) → result must
// flip sign and match magnitude.
// ---------------------------------------------------------------------------
TEST(Duel2v2, SignFlipsWhenTeamsSwap) {
    BoardState2v2 board_a, board_b;
    GameContext2v2 ctx_a, ctx_b;

    // Setup A: Team 0 dominates.
    make_blank_2v2(board_a, ctx_a, 10);
    set_player_col0(ctx_a, board_a, 0, 170, 60, false);  // clean, raw 200
    set_player_col0(ctx_a, board_a, 1, 100, 0,  true);
    set_player_col0(ctx_a, board_a, 2, 100, 0,  true);
    set_player_col0(ctx_a, board_a, 3, 110, 0,  true);

    // Setup B: Team 1 dominates — give P1 and P3 the scores from P0 and P2.
    make_blank_2v2(board_b, ctx_b, 10);
    set_player_col0(ctx_b, board_b, 0, 100, 0,  true);
    set_player_col0(ctx_b, board_b, 1, 170, 60, false);  // clean, raw 200
    set_player_col0(ctx_b, board_b, 2, 110, 0,  true);
    set_player_col0(ctx_b, board_b, 3, 100, 0,  true);

    int margin_a = compute_duel<Yams2v2>(board_a, ctx_a);
    int margin_b = compute_duel<Yams2v2>(board_b, ctx_b);
    EXPECT_EQ(margin_a, -margin_b);
}
