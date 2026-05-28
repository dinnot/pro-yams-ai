#include <gtest/gtest.h>

#include <cstdint>

#include "engine/board_init.h"
#include "engine/board_state.h"
#include "engine/constants.h"
#include "engine/game_context.h"
#include "engine/rng.h"
#include "solver/dp_eval.h"
#include "solver/dp_tables.h"
#include "solver/precomputed_tables.h"

// ===========================================================================
// Middle-section DP behaviour tests, exercised at the *board → EV/prob*
// boundary (build_Sc + get_middle_*). This is deliberately representation-
// independent: the inputs are a board + context and the outputs are the
// middle EV / non-scratch probability, both fixed by the game rules. It must
// survive any change to the internal middle-state encoding.
//
// Unlike dp_tables_monotonicity_test (which builds the full ~2 GB cache and is
// not registered with CTest), this fixture computes ONLY the middle table
// (~0.8 MB), so it is fast and runs in CI.
// ===========================================================================

// Free function defined in dp_tables_compute.cc; fills dp.dp_mid only.
void compute_middle_dp(DPTables& dp, const PrecomputedTables& tables);

namespace {

class MiddleDPTest : public ::testing::Test {
protected:
    static PrecomputedTables pt;
    static DPTables dp;
    static bool initialised;

    static void SetUpTestSuite() {
        if (initialised) return;
        init_precomputed_tables(pt);
        dp.dp_mid.assign(static_cast<std::size_t>(kDPNumTurns) * kDPNumVariants
                             * kDPMiddleStates,
                         DPVal{0.0f, 0.0f, 0.0f});
        compute_middle_dp(dp, pt);
        initialised = true;
    }

    // Fresh 1v1 board+context: all cells empty, no golden maxima.
    static void fresh(BoardState& board, GameContext& ctx) {
        RNG rng(123);
        init_board(board, rng);
        init_context(ctx, board);
    }

    struct MidQ { float ev; float prob; };

    // Run build_Sc for player 0 / column, then query the middle table.
    static MidQ query(const BoardState& board, const GameContext& ctx,
                      int col, int T) {
        int8_t Sc_U[6], Sc_M[3], Sc_L[5];
        int EU, EM, EL;
        build_Sc(0, col, board, ctx, Sc_U, Sc_M, Sc_L, EU, EM, EL);
        Variant v = get_variant(col);
        return {get_middle_ev(dp, v, Sc_M, T), get_middle_prob(dp, v, Sc_M, T)};
    }
};
PrecomputedTables MiddleDPTest::pt;
DPTables MiddleDPTest::dp;
bool MiddleDPTest::initialised = false;

constexpr int kT = 10;  // generous turn budget for the 2 middle cells

}  // namespace

// ---------------------------------------------------------------------------
// Baseline: an unconstrained middle section is scorable.
// ---------------------------------------------------------------------------
TEST_F(MiddleDPTest, EmptyMiddle_PositiveEvAndProb) {
    BoardState board; GameContext ctx; fresh(board, ctx);
    auto q = query(board, ctx, kColFree, kT);
    EXPECT_GT(q.ev, 0.0f);
    EXPECT_GT(q.prob, 0.0f);
    EXPECT_LE(q.prob, 1.0f + 1e-5f);
}

// ---------------------------------------------------------------------------
// Range/sanity sweep — EV >= 0 and prob in [0,1] across golden levels and T.
// ---------------------------------------------------------------------------
TEST_F(MiddleDPTest, RangeSanityAcrossGoldenAndTurns) {
    for (int g_ss : {0, 22, 25, 29}) {
        for (int T = 2; T <= 78; ++T) {
            BoardState board; GameContext ctx; fresh(board, ctx);
            ctx.golden_max[kColFree][kRowSS] = static_cast<int8_t>(g_ss);
            auto q = query(board, ctx, kColFree, T);
            EXPECT_GE(q.ev, 0.0f) << "g_ss=" << g_ss << " T=" << T;
            EXPECT_GE(q.prob, 0.0f) << "g_ss=" << g_ss << " T=" << T;
            EXPECT_LE(q.prob, 1.0f + 1e-5f) << "g_ss=" << g_ss << " T=" << T;
        }
    }
}

// ---------------------------------------------------------------------------
// A higher SS golden threshold can only make the middle harder, never easier.
// ---------------------------------------------------------------------------
TEST_F(MiddleDPTest, HigherSsGoldenDoesNotIncreaseEv) {
    auto ev_for = [&](int g_ss) {
        BoardState board; GameContext ctx; fresh(board, ctx);
        ctx.golden_max[kColFree][kRowSS] = static_cast<int8_t>(g_ss);
        return query(board, ctx, kColFree, kT).ev;
    };
    float e0 = ev_for(0), e25 = ev_for(25), e29 = ev_for(29);
    EXPECT_GE(e0 + 1e-4f, e25);
    EXPECT_GE(e25 + 1e-4f, e29);
}

// ---------------------------------------------------------------------------
// LS-beats-SS coupling: with SS already filled, a higher recorded SS forces a
// higher LS threshold (LS > max_SS), so the LS-only EV must not increase.
// ---------------------------------------------------------------------------
TEST_F(MiddleDPTest, HigherFilledSsLowersLsOnlyEv) {
    auto ev_with_filled_ss = [&](int ss_val) {
        BoardState board; GameContext ctx; fresh(board, ctx);
        board.cells[0][kColFree][kRowSS] = static_cast<int8_t>(ss_val);
        ctx.golden_max[kColFree][kRowSS] = static_cast<int8_t>(ss_val);
        return query(board, ctx, kColFree, kT).ev;  // only LS is open
    };
    // SS=22 → LS must beat 22 (>=23); SS=28 → LS must beat 28 (>=29).
    EXPECT_GT(ev_with_filled_ss(22), ev_with_filled_ss(28));
}

// ---------------------------------------------------------------------------
// Both rows mutually scratched → no middle points and zero non-scratch prob.
// ---------------------------------------------------------------------------
TEST_F(MiddleDPTest, BothScratched_ZeroEvAndProb) {
    BoardState board; GameContext ctx; fresh(board, ctx);
    ctx.ss_scratched[0][kColFree] = true;
    ctx.ls_scratched[0][kColFree] = true;
    auto q = query(board, ctx, kColFree, kT);
    EXPECT_NEAR(q.ev, 0.0f, 1e-4f);
    EXPECT_NEAR(q.prob, 0.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// SS forced scratch under a filled LS with an equal SS golden: SS must be
// >= 25 (golden) and < 25 (our LS) → impossible. The lone open middle cell
// (SS) contributes zero EV / zero non-scratch probability.
// ---------------------------------------------------------------------------
TEST_F(MiddleDPTest, SsForcedScratchUnderFilledLsEqualGolden_ZeroEv) {
    BoardState board; GameContext ctx; fresh(board, ctx);
    board.cells[0][kColFree][kRowLS] = 25;
    ctx.golden_max[kColFree][kRowLS] = 25;
    ctx.golden_max[kColFree][kRowSS] = 25;  // opponent recorded SS=25
    auto q = query(board, ctx, kColFree, kT);
    EXPECT_NEAR(q.ev, 0.0f, 1e-4f);
    EXPECT_NEAR(q.prob, 0.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// SS forced scratch under a filled LS=20 (the floor=20 edge): SS must be < 20
// but its natural floor is 20 → impossible.
// ---------------------------------------------------------------------------
TEST_F(MiddleDPTest, SsForcedScratchUnderFilledLs20_ZeroEv) {
    BoardState board; GameContext ctx; fresh(board, ctx);
    board.cells[0][kColFree][kRowLS] = 20;
    ctx.golden_max[kColFree][kRowLS] = 20;
    auto q = query(board, ctx, kColFree, kT);
    EXPECT_NEAR(q.ev, 0.0f, 1e-4f);
    EXPECT_NEAR(q.prob, 0.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// SOFT CAP (new behaviour): a filled LS bounds SS strictly from above
// (SS < LS), so a lower LS leaves a narrower legal SS band and the SS-only EV
// must be strictly smaller. SS raw is capped at 29, so LS=30 imposes no real
// constraint (band [20,29]); LS=25 → [20,24]; LS=21 → [20,20].
// ---------------------------------------------------------------------------
TEST_F(MiddleDPTest, FilledLsCapsSsEv_Monotonic) {
    auto ev_ss_under_ls = [&](int ls_val) {
        BoardState board; GameContext ctx; fresh(board, ctx);
        board.cells[0][kColFree][kRowLS] = static_cast<int8_t>(ls_val);
        ctx.golden_max[kColFree][kRowLS] = static_cast<int8_t>(ls_val);
        return query(board, ctx, kColFree, kT).ev;  // only SS open
    };
    float e21 = ev_ss_under_ls(21);  // SS band [20,20]
    float e25 = ev_ss_under_ls(25);  // SS band [20,24]
    float e30 = ev_ss_under_ls(30);  // SS band [20,29] (cap non-binding)
    EXPECT_LT(e21 + 1e-3f, e25) << "tighter cap must lower SS EV";
    EXPECT_LT(e25 + 1e-3f, e30) << "tighter cap must lower SS EV";
}

// ---------------------------------------------------------------------------
// More turns can never reduce middle EV or non-scratch probability.
// ---------------------------------------------------------------------------
TEST_F(MiddleDPTest, MoreTurnsDoNotHurt) {
    BoardState board; GameContext ctx; fresh(board, ctx);
    float prev_ev = -1.0f, prev_p = -1.0f;
    for (int T = 2; T <= 40; ++T) {
        auto q = query(board, ctx, kColFree, T);
        EXPECT_GE(q.ev,   prev_ev - 1e-3f) << "T=" << T;
        EXPECT_GE(q.prob, prev_p  - 1e-5f) << "T=" << T;
        prev_ev = q.ev; prev_p = q.prob;
    }
}
