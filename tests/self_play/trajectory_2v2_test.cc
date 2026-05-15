#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include "engine/game_traits.h"
#include "self_play/game_instance.h"
#include "self_play/training_data.h"

// ---------------------------------------------------------------------------
// 2v2 trajectory tests (Task 6 validation).
//
// The team-aware fix:
//   - terminal target: are_teammates(my_player, 0) ? terminal_p0 : flip
//   - TD(0)/TD(λ) bootstrap: are_teammates(future.player, my_player)
//                            ? future.value (no flip)
//                            : flip(future.value)
//
// Team assignment in Yams2v2:
//   Team 0 = {P0, P2}   Team 1 = {P1, P3}
//
// GameInstance is still 1v1-shaped (Task 7 will template it), but the
// trajectory field (156 slots) is more than enough to seat a handful of
// hand-crafted 2v2-style steps with player IDs in {0, 1, 2, 3}. We feed
// these through extract_training_samples<Yams2v2>() to exercise the
// team-aware math.
// ---------------------------------------------------------------------------

namespace {

// Build a synthetic 2v2-style trajectory: turn order clockwise A→B→C→D.
// Each step gets the supplied value. Caller specifies game.result.
GameInstance make_2v2_trajectory(int steps, double result_p0,
                                  double value_per_step) {
    GameInstance game{};
    game.trajectory_length = steps;
    game.result = result_p0;
    game.final_duel_margin = 0;
    for (int i = 0; i < steps; ++i) {
        game.trajectory[i].player = static_cast<int8_t>(i % 4);
        game.trajectory[i].value  = value_per_step;
        game.trajectory[i].is_exploratory = false;
        // Tensor contents irrelevant — fill with marker for debugging.
        for (int j = 0; j < kTensorSize; ++j)
            game.trajectory[i].tensor[j] = static_cast<float>(i);
    }
    return game;
}

}  // namespace

// ---------------------------------------------------------------------------
// MC mode: Team 0 wins (result=1.0). P0/P2 are on Team 0 (target=1.0).
// P1/P3 are on Team 1 (target=0.0). The pivotal validation from the plan.
// ---------------------------------------------------------------------------
TEST(Trajectory2v2Test, MC_TeamMatesShareTarget) {
    // 8 steps = 2 full clockwise rounds.
    GameInstance game = make_2v2_trajectory(/*steps=*/8, /*result_p0=*/1.0, /*value=*/0.5);

    TrainingSample samples[8];
    int n = extract_training_samples<Yams2v2>(
        game, TDMode::kMC, /*td_lambda=*/0.0,
        /*use_margin=*/false, /*margin_scale=*/4000.0, /*use_pbrs=*/false,
        samples, 8);
    ASSERT_EQ(n, 8);

    // Players cycle 0, 1, 2, 3, 0, 1, 2, 3.
    // P0 (samples 0, 4) and P2 (samples 2, 6): Team 0 → target 1.0.
    EXPECT_NEAR(samples[0].target, 1.0, 1e-9) << "P0 (Team 0) gets 1.0";
    EXPECT_NEAR(samples[2].target, 1.0, 1e-9) << "P2 (Team 0) gets 1.0";
    EXPECT_NEAR(samples[4].target, 1.0, 1e-9);
    EXPECT_NEAR(samples[6].target, 1.0, 1e-9);
    // P1 (samples 1, 5) and P3 (samples 3, 7): Team 1 → target 0.0.
    EXPECT_NEAR(samples[1].target, 0.0, 1e-9) << "P1 (Team 1) gets 0.0";
    EXPECT_NEAR(samples[3].target, 0.0, 1e-9) << "P3 (Team 1) gets 0.0";
    EXPECT_NEAR(samples[5].target, 0.0, 1e-9);
    EXPECT_NEAR(samples[7].target, 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// MC mode with margin: same teammates share the same signed target.
// ---------------------------------------------------------------------------
TEST(Trajectory2v2Test, MC_Margin_TeamMatesShareTarget) {
    GameInstance game = make_2v2_trajectory(/*steps=*/4, /*result_p0=*/0.0, /*value=*/0.0);
    game.final_duel_margin = 8000;  // tanh(8000/4000) = tanh(2) ≈ 0.964
    const double expected_t0 = std::tanh(8000.0 / 4000.0);

    TrainingSample samples[4];
    int n = extract_training_samples<Yams2v2>(
        game, TDMode::kMC, 0.0, /*use_margin=*/true, 4000.0, false,
        samples, 4);
    ASSERT_EQ(n, 4);

    EXPECT_NEAR(samples[0].target,  expected_t0, 1e-6) << "P0 (Team 0)";
    EXPECT_NEAR(samples[1].target, -expected_t0, 1e-6) << "P1 (Team 1)";
    EXPECT_NEAR(samples[2].target,  expected_t0, 1e-6) << "P2 (Team 0)";
    EXPECT_NEAR(samples[3].target, -expected_t0, 1e-6) << "P3 (Team 1)";
}

// ---------------------------------------------------------------------------
// TD(0): from P0's step, the next step is P1 (opp) → flip.
// From a step whose "next" happens to be a teammate (engineered trajectory),
// the bootstrap must NOT flip.
//
// For this test we override the trajectory to put a teammate AS the next
// player after step 0 (artificial: real games never have consecutive
// teammate turns, but the team-aware logic must still be correct).
// ---------------------------------------------------------------------------
TEST(Trajectory2v2Test, TD0_BootstrapFromOpp_Flips) {
    GameInstance game = make_2v2_trajectory(/*steps=*/4, /*result_p0=*/1.0, /*value=*/0.7);

    TrainingSample samples[4];
    int n = extract_training_samples<Yams2v2>(
        game, TDMode::kTD0, 0.0, false, 4000.0, false, samples, 4);
    ASSERT_EQ(n, 4);

    // Step 0 (P0). Next step is P1 (opp) → flip(0.7) = 0.3.
    EXPECT_NEAR(samples[0].target, 1.0 - 0.7, 1e-9);
    // Step 1 (P1). Next step is P2 (opp from P1's perspective) → flip(0.7) = 0.3.
    EXPECT_NEAR(samples[1].target, 1.0 - 0.7, 1e-9);
    // Step 2 (P2). Next step is P3 (opp from P2's perspective) → flip(0.7) = 0.3.
    EXPECT_NEAR(samples[2].target, 1.0 - 0.7, 1e-9);
    // Step 3 (P3) is the LAST step → falls through to terminal_for(P3).
    // P3 is on Team 1, result_p0=1.0 → target = 1 - 1.0 = 0.0.
    EXPECT_NEAR(samples[3].target, 0.0, 1e-9);
}

TEST(Trajectory2v2Test, TD0_BootstrapFromTeammate_NoFlip) {
    GameInstance game = make_2v2_trajectory(/*steps=*/4, /*result_p0=*/1.0, /*value=*/0.7);
    // Override step 1's player to P2 — same team as P0. The bootstrap from
    // step 1's value should NOT flip (teammate value already encodes my-team
    // win probability).
    game.trajectory[1].player = 2;

    TrainingSample samples[4];
    int n = extract_training_samples<Yams2v2>(
        game, TDMode::kTD0, 0.0, false, 4000.0, false, samples, 4);
    ASSERT_EQ(n, 4);

    // Step 0 (P0). Next step is P2 (teammate) → no flip → 0.7.
    EXPECT_NEAR(samples[0].target, 0.7, 1e-9);
}

// ---------------------------------------------------------------------------
// TD(λ) — at λ=0 it should collapse to TD(0). Sanity check the team-aware
// bootstrap also fires in the λ branch.
// ---------------------------------------------------------------------------
TEST(Trajectory2v2Test, TDLambda_Zero_EqualsTD0) {
    GameInstance game = make_2v2_trajectory(/*steps=*/6, /*result_p0=*/0.4, /*value=*/0.55);

    TrainingSample td0[6];
    TrainingSample tdL[6];
    int n0 = extract_training_samples<Yams2v2>(
        game, TDMode::kTD0, 0.0, false, 4000.0, false, td0, 6);
    int nL = extract_training_samples<Yams2v2>(
        game, TDMode::kTDLambda, 0.0, false, 4000.0, false, tdL, 6);
    ASSERT_EQ(n0, 6);
    ASSERT_EQ(nL, 6);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(td0[i].target, tdL[i].target, 1e-9) << "step " << i;
    }
}

// ---------------------------------------------------------------------------
// TD(λ) Watkin's cut: an exploratory future step halts the trace and uses
// that step's (team-aware) bootstrap directly. Specifically, a Watkin's cut
// against a teammate's exploratory value should NOT flip.
// ---------------------------------------------------------------------------
TEST(Trajectory2v2Test, TDLambda_WatkinsCutAgainstTeammate_NoFlip) {
    GameInstance game = make_2v2_trajectory(/*steps=*/6, /*result_p0=*/1.0, /*value=*/0.3);

    // Make step 2 (P2 = teammate of P0) an exploratory state — Watkin's cut.
    game.trajectory[2].is_exploratory = true;
    game.trajectory[2].value = 0.85;

    TrainingSample samples[6];
    int n = extract_training_samples<Yams2v2>(
        game, TDMode::kTDLambda, /*lambda=*/0.5,
        false, 4000.0, false, samples, 6);
    ASSERT_EQ(n, 6);

    // For step 0 (P0): trace walks i+1 (P1, opp, value=0.3) and i+2 (P2,
    // teammate, value=0.85, exploratory). The Watkin's cut consumes the
    // P2 step. Final target is the normalized weighted average of:
    //   step 1: (1-λ)·1 ·flip(0.3) = 0.5·1·0.7 = 0.35   (weight 0.5)
    //   step 2: λ·1·0.85           = 0.5·0.85 = 0.425    (weight 0.5)
    //   sum = 0.775, weight_sum = 1.0 → 0.775
    // Crucially: P2's value is taken AS-IS (no flip) because P0 and P2
    // are teammates.
    EXPECT_NEAR(samples[0].target,
                (0.5 * (1.0 - 0.3) + 0.5 * 0.85) / 1.0,
                1e-9)
        << "Watkin's cut against teammate's exploratory value must not flip";
}
