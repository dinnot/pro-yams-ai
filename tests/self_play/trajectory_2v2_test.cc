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
// Now that GameInstanceT<Yams2v2> exists (Task 7), the test uses real 2v2
// instances. The trajectory field has 312 slots, more than enough for the
// hand-crafted scenarios below.
// ---------------------------------------------------------------------------

namespace {

constexpr int kT2v2 = Yams2v2::kTensorSize;

// Build a synthetic 2v2-style trajectory: turn order clockwise A→B→C→D.
// Each step gets the supplied value. Caller specifies game.result.
GameInstance2v2 make_2v2_trajectory(int steps, double result_p0,
                                     double value_per_step) {
    GameInstance2v2 game{};
    game.trajectory_length = steps;
    game.result = result_p0;
    game.final_duel_margin = 0;
    for (int i = 0; i < steps; ++i) {
        game.trajectory[i].player = static_cast<int8_t>(i % 4);
        game.trajectory[i].value  = value_per_step;
        game.trajectory[i].is_exploratory = false;
        for (int j = 0; j < kT2v2; ++j)
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
    GameInstance2v2 game = make_2v2_trajectory(/*steps=*/8, /*result_p0=*/1.0, /*value=*/0.5);

    TrainingSample2v2 samples[8];
    int n = extract_training_samples<Yams2v2>(
        game, TDMode::kMC, /*td_lambda=*/0.0,
        /*use_margin=*/false, /*margin_scale=*/4000.0, /*use_pbrs=*/false,
        samples, 8);
    ASSERT_EQ(n, 8);

    EXPECT_NEAR(samples[0].target, 1.0, 1e-9) << "P0 (Team 0) gets 1.0";
    EXPECT_NEAR(samples[2].target, 1.0, 1e-9) << "P2 (Team 0) gets 1.0";
    EXPECT_NEAR(samples[4].target, 1.0, 1e-9);
    EXPECT_NEAR(samples[6].target, 1.0, 1e-9);
    EXPECT_NEAR(samples[1].target, 0.0, 1e-9) << "P1 (Team 1) gets 0.0";
    EXPECT_NEAR(samples[3].target, 0.0, 1e-9) << "P3 (Team 1) gets 0.0";
    EXPECT_NEAR(samples[5].target, 0.0, 1e-9);
    EXPECT_NEAR(samples[7].target, 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// MC mode with margin: same teammates share the same signed target.
// ---------------------------------------------------------------------------
TEST(Trajectory2v2Test, MC_Margin_TeamMatesShareTarget) {
    GameInstance2v2 game = make_2v2_trajectory(/*steps=*/4, /*result_p0=*/0.0, /*value=*/0.0);
    game.final_duel_margin = 8000;
    const double expected_t0 = std::tanh(8000.0 / 4000.0);

    TrainingSample2v2 samples[4];
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
// ---------------------------------------------------------------------------
TEST(Trajectory2v2Test, TD0_BootstrapFromOpp_Flips) {
    GameInstance2v2 game = make_2v2_trajectory(/*steps=*/4, /*result_p0=*/1.0, /*value=*/0.7);

    TrainingSample2v2 samples[4];
    int n = extract_training_samples<Yams2v2>(
        game, TDMode::kTD0, 0.0, false, 4000.0, false, samples, 4);
    ASSERT_EQ(n, 4);

    EXPECT_NEAR(samples[0].target, 1.0 - 0.7, 1e-9);
    EXPECT_NEAR(samples[1].target, 1.0 - 0.7, 1e-9);
    EXPECT_NEAR(samples[2].target, 1.0 - 0.7, 1e-9);
    EXPECT_NEAR(samples[3].target, 0.0, 1e-9);
}

TEST(Trajectory2v2Test, TD0_BootstrapFromTeammate_NoFlip) {
    GameInstance2v2 game = make_2v2_trajectory(/*steps=*/4, /*result_p0=*/1.0, /*value=*/0.7);
    game.trajectory[1].player = 2;

    TrainingSample2v2 samples[4];
    int n = extract_training_samples<Yams2v2>(
        game, TDMode::kTD0, 0.0, false, 4000.0, false, samples, 4);
    ASSERT_EQ(n, 4);

    EXPECT_NEAR(samples[0].target, 0.7, 1e-9);
}

// ---------------------------------------------------------------------------
// TD(λ) — at λ=0 it should collapse to TD(0).
// ---------------------------------------------------------------------------
TEST(Trajectory2v2Test, TDLambda_Zero_EqualsTD0) {
    GameInstance2v2 game = make_2v2_trajectory(/*steps=*/6, /*result_p0=*/0.4, /*value=*/0.55);

    TrainingSample2v2 td0[6];
    TrainingSample2v2 tdL[6];
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
// TD(λ) Watkin's cut: an exploratory future step halts the trace.
// ---------------------------------------------------------------------------
TEST(Trajectory2v2Test, TDLambda_WatkinsCutAgainstTeammate_NoFlip) {
    GameInstance2v2 game = make_2v2_trajectory(/*steps=*/6, /*result_p0=*/1.0, /*value=*/0.3);

    game.trajectory[2].is_exploratory = true;
    game.trajectory[2].value = 0.85;

    TrainingSample2v2 samples[6];
    int n = extract_training_samples<Yams2v2>(
        game, TDMode::kTDLambda, /*lambda=*/0.5,
        false, 4000.0, false, samples, 6);
    ASSERT_EQ(n, 6);

    EXPECT_NEAR(samples[0].target,
                (0.5 * (1.0 - 0.3) + 0.5 * 0.85) / 1.0,
                1e-9)
        << "Watkin's cut against teammate's exploratory value must not flip";
}
