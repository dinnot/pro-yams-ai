#include <gtest/gtest.h>
#include "self_play/training_data.h"
#include "self_play/game_instance.h"

#include <cstring>

// ---------------------------------------------------------------------------
// Helpers: build a synthetic completed game with known trajectory.
// ---------------------------------------------------------------------------
static GameInstance make_game_with_trajectory(int steps, double result_p0) {
    GameInstance game{};
    game.trajectory_length = steps;
    game.result = result_p0;

    // Alternate players. Assign synthetic values.
    for (int i = 0; i < steps; ++i) {
        game.trajectory[i].player = static_cast<int8_t>(i % 2);
        game.trajectory[i].value  = 0.6;  // constant V(s) = 0.6
        // Tensor: fill with step index for identification (irrelevant to target).
        for (int j = 0; j < kTensorSize; ++j)
            game.trajectory[i].tensor[j] = static_cast<float>(i);
    }
    return game;
}

// ---------------------------------------------------------------------------
// MC targets — must equal game outcome from each player's perspective.
// ---------------------------------------------------------------------------
TEST(TrajectoryTest, MC_TargetsMatchOutcome) {
    GameInstance game = make_game_with_trajectory(4, 1.0);  // player 0 wins

    TrainingSample samples[4];
    int n = extract_training_samples(game, TDMode::kMC, 0.0, false, 4000.0, samples, 4);

    ASSERT_EQ(n, 4);
    // Step 0, 2: player 0 → target = 1.0
    EXPECT_NEAR(samples[0].target, 1.0, 1e-9);
    EXPECT_NEAR(samples[2].target, 1.0, 1e-9);
    // Step 1, 3: player 1 → target = 1 - 1.0 = 0.0
    EXPECT_NEAR(samples[1].target, 0.0, 1e-9);
    EXPECT_NEAR(samples[3].target, 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// TD(0) targets — bootstrap from next step's value.
// ---------------------------------------------------------------------------
TEST(TrajectoryTest, TD0_Bootstrap) {
    // All V(s) = 0.6. Steps alternate players.
    GameInstance game = make_game_with_trajectory(4, 0.8);

    TrainingSample samples[4];
    int n = extract_training_samples(game, TDMode::kTD0, 0.0, false, 4000.0, samples, 4);

    ASSERT_EQ(n, 4);

    // For step 0 (player 0): next step (1) has player 1, V=0.6. target = 1 - 0.6 = 0.4
    EXPECT_NEAR(samples[0].target, 1.0 - 0.6, 1e-9);
    // For step 1: next step (2) has player 0, V=0.6. target = 1 - 0.6 = 0.4
    EXPECT_NEAR(samples[1].target, 1.0 - 0.6, 1e-9);
    // For step 2: next step (3) V=0.6. target = 1 - 0.6 = 0.4
    EXPECT_NEAR(samples[2].target, 1.0 - 0.6, 1e-9);
    // For step 3 (last, player 1): terminal. result=0.8 for p0 → target = 1 - 0.8 = 0.2
    EXPECT_NEAR(samples[3].target, 1.0 - 0.8, 1e-9);
}

// ---------------------------------------------------------------------------
// TD(λ) with λ=0 → matches TD(0).
// ---------------------------------------------------------------------------
TEST(TrajectoryTest, TDLambda_Lambda0_EqualsTD0) {
    GameInstance game = make_game_with_trajectory(4, 0.8);

    TrainingSample td0_samples[4], tdl_samples[4];
    extract_training_samples(game, TDMode::kTD0, 0.0, false, 4000.0, td0_samples, 4);
    extract_training_samples(game, TDMode::kTDLambda, 0.0, false, 4000.0, tdl_samples, 4);

    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(tdl_samples[i].target, td0_samples[i].target, 1e-9)
            << "λ=0 target mismatch at step " << i;
}

// ---------------------------------------------------------------------------
// TD(λ) with λ=1 → matches MC.
// ---------------------------------------------------------------------------
TEST(TrajectoryTest, TDLambda_Lambda1_EqualsMC) {
    GameInstance game = make_game_with_trajectory(4, 0.8);

    TrainingSample mc_samples[4], tdl_samples[4];
    extract_training_samples(game, TDMode::kMC, 1.0, false, 4000.0, mc_samples, 4);
    extract_training_samples(game, TDMode::kTDLambda, 1.0, false, 4000.0, tdl_samples, 4);

    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(tdl_samples[i].target, mc_samples[i].target, 1e-9)
            << "λ=1 target mismatch at step " << i;
}

// ---------------------------------------------------------------------------
// TD(λ=0.5) — with V=0.5 (neutral), targets lie between TD(0) and MC.
//
// When V(s)=0.5 uniformly, both same-player and opponent bootstrap values
// equal 0.5, so TDλ is a weighted avg of 0.5 and the terminal, placing it
// strictly between the TD(0) target (0.5) and the MC target.
// ---------------------------------------------------------------------------
TEST(TrajectoryTest, TDLambda_Half_BetweenTD0AndMC) {
    // V = 0.5 everywhere: all bootstrapped values are 0.5.
    // game.result = 1.0 → player 0 targets: MC=1.0, TD0=0.5; TDλ in (0.5, 1.0)
    //                    → player 1 targets: MC=0.0, TD0=0.5; TDλ in (0.0, 0.5)
    GameInstance game{};
    game.trajectory_length = 6;
    game.result = 1.0;  // player 0 wins definitively

    for (int i = 0; i < 6; ++i) {
        game.trajectory[i].player = static_cast<int8_t>(i % 2);
        game.trajectory[i].value  = 0.5;  // neutral V(s)
        for (int j = 0; j < kTensorSize; ++j)
            game.trajectory[i].tensor[j] = 0.0f;
    }

    TrainingSample td0[6], mc[6], tdl[6];
    extract_training_samples(game, TDMode::kTD0,      0.0, false, 4000.0, td0, 6);
    extract_training_samples(game, TDMode::kMC,       0.0, false, 4000.0, mc,  6);
    extract_training_samples(game, TDMode::kTDLambda, 0.5, false, 4000.0, tdl, 6);

    for (int i = 0; i < 6; ++i) {
        double lo = std::min(td0[i].target, mc[i].target);
        double hi = std::max(td0[i].target, mc[i].target);
        EXPECT_GE(tdl[i].target, lo - 1e-9)
            << "TDλ below min(TD0,MC) at step " << i;
        EXPECT_LE(tdl[i].target, hi + 1e-9)
            << "TDλ above max(TD0,MC) at step " << i;
    }
}

// ---------------------------------------------------------------------------
// All targets must lie in [0, 1].
// ---------------------------------------------------------------------------
TEST(TrajectoryTest, AllTargets_InUnitRange) {
    GameInstance game = make_game_with_trajectory(10, 0.0);

    TrainingSample samples[10];
    for (auto mode : {TDMode::kTD0, TDMode::kMC}) {
        extract_training_samples(game, mode, 0.0, false, 4000.0, samples, 10);
        for (int i = 0; i < 10; ++i) {
            EXPECT_GE(samples[i].target, 0.0) << "mode=" << (int)mode << " step=" << i;
            EXPECT_LE(samples[i].target, 1.0) << "mode=" << (int)mode << " step=" << i;
        }
    }
    extract_training_samples(game, TDMode::kTDLambda, 0.7, false, 4000.0, samples, 10);
    for (int i = 0; i < 10; ++i) {
        EXPECT_GE(samples[i].target, 0.0) << "TDλ step=" << i;
        EXPECT_LE(samples[i].target, 1.0) << "TDλ step=" << i;
    }
}

// ---------------------------------------------------------------------------
// Tensor is copied correctly.
// ---------------------------------------------------------------------------
TEST(TrajectoryTest, Tensor_CopiedToSample) {
    GameInstance game = make_game_with_trajectory(3, 0.5);

    TrainingSample samples[3];
    extract_training_samples(game, TDMode::kMC, 0.0, false, 4000.0, samples, 3);

    for (int i = 0; i < 3; ++i) {
        // Each step's tensor was filled with float(i).
        for (int j = 0; j < kTensorSize; ++j)
            EXPECT_FLOAT_EQ(samples[i].state[j], static_cast<float>(i))
                << "Tensor mismatch at step " << i << " feature " << j;
    }
}

// ---------------------------------------------------------------------------
// max_samples limit is respected.
// ---------------------------------------------------------------------------
TEST(TrajectoryTest, MaxSamples_Respected) {
    GameInstance game = make_game_with_trajectory(10, 0.5);

    TrainingSample samples[5];
    int n = extract_training_samples(game, TDMode::kMC, 0.0, false, 4000.0, samples, 5);
    EXPECT_EQ(n, 5);
}
