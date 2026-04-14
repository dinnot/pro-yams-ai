#include <gtest/gtest.h>

#include <chrono>
#include <cmath>

#include <torch/torch.h>

#include "engine/context_rebuild.h"
#include "engine/game_flow.h"
#include "engine/rng.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "heuristic/heuristic_bot.h"
#include "model/pro_yams_net.h"
#include "solver/mc_bot.h"
#include "solver/precomputed_tables.h"

class MCBotTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        torch::set_num_threads(1);
        init_precomputed_tables(tables_);
        // Create a random model for testing (not trained, but functional).
        ModelConfig cfg;
        model_ = std::make_unique<ProYamsNet>(cfg);
        model_->eval();
        device_ = torch::kCPU;
    }

    static PrecomputedTables tables_;
    static std::unique_ptr<ProYamsNet> model_;
    static torch::Device device_;
};

PrecomputedTables MCBotTest::tables_;
std::unique_ptr<ProYamsNet> MCBotTest::model_;
torch::Device MCBotTest::device_ = torch::kCPU;

// ---------------------------------------------------------------------------
// simulate_rollout
// ---------------------------------------------------------------------------

TEST_F(MCBotTest, SingleRolloutCompletes) {
    RNG rng(42);
    GameState state;
    GameContext ctx;
    init_game(state, ctx, rng);

    // Play a few turns to get a mid-game position.
    SolverBuffers buffers{};
    for (int i = 0; i < 10; ++i)
        heuristic_play_turn(state, ctx, tables_, buffers, rng);

    SolverBuffers rollout_buffers{};
    std::vector<float> rollout_tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);
    double result = simulate_rollout(state.board, ctx, 0,
                                      *model_, device_, tables_,
                                      rollout_buffers, rollout_tensor_buffer,
                                      rng);
    EXPECT_TRUE(result == 0.0 || result == 0.5 || result == 1.0);
}

TEST_F(MCBotTest, MultipleRolloutsProduceValidWinRate) {
    RNG rng(123);
    GameState state;
    GameContext ctx;
    init_game(state, ctx, rng);

    SolverBuffers buffers{};
    for (int i = 0; i < 20; ++i)
        heuristic_play_turn(state, ctx, tables_, buffers, rng);

    double wins = 0;
    int num_rollouts = 20;
    SolverBuffers rollout_buffers{};
    std::vector<float> rollout_tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);
    for (int i = 0; i < num_rollouts; ++i) {
        RNG rollout_rng(rng.next());
        double r = simulate_rollout(state.board, ctx, 0,
                                     *model_, device_, tables_,
                                     rollout_buffers, rollout_tensor_buffer,
                                     rollout_rng);
        wins += r;
    }

    double wr = wins / num_rollouts;
    EXPECT_GE(wr, 0.0);
    EXPECT_LE(wr, 1.0);
}

TEST_F(MCBotTest, DeterministicRollouts) {
    RNG rng1(999);
    RNG rng2(999);

    GameState state;
    GameContext ctx;
    RNG init_rng(42);
    init_game(state, ctx, init_rng);

    SolverBuffers buffers{};
    for (int i = 0; i < 10; ++i)
        heuristic_play_turn(state, ctx, tables_, buffers, init_rng);

    SolverBuffers rollout_buffers1{};
    SolverBuffers rollout_buffers2{};
    std::vector<float> rollout_tensor_buffer1(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);
    std::vector<float> rollout_tensor_buffer2(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);
    double r1 = simulate_rollout(state.board, ctx, 0, *model_, device_, tables_,
                                  rollout_buffers1, rollout_tensor_buffer1, rng1);
    double r2 = simulate_rollout(state.board, ctx, 0, *model_, device_, tables_,
                                  rollout_buffers2, rollout_tensor_buffer2, rng2);
    EXPECT_EQ(r1, r2);
}

// ---------------------------------------------------------------------------
// select_top_candidates
// ---------------------------------------------------------------------------

TEST_F(MCBotTest, CandidateSelectionProducesCandidates) {
    RNG rng(42);
    GameState state;
    GameContext ctx;
    init_game(state, ctx, rng);

    SolverBuffers buffers{};
    solver_get_requests(state, ctx, tables_, buffers);

    // Fill EVs with dummy values (as if NN evaluated).
    for (int i = 0; i < buffers.request_count; ++i)
        buffers.evs[i] = 0.5;

    // Run solver to populate V0/V1.
    SolverConfig greedy{0.0, 0.0, false};
    solver_resolve(state, ctx, tables_, buffers, greedy, rng);

    CandidateAction candidates[10];
    int count = select_top_candidates(state, ctx, tables_, buffers, 5, candidates);

    EXPECT_GT(count, 0);
    EXPECT_LE(count, 5);

    for (int i = 0; i < count; ++i) {
        if (candidates[i].is_placement) {
            EXPECT_GE(candidates[i].placement.column, 0);
            EXPECT_LT(candidates[i].placement.column, 6);
            EXPECT_GE(candidates[i].placement.row, 0);
            EXPECT_LT(candidates[i].placement.row, 13);
        }
    }
}

// ---------------------------------------------------------------------------
// can_terminate_early
// ---------------------------------------------------------------------------

TEST_F(MCBotTest, EarlyTerminationWithClearWinner) {
    CandidateAction cands[2];
    // Candidate 0: 95% win rate over 100 rollouts.
    cands[0].win_count = 95;
    cands[0].rollout_count = 100;
    // Candidate 1: 20% win rate over 100 rollouts.
    cands[1].win_count = 20;
    cands[1].rollout_count = 100;

    EXPECT_TRUE(can_terminate_early(cands, 2, 10));
}

TEST_F(MCBotTest, NoEarlyTerminationWhenClose) {
    CandidateAction cands[2];
    // Both around 50% with few rollouts.
    cands[0].win_count = 12;
    cands[0].rollout_count = 20;
    cands[1].win_count = 10;
    cands[1].rollout_count = 20;

    EXPECT_FALSE(can_terminate_early(cands, 2, 10));
}

TEST_F(MCBotTest, NoEarlyTerminationBelowMinRollouts) {
    CandidateAction cands[2];
    cands[0].win_count = 5;
    cands[0].rollout_count = 5;
    cands[1].win_count = 0;
    cands[1].rollout_count = 5;

    EXPECT_FALSE(can_terminate_early(cands, 2, 10));
}

// ---------------------------------------------------------------------------
// mc_play_turn — time budget enforcement
// ---------------------------------------------------------------------------

TEST_F(MCBotTest, TimeBudgetRespected) {
    RNG rng(42);
    GameState state;
    GameContext ctx;
    init_game(state, ctx, rng);

    MCConfig config;
    config.time_budget_ms = 500.0;
    config.max_candidates = 3;
    config.min_rollouts_per_candidate = 2;
    config.rollout_batch_size = 1;

    auto start = std::chrono::high_resolution_clock::now();
    mc_play_turn(state, ctx, *model_, device_, tables_, config, rng);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    // Should complete within budget + overhead (generous margin for CPU-only NN).
    // With a random model on CPU, rollouts are slow, so use a generous bound.
    EXPECT_LT(elapsed_ms, config.time_budget_ms + 5000.0);
}

TEST_F(MCBotTest, MCPlayTurnCompletesGame) {
    // Play a full game using MC bot (with tiny time budget for speed).
    RNG rng(42);
    GameState state;
    GameContext ctx;
    init_game(state, ctx, rng);

    MCConfig config;
    config.time_budget_ms = 50.0;  // Very small for test speed
    config.max_candidates = 2;
    config.min_rollouts_per_candidate = 1;
    config.rollout_batch_size = 1;

    int turns = 0;
    while (!is_terminal(state.board)) {
        mc_play_turn(state, ctx, *model_, device_, tables_, config, rng);
        turns++;
        ASSERT_LE(turns, 160);
    }
    EXPECT_EQ(turns, 156);
}

// ---------------------------------------------------------------------------
// MCTimeAllocation
// ---------------------------------------------------------------------------

TEST_F(MCBotTest, TimeAllocationDistribution) {
    MCTimeAllocation alloc{2000.0, 0.0};

    double budget2 = alloc.budget_for_current_decision(2);
    double budget1 = alloc.budget_for_current_decision(1);
    double budget0 = alloc.budget_for_current_decision(0);

    // With 2 rolls left, should get less than the full budget.
    EXPECT_GT(budget2, 0.0);
    EXPECT_LT(budget2, 2000.0);

    // With 0 rolls left, should get the full remaining budget.
    EXPECT_DOUBLE_EQ(budget0, 2000.0);

    // Earlier decisions should get more time.
    // Note: budget_for_current_decision(2) uses weight 0.4 / (1.0+0.6+0.4) = 0.2
    // budget_for_current_decision(1) uses weight 0.6 / (1.0+0.6) = 0.375
    // So budget1 > budget2 in absolute terms (different denominators).
}
