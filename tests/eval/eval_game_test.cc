#include <gtest/gtest.h>

#include "eval/evaluator.h"
#include "engine/rng.h"
#include "model/pro_yams_net.h"
#include "solver/precomputed_tables.h"

#include <ATen/Parallel.h>

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------
static PrecomputedTables* g_tables = nullptr;

static void ensure_tables() {
    if (!g_tables) {
        torch::set_num_threads(1);
        at::set_num_interop_threads(1);
        g_tables = new PrecomputedTables();
        init_precomputed_tables(*g_tables);
    }
}

static std::shared_ptr<ProYamsNet> make_model() {
    ModelConfig cfg;
    cfg.hidden_layers = 1;
    cfg.hidden_width  = 32;
    auto net = std::make_shared<ProYamsNet>(cfg);
    net->eval();
    return net;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(EvalGameTest, NNAsP0Completes) {
    ensure_tables();
    auto model = make_model();
    RNG rng(42);
    int duel_margin = 0;
    double result = play_eval_game(*model, torch::kCPU, *g_tables, 0, rng, duel_margin);
    EXPECT_TRUE(result == 0.0 || result == 0.5 || result == 1.0);
}

TEST(EvalGameTest, NNAsP1Completes) {
    ensure_tables();
    auto model = make_model();
    RNG rng(99);
    int duel_margin = 0;
    double result = play_eval_game(*model, torch::kCPU, *g_tables, 1, rng, duel_margin);
    EXPECT_TRUE(result == 0.0 || result == 0.5 || result == 1.0);
}

TEST(EvalGameTest, DeterministicWithSameSeed) {
    ensure_tables();
    auto model = make_model();

    RNG rng1(12345);
    int margin1 = 0;
    double result1 = play_eval_game(*model, torch::kCPU, *g_tables, 0, rng1, margin1);

    RNG rng2(12345);
    int margin2 = 0;
    double result2 = play_eval_game(*model, torch::kCPU, *g_tables, 0, rng2, margin2);

    EXPECT_EQ(result1, result2);
    EXPECT_EQ(margin1, margin2);
}

TEST(EvalGameTest, DuelMarginSignConsistency) {
    ensure_tables();
    auto model = make_model();

    // Play enough games to get a win and a loss
    for (int seed = 0; seed < 20; ++seed) {
        RNG rng(static_cast<uint64_t>(seed));
        int margin = 0;
        double result = play_eval_game(*model, torch::kCPU, *g_tables, 0, rng, margin);
        if (result == 1.0) {
            EXPECT_GT(margin, 0) << "NN won but margin non-positive, seed=" << seed;
        } else if (result == 0.0) {
            EXPECT_LT(margin, 0) << "Heuristic won but margin non-negative, seed=" << seed;
        }
    }
}
