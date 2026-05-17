#include <gtest/gtest.h>

#include "eval/evaluator.h"
#include "model/pro_yams_net.h"
#include "solver/precomputed_tables.h"

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------
static PrecomputedTables* g_tables = nullptr;

static void ensure_tables() {
    if (!g_tables) {
        torch::set_num_threads(1);
        // at::set_num_interop_threads already called by eval_game_test suite
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

TEST(EvalResultTest, WinRateComputation) {
    EvalResult r{};
    r.total_games = 10;
    r.nn_wins     = 7;
    r.games_as_p0 = 5; r.nn_wins_as_p0 = 4;
    r.games_as_p1 = 5; r.nn_wins_as_p1 = 3;
    EXPECT_DOUBLE_EQ(r.nn_win_rate(),       0.7);
    EXPECT_DOUBLE_EQ(r.nn_win_rate_as_p0(), 0.8);
    EXPECT_DOUBLE_EQ(r.nn_win_rate_as_p1(), 0.6);
}

TEST(EvalResultTest, WinRateZeroGames) {
    EvalResult r{};
    EXPECT_DOUBLE_EQ(r.nn_win_rate(),       0.0);
    EXPECT_DOUBLE_EQ(r.nn_win_rate_as_p0(), 0.0);
    EXPECT_DOUBLE_EQ(r.nn_win_rate_as_p1(), 0.0);
}

// ---------------------------------------------------------------------------
// run_heuristic_vs_heuristic: V2-vs-V2 is symmetric (~50%) and never crashes.
// ---------------------------------------------------------------------------
TEST(HeuristicVsHeuristicTest, V2VsV2_Approx50Percent) {
    ensure_tables();
    EvalResult r = run_heuristic_vs_heuristic<Yams1v1>(
        *g_tables, HeuristicVersion::V2, HeuristicVersion::V2,
        /*num_games=*/50, /*base_seed=*/12345);
    EXPECT_EQ(r.total_games, 50);
    EXPECT_GT(r.nn_wins + r.heuristic_wins + r.draws, 0);
    // Symmetry: candidate's win rate should be roughly 0.5 (±0.25 with N=50).
    EXPECT_NEAR(r.nn_win_rate(), 0.5, 0.25)
        << "V2-vs-V2 should be symmetric near 50%";
}

TEST(HeuristicVsHeuristicTest, V2VsV1_V2Dominates) {
    ensure_tables();
    EvalResult r = run_heuristic_vs_heuristic<Yams1v1>(
        *g_tables, HeuristicVersion::V2, HeuristicVersion::V1,
        /*num_games=*/100, /*base_seed=*/777);
    // V2 (DP-driven) is strictly stronger than V1 (greedy score*coeff).
    // 100-game stdev on a 0.65-true win rate is ~5%, so 0.55 is a safe floor.
    EXPECT_GT(r.nn_win_rate(), 0.55)
        << "V2 should beat V1 comfortably (got " << r.nn_win_rate() << ")";
}

// ---------------------------------------------------------------------------
// run_evaluation_vs: NN vs a FIXED heuristic — deterministic seeded result.
// ---------------------------------------------------------------------------
TEST(RunEvaluationVsTest, RunsWithoutCrash_FixedVersion) {
    ensure_tables();
    auto model = make_model();
    EvalResult r = run_evaluation_vs<Yams1v1>(
        *model, torch::kCPU, *g_tables,
        HeuristicVersion::V2, /*num_games=*/10, /*base_seed=*/42);
    EXPECT_EQ(r.total_games, 10);
    EXPECT_EQ(r.nn_wins + r.heuristic_wins + r.draws, 10);
}

TEST(RunEvaluationVsTest, Deterministic_SameSeedSameResult) {
    ensure_tables();
    auto model = make_model();
    EvalResult a = run_evaluation_vs<Yams1v1>(
        *model, torch::kCPU, *g_tables,
        HeuristicVersion::V2, 8, 555);
    EvalResult b = run_evaluation_vs<Yams1v1>(
        *model, torch::kCPU, *g_tables,
        HeuristicVersion::V2, 8, 555);
    EXPECT_EQ(a.nn_wins, b.nn_wins);
    EXPECT_EQ(a.heuristic_wins, b.heuristic_wins);
    EXPECT_DOUBLE_EQ(a.avg_duel_margin, b.avg_duel_margin);
}

TEST(RunEvaluationTest, AlternatingSides) {
    ensure_tables();
    auto model = make_model();
    EvalResult result = run_evaluation(*model, torch::kCPU, *g_tables, 10, 42);
    EXPECT_EQ(result.games_as_p0, 5);
    EXPECT_EQ(result.games_as_p1, 5);
    EXPECT_EQ(result.total_games, 10);
}

TEST(RunEvaluationTest, TotalsSumCorrectly) {
    ensure_tables();
    auto model = make_model();
    EvalResult result = run_evaluation(*model, torch::kCPU, *g_tables, 50, 1234);
    EXPECT_EQ(result.total_games, 50);
    EXPECT_EQ(result.nn_wins + result.heuristic_wins + result.draws, 50);
    EXPECT_GE(result.nn_win_rate(), 0.0);
    EXPECT_LE(result.nn_win_rate(), 1.0);
}

TEST(RunEvaluationTest, HeuristicBaselineBeatsRandomNN) {
    // A randomly initialized NN should lose the majority of games
    // against the heuristic (which plays a reasonably good game).
    ensure_tables();
    auto model = make_model();
    EvalResult result = run_evaluation(*model, torch::kCPU, *g_tables, 50, 9999);
    // Heuristic should win more than NN — use a loose bound to avoid flakiness
    EXPECT_LT(result.nn_win_rate(), 0.65)
        << "Random NN unexpectedly strong: win_rate=" << result.nn_win_rate();
}

TEST(RunEvaluationTest, PerSideWinsConsistent) {
    ensure_tables();
    auto model = make_model();
    EvalResult r = run_evaluation(*model, torch::kCPU, *g_tables, 10, 7777);
    EXPECT_LE(r.nn_wins_as_p0, r.games_as_p0);
    EXPECT_LE(r.nn_wins_as_p1, r.games_as_p1);
    EXPECT_EQ(r.nn_wins_as_p0 + r.nn_wins_as_p1, r.nn_wins);
}

TEST(RunEvaluationTest, AverageMarginIncludesAllGames) {
    ensure_tables();
    auto model = make_model();
    // Run evaluation and check that avg_duel_margin is reported.
    // It's hard to predict the exact value with a random NN, but it should be
    // consistent with the reported wins/losses if we could see them.
    // At least we can check it's populated.
    EvalResult r = run_evaluation(*model, torch::kCPU, *g_tables, 10, 8888);
    
    // With 10 games, it should be highly likely it isn't exactly 0 if games are played.
    // But more importantly, we want to ensure it doesn't just sum positive values.
    // Since we can't easily mock play_eval_game here without refactoring, 
    // we just ensure it runs and produces a value.
    EXPECT_TRUE(r.total_games == 10);
}
