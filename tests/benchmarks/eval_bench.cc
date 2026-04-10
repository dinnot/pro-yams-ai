#include <benchmark/benchmark.h>

#include "eval/evaluator.h"
#include "engine/rng.h"
#include "model/pro_yams_net.h"
#include "solver/precomputed_tables.h"

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------
static PrecomputedTables* g_tables = nullptr;
static std::shared_ptr<ProYamsNet> g_model;

static void ensure_init() {
    if (!g_tables) {
        torch::set_num_threads(1);
        g_tables = new PrecomputedTables();
        init_precomputed_tables(*g_tables);

        ModelConfig cfg;
        cfg.hidden_layers = 1;
        cfg.hidden_width  = 64;
        g_model = std::make_shared<ProYamsNet>(cfg);
        g_model->eval();
    }
}

// ---------------------------------------------------------------------------
// BM_SingleEvalGame
// ---------------------------------------------------------------------------
static void BM_SingleEvalGame(benchmark::State& state) {
    ensure_init();
    for (auto _ : state) {
        RNG rng(42);
        int margin = 0;
        double result = play_eval_game(*g_model, torch::kCPU, *g_tables, 0, rng, margin);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_SingleEvalGame);

// ---------------------------------------------------------------------------
// BM_FullEvaluation
// ---------------------------------------------------------------------------
static void BM_FullEvaluation(benchmark::State& state) {
    ensure_init();
    int num_games = static_cast<int>(state.range(0));
    for (auto _ : state) {
        EvalResult result = run_evaluation(*g_model, torch::kCPU, *g_tables,
                                           num_games, 12345);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * num_games);
}
BENCHMARK(BM_FullEvaluation)->Arg(10)->Arg(50)->Arg(200);
