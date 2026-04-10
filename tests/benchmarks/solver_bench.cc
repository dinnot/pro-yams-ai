#include <benchmark/benchmark.h>
#include "solver/precomputed_tables.h"
#include "solver/solver.h"
#include "heuristic/heuristic_bot.h"
#include "engine/board_init.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"

// Initialise tables once for the whole benchmark binary.
static PrecomputedTables g_tables;

static void EnsureInit() {
    static bool done = false;
    if (!done) { init_precomputed_tables(g_tables); done = true; }
}

// ---------------------------------------------------------------------------
// BM_InitPrecomputedTables — startup cost (expected: under 1 second)
// ---------------------------------------------------------------------------
static void BM_InitPrecomputedTables(benchmark::State& state) {
    for (auto _ : state) {
        PrecomputedTables t;
        init_precomputed_tables(t);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_InitPrecomputedTables)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_GetDiceStateId — sort + linearise + lookup
// ---------------------------------------------------------------------------
static void BM_GetDiceStateId(benchmark::State& state) {
    EnsureInit();
    int8_t dice[5] = {3, 1, 5, 2, 4};
    for (auto _ : state) {
        benchmark::DoNotOptimize(get_dice_state_id(dice, g_tables));
    }
}
BENCHMARK(BM_GetDiceStateId);

// ---------------------------------------------------------------------------
// BM_TransitionIteration — iterate all transitions for a held config and
// compute a weighted sum (simulates the solver's inner expectimax loop)
// ---------------------------------------------------------------------------
static void BM_TransitionIteration(benchmark::State& state) {
    EnsureInit();
    // Use hold-none (empty held config) — largest fan-out
    int held_id = g_tables.moves[0][0];
    for (auto _ : state) {
        int count = 0;
        const Transition* tr = get_transitions(held_id, count, g_tables);
        double sum = 0.0;
        for (int i = 0; i < count; ++i)
            sum += tr[i].probability * (double)tr[i].target_state_id;
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_TransitionIteration);

// ---------------------------------------------------------------------------
// BM_GetFilteredScores — filtered score lookup
// ---------------------------------------------------------------------------
static void BM_GetFilteredScores(benchmark::State& state) {
    EnsureInit();
    for (auto _ : state) {
        for (int row = 0; row < kNumRows; ++row) {
            int count = 0;
            const int8_t* s = get_filtered_scores(row, 10, count, g_tables);
            benchmark::DoNotOptimize(s);
        }
    }
}
BENCHMARK(BM_GetFilteredScores);

// ---------------------------------------------------------------------------
// Task 05 benchmarks
// ---------------------------------------------------------------------------

static void make_mid_game_gs(GameState& gs, GameContext& ctx) {
    EnsureInit();
    RNG rng(77);
    init_game(gs, ctx, rng);
    SolverBuffers buf;
    for (int i = 0; i < 30 && !is_terminal(gs.board); ++i)
        heuristic_play_turn(gs, ctx, g_tables, buf, rng);
}

// BM_SolverGetRequests
static void BM_SolverGetRequests(benchmark::State& state) {
    EnsureInit();
    GameState gs; GameContext ctx;
    make_mid_game_gs(gs, ctx);
    SolverBuffers buffers;
    for (auto _ : state) {
        solver_get_requests(gs, ctx, g_tables, buffers);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_SolverGetRequests);

// BM_SolverResolve — full DP (V0 + V1 + V2) with pre-populated EVs
static void BM_SolverResolve(benchmark::State& state) {
    EnsureInit();
    GameState gs; GameContext ctx;
    make_mid_game_gs(gs, ctx);
    SolverBuffers buffers;
    solver_get_requests(gs, ctx, g_tables, buffers);
    for (int i = 0; i < buffers.request_count; ++i) buffers.evs[i] = (double)i;
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver_resolve_greedy(gs, ctx, g_tables, buffers));
    }
}
BENCHMARK(BM_SolverResolve);

// BM_HeuristicEvaluate — batch evaluation
static void BM_HeuristicEvaluate(benchmark::State& state) {
    EnsureInit();
    GameState gs; GameContext ctx;
    make_mid_game_gs(gs, ctx);
    SolverBuffers buffers;
    solver_get_requests(gs, ctx, g_tables, buffers);
    for (auto _ : state) {
        heuristic_evaluate(gs.board, ctx, buffers.requests, buffers.request_count, buffers.evs);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_HeuristicEvaluate);

// BM_HeuristicPlayTurn — one complete turn
static void BM_HeuristicPlayTurn(benchmark::State& state) {
    EnsureInit();
    for (auto _ : state) {
        state.PauseTiming();
        GameState gs; GameContext ctx;
        RNG rng(state.iterations());
        init_game(gs, ctx, rng);
        SolverBuffers buffers;
        state.ResumeTiming();
        heuristic_play_turn(gs, ctx, g_tables, buffers, rng);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_HeuristicPlayTurn);

// BM_HeuristicFullGame — complete game end-to-end
static void BM_HeuristicFullGame(benchmark::State& state) {
    EnsureInit();
    for (auto _ : state) {
        RNG rng(state.iterations());
        benchmark::DoNotOptimize(play_heuristic_game(rng, g_tables));
    }
}
BENCHMARK(BM_HeuristicFullGame);

// BM_SoftmaxSample — sampling with various array sizes
static void BM_SoftmaxSample(benchmark::State& state) {
    EnsureInit();
    RNG rng(42);
    double vals[32];
    for (int i = 0; i < 32; ++i) vals[i] = 0.5 + 0.01 * i;
    int count = state.range(0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(softmax_sample(vals, count, 1.0, rng));
    }
}
BENCHMARK(BM_SoftmaxSample)->Arg(2)->Arg(8)->Arg(32);
