#include <benchmark/benchmark.h>
#include "engine/tensor.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"
#include "heuristic/heuristic_bot.h"
#include "engine/board_init.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"

static PrecomputedTables g_tables;

static void EnsureInit() {
    static bool done = false;
    if (!done) { init_precomputed_tables(g_tables); done = true; }
}

static void make_mid_game_state(GameState& gs, GameContext& ctx) {
    EnsureInit();
    RNG rng(42);
    init_game(gs, ctx, rng);
    SolverBuffers buf;
    for (int i = 0; i < 30 && !is_terminal(gs.board); ++i)
        heuristic_play_turn(gs, ctx, g_tables, buf, rng);
}

// ---------------------------------------------------------------------------
// BM_ProbabilityTableInit — one-time cost (called inside init_precomputed_tables)
// ---------------------------------------------------------------------------
static void BM_ProbabilityTableInit(benchmark::State& state) {
    for (auto _ : state) {
        PrecomputedTables t;
        init_precomputed_tables(t);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_ProbabilityTableInit)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_ProbabilityLookup — single lookup + compounding
// ---------------------------------------------------------------------------
static void BM_ProbabilityLookup(benchmark::State& state) {
    EnsureInit();
    for (auto _ : state) {
        double sum = 0.0;
        for (int row = 0; row < kNumRows; ++row)
            for (int t = 0; t <= 100; ++t)
                sum += g_tables.prob_tables.prob_3rolls[row][t];
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_ProbabilityLookup);

// ---------------------------------------------------------------------------
// BM_GenerateTensor — single tensor from mid-game board
// ---------------------------------------------------------------------------
static void BM_GenerateTensor(benchmark::State& state) {
    EnsureInit();
    GameState gs; GameContext ctx;
    make_mid_game_state(gs, ctx);
    float out[kTensorSize];
    for (auto _ : state) {
        generate_tensor(gs.board, ctx, gs.board.current_player, g_tables, out);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_GenerateTensor);

// ---------------------------------------------------------------------------
// BM_GenerateTensorBatch — batch generation for N afterstates
// ---------------------------------------------------------------------------
static void BM_GenerateTensorBatch(benchmark::State& state) {
    EnsureInit();
    GameState gs; GameContext ctx;
    make_mid_game_state(gs, ctx);
    gs.rolls_left = 0;
    SolverBuffers buffers;
    solver_get_requests(gs, ctx, g_tables, buffers);

    int n_req = buffers.request_count;
    int batch_size = static_cast<int>(state.range(0));
    // Clamp to actual requests if batch_size exceeds them
    if (batch_size > n_req) batch_size = n_req;

    std::vector<float> out(static_cast<size_t>(batch_size) * kTensorSize);

    for (auto _ : state) {
        generate_tensor_batch(gs.board, ctx, gs.board.current_player,
                              buffers.requests, batch_size, g_tables, out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_GenerateTensorBatch)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMicrosecond);
