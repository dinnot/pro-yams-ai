#include <benchmark/benchmark.h>
#include "engine/scoring.h"
#include "engine/legal_moves.h"
#include "engine/placement.h"
#include "engine/duel.h"
#include "engine/board_init.h"
#include "engine/solver_tables.h"
#include "engine/rng.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void make_mid_game(BoardState& board, GameContext& ctx) {
    RNG rng(42);
    init_board(board, rng);
    init_context(ctx, board);
    // Fill about half the board with random placements
    for (int i = 0; i < 40; ++i) {
        int p = board.current_player;
        auto& cache = ctx.legal_all[p];
        if (cache.count == 0) break;
        int idx = rng.uniform_int(0, cache.count - 1);
        auto pl = cache.placements[idx];
        int8_t dice[5] = {1,1,1,1,1};
        int score = calculate_score(pl.row, dice, p, pl.column, board, ctx);
        apply_placement(p, pl.column, pl.row, score, board, ctx);
        board.current_player = (int8_t)(1 - board.current_player);
    }
}

static void make_complete_game(BoardState& board, GameContext& ctx) {
    RNG rng(99);
    init_board(board, rng);
    init_context(ctx, board);
    while (board.cells_filled < kTotalCells) {
        int p = board.current_player;
        auto& cache = ctx.legal_all[p];
        if (cache.count == 0) break;
        int idx = rng.uniform_int(0, cache.count - 1);
        auto pl = cache.placements[idx];
        apply_placement(p, pl.column, pl.row, 0, board, ctx);
        board.current_player = (int8_t)(1 - board.current_player);
    }
}

// ---------------------------------------------------------------------------
// Task 01: placeholder benchmark
// ---------------------------------------------------------------------------
static void BM_Placeholder(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(1 + 1);
    }
}
BENCHMARK(BM_Placeholder);

// ---------------------------------------------------------------------------
// Task 02: engine benchmarks
// ---------------------------------------------------------------------------
static void BM_InitSolverTables(benchmark::State& state) {
    SolverTables t;
    for (auto _ : state) {
        init_solver_tables(t);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_InitSolverTables);

static void BM_CalculateScore(benchmark::State& state) {
    BoardState board; GameContext ctx;
    make_mid_game(board, ctx);
    int8_t dice[] = {3,3,3,1,2};
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            calculate_score(kRowFH, dice, 0, kColFree, board, ctx)
        );
    }
}
BENCHMARK(BM_CalculateScore);

static void BM_ApplyPlacement(benchmark::State& state) {
    BoardState board_orig; GameContext ctx_orig;
    make_mid_game(board_orig, ctx_orig);
    for (auto _ : state) {
        BoardState board = board_orig;
        GameContext ctx = ctx_orig;
        state.PauseTiming();
        // Find a legal placement
        int p = board.current_player;
        if (ctx.legal_all[p].count == 0) { state.ResumeTiming(); continue; }
        auto pl = ctx.legal_all[p].placements[0];
        state.ResumeTiming();
        apply_placement(p, pl.column, pl.row, 0, board, ctx);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_ApplyPlacement);

static void BM_RebuildLegalPlacements(benchmark::State& state) {
    BoardState board; GameContext ctx;
    make_mid_game(board, ctx);
    for (auto _ : state) {
        rebuild_legal_placements(0, board, ctx);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_RebuildLegalPlacements);

static void BM_ComputeDuel(benchmark::State& state) {
    BoardState board; GameContext ctx;
    make_complete_game(board, ctx);
    for (auto _ : state) {
        benchmark::DoNotOptimize(compute_duel(board, ctx));
    }
}
BENCHMARK(BM_ComputeDuel);

static void BM_BoardStateCopy(benchmark::State& state) {
    BoardState src; GameContext ctx;
    make_mid_game(src, ctx);
    for (auto _ : state) {
        BoardState dst = src;
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_BoardStateCopy);
