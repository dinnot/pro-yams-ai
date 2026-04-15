#include <benchmark/benchmark.h>
#include "self_play/game_instance.h"
#include "self_play/game_queues.h"
#include "self_play/orchestrator.h"
#include "self_play/training_data.h"
#include "solver/precomputed_tables.h"
#include "model/inference.h"
#include "model/pro_yams_net.h"
#include "engine/game_flow.h"
#include "engine/tensor.h"

#include <atomic>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Shared globals
// ---------------------------------------------------------------------------
static PrecomputedTables* g_tables = nullptr;
static std::shared_ptr<InferenceEngine> g_engine;

static void EnsureInit() {
    static bool done = false;
    if (!done) {
        g_tables = new PrecomputedTables();
        init_precomputed_tables(*g_tables);

        ModelConfig cfg;
        auto model = std::make_shared<ProYamsNet>(cfg);
        initialize_weights(*model);
        model->eval();
        torch::Device device = get_device();
        model->to(device);
        g_engine = std::make_shared<InferenceEngine>(model, device);
        done = true;
    }
}

// ---------------------------------------------------------------------------
// BM_WorkerIteration — one kNeedRequests cycle (solver_get_requests + tensor batch)
// ---------------------------------------------------------------------------
static void BM_WorkerIteration(benchmark::State& state) {
    EnsureInit();
    GameInstance game;
    game.rng = RNG(42);
    init_game(game.state, game.ctx, game.rng);
    game.phase = GamePhase::kNeedRequests;

    for (auto _ : state) {
        solver_get_requests(game.state, game.ctx, *g_tables, game.solver_buffers);
        generate_tensor_batch(game.state.board, game.ctx,
                              game.state.board.current_player,
                              game.solver_buffers.requests,
                              game.solver_buffers.request_count,
                              *g_tables, game.tensor_buffer);
        benchmark::DoNotOptimize(game.tensor_buffer[0]);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WorkerIteration)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_TrainingDataExtraction — extract samples from a completed trajectory.
// ---------------------------------------------------------------------------
static void BM_TrainingDataExtraction(benchmark::State& state) {
    // Build a synthetic completed game.
    GameInstance game{};
    game.trajectory_length = GameInstance::kMaxTrajectorySteps;
    game.result = 1.0;
    for (int i = 0; i < game.trajectory_length; ++i) {
        game.trajectory[i].player = static_cast<int8_t>(i % 2);
        game.trajectory[i].value  = 0.5;
    }

    TrainingSample samples[GameInstance::kMaxTrajectorySteps];
    TDMode mode = static_cast<TDMode>(state.range(0));
    double lambda = 0.9;

    for (auto _ : state) {
        int n = extract_training_samples(game, mode, lambda, false, 4000.0,
                                          false, samples, GameInstance::kMaxTrajectorySteps);
        benchmark::DoNotOptimize(n);
    }
    state.SetItemsProcessed(state.iterations() * game.trajectory_length);
}
BENCHMARK(BM_TrainingDataExtraction)
    ->Arg((int)TDMode::kTD0)
    ->Arg((int)TDMode::kTDLambda)
    ->Arg((int)TDMode::kMC)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_QueueThroughput — push/pop under contention.
// ---------------------------------------------------------------------------
static void BM_QueueThroughput(benchmark::State& state) {
    GameQueue q;
    GameInstance dummy;

    for (auto _ : state) {
        q.push(&dummy);
        GameInstance* g = q.try_pop();
        benchmark::DoNotOptimize(g);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_QueueThroughput)->Unit(benchmark::kNanosecond);
