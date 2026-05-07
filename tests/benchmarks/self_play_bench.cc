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

    std::vector<float> tmp_tensor_buffer(kMaxAfterstateRequests * kTensorSize);
    for (auto _ : state) {
        solver_get_requests(game.state, game.ctx, *g_tables, game.solver_buffers);
        generate_tensor_batch(game.state.board, game.ctx,
                              game.state.board.current_player,
                              game.solver_buffers.requests,
                              game.solver_buffers.request_count,
                              *g_tables, tmp_tensor_buffer.data());
        benchmark::DoNotOptimize(tmp_tensor_buffer[0]);
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

// ---------------------------------------------------------------------------
// BM_SelfPlay_Throughput — End-to-end full orchestrator throughput.
// ---------------------------------------------------------------------------
static void RunSelfPlayThroughput(benchmark::State& state, bool use_dummy) {
    EnsureInit();

    SelfPlayConfig sp_config;
    sp_config.max_inference_batch = 4096;
    sp_config.min_games_per_batch = 2;
    sp_config.batch_timeout_ms = 5;
    sp_config.num_workers = 24;
    sp_config.num_games = 2048;
    sp_config.num_coordinators = 3;
    sp_config.debug_mode = false;

    SolverConfig sv_config;
    sv_config.placement_temperature = 0.5;
    sv_config.hold_temperature = 0.5;
    sv_config.exploration_enabled = true;
    sv_config.heuristic_weight = 0.1;
    sv_config.use_duel_margin_maximization = false;
    sv_config.use_pbrs = false;
    sv_config.debug_mode = false;

    g_engine->set_dummy_mode(use_dummy);

    SelfPlayOrchestrator orchestrator(sp_config, *g_tables, *g_engine, sv_config);
    orchestrator.start();

    // Warm-up to let queues fill and threads stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    int total_collected = 0;
    GameInstance* collect_buf[100];

    // Drain initial completed games
    while (orchestrator.completed_queue_size() > 0) {
        int n = orchestrator.collect_completed(collect_buf, 100);
        for (int i = 0; i < n; ++i) {
            if(total_collected % 10000 == 0) std::cout << "Traj len: " << collect_buf[i]->trajectory_length << "\n"; orchestrator.recycle_game(collect_buf[i], 42);
        }
    }

    for (auto _ : state) {
        // Collect batches of completed games and recycle them immediately
        int n = orchestrator.collect_completed(collect_buf, 100);
        for (int i = 0; i < n; ++i) {
            if(total_collected % 10000 == 0) std::cout << "Traj len: " << collect_buf[i]->trajectory_length << "\n"; orchestrator.recycle_game(collect_buf[i], 42);
            total_collected++; if(total_collected % 10000 == 0) std::cout << "Traj len: " << collect_buf[i]->trajectory_length << "\n";
        }
        if (n == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    orchestrator.stop();
    g_engine->set_dummy_mode(false); // Reset

    state.SetItemsProcessed(total_collected);
}

static void BM_SelfPlay_RealInference(benchmark::State& state) {
    RunSelfPlayThroughput(state, false);
}
// Run for 5 seconds to get a stable throughput measurement
BENCHMARK(BM_SelfPlay_RealInference)->Unit(benchmark::kMillisecond)->MinTime(5.0);

static void BM_SelfPlay_DummyInference(benchmark::State& state) {
    RunSelfPlayThroughput(state, true);
}
BENCHMARK(BM_SelfPlay_DummyInference)->Unit(benchmark::kMillisecond)->MinTime(5.0);
