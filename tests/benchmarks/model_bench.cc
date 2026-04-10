#include <benchmark/benchmark.h>
#include "model/pro_yams_net.h"
#include "model/inference.h"
#include "model/trainer.h"
#include "engine/tensor.h"

#include <filesystem>
#include <vector>

// ---------------------------------------------------------------------------
// Shared setup: model + device
// ---------------------------------------------------------------------------
static std::shared_ptr<ProYamsNet> g_model;
static torch::Device               g_device  = torch::Device(torch::kCPU);
static std::shared_ptr<InferenceEngine> g_engine;

static void EnsureModelInit() {
    static bool done = false;
    if (!done) {
        g_device = get_device();
        ModelConfig cfg;
        g_model = std::make_shared<ProYamsNet>(cfg);
        initialize_weights(*g_model);
        g_model->to(g_device);
        g_model->eval();
        g_engine = std::make_shared<InferenceEngine>(g_model, g_device);
        done = true;
    }
}

// ---------------------------------------------------------------------------
// BM_ForwardPass (GPU if available)
// ---------------------------------------------------------------------------
static void BM_ForwardPass(benchmark::State& state) {
    EnsureModelInit();
    int bs = static_cast<int>(state.range(0));
    auto input = torch::rand({bs, kTensorSize}, torch::TensorOptions().device(g_device));
    for (auto _ : state) {
        torch::NoGradGuard no_grad;
        auto output = g_model->forward(input);
        if (g_device.type() == torch::kCUDA) {
            torch::cuda::synchronize();  // Force synchronization
        }
        benchmark::DoNotOptimize(output);
    }
    state.SetItemsProcessed(state.iterations() * bs);
}
BENCHMARK(BM_ForwardPass)->Arg(1)->Arg(64)->Arg(256)->Arg(512)->Arg(1024)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_ForwardPassCPU (always CPU)
// ---------------------------------------------------------------------------
static void BM_ForwardPassCPU(benchmark::State& state) {
    static std::shared_ptr<ProYamsNet> cpu_model;
    static bool done = false;
    if (!done) {
        ModelConfig cfg;
        cpu_model = std::make_shared<ProYamsNet>(cfg);
        initialize_weights(*cpu_model);
        cpu_model->eval();
        done = true;
    }
    int bs = static_cast<int>(state.range(0));
    auto input = torch::rand({bs, kTensorSize});
    for (auto _ : state) {
        torch::NoGradGuard no_grad;
        benchmark::DoNotOptimize(cpu_model->forward(input));
    }
    state.SetItemsProcessed(state.iterations() * bs);
}
BENCHMARK(BM_ForwardPassCPU)->Arg(1)->Arg(64)->Arg(256)->Arg(512)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_BatchInference — full pipeline: CPU float* → GPU → float* output
// ---------------------------------------------------------------------------
static void BM_BatchInference(benchmark::State& state) {
    EnsureModelInit();
    int bs = static_cast<int>(state.range(0));
    std::vector<float>  input(bs * kTensorSize);
    for (auto& v : input) v = static_cast<float>(rand()) / RAND_MAX;
    std::vector<double> output(bs);

    for (auto _ : state) {
        g_engine->batch_inference(input.data(), bs, output.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * bs);
}
BENCHMARK(BM_BatchInference)->Arg(1)->Arg(64)->Arg(256)->Arg(512)->Arg(1024)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_TrainStep — forward + backward + Adam step
// ---------------------------------------------------------------------------
static void BM_TrainStep(benchmark::State& state) {
    int bs = static_cast<int>(state.range(0));
    ModelConfig cfg;
    ModelTrainer trainer(cfg, get_device());
    std::vector<float>  states(bs * kTensorSize);
    std::vector<double> targets(bs);
    for (auto& v : states)  v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : targets) v = static_cast<double>(rand()) / RAND_MAX;

    for (auto _ : state) {
        benchmark::DoNotOptimize(
            trainer.train_step(states.data(), targets.data(), bs));
    }
    state.SetItemsProcessed(state.iterations() * bs);
}
BENCHMARK(BM_TrainStep)->Arg(32)->Arg(128)->Arg(512)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_ModelClone — cloning a model for inference swap
// ---------------------------------------------------------------------------
static void BM_ModelClone(benchmark::State& state) {
    EnsureModelInit();
    for (auto _ : state) {
        auto cloned = clone_model(*g_model, torch::Device(torch::kCPU));
        benchmark::DoNotOptimize(cloned);
    }
}
BENCHMARK(BM_ModelClone)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_CheckpointSave + BM_CheckpointLoad
// ---------------------------------------------------------------------------
static void BM_CheckpointSave(benchmark::State& state) {
    ModelConfig cfg;
    ModelTrainer trainer(cfg, torch::Device(torch::kCPU));
    std::filesystem::create_directories("/tmp/model_bench_ckpt");

    for (auto _ : state) {
        trainer.save_checkpoint("/tmp/model_bench_ckpt/bench", 0, 1.0, 0.05);
        benchmark::ClobberMemory();
    }
    std::filesystem::remove_all("/tmp/model_bench_ckpt");
}
BENCHMARK(BM_CheckpointSave)->Unit(benchmark::kMillisecond);

static void BM_CheckpointLoad(benchmark::State& state) {
    ModelConfig cfg;
    ModelTrainer trainer(cfg, torch::Device(torch::kCPU));
    std::filesystem::create_directories("/tmp/model_bench_load");
    trainer.save_checkpoint("/tmp/model_bench_load/bench", 0, 1.0, 0.05);

    for (auto _ : state) {
        state.PauseTiming();
        ModelTrainer loader(cfg, torch::Device(torch::kCPU));
        state.ResumeTiming();
        int s; double t, e;
        loader.load_checkpoint("/tmp/model_bench_load/bench", s, t, e);
        benchmark::ClobberMemory();
    }
    std::filesystem::remove_all("/tmp/model_bench_load");
}
BENCHMARK(BM_CheckpointLoad)->Unit(benchmark::kMillisecond);
