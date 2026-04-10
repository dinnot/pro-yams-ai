#include <benchmark/benchmark.h>
#include "training/replay_buffer.h"
#include "training/logging.h"
#include "training/metrics.h"
#include "engine/rng.h"
#include "engine/tensor.h"

#include <filesystem>

// ---------------------------------------------------------------------------
// BM_ReplayBuffer_Add — throughput of adding single samples
// ---------------------------------------------------------------------------

static void BM_ReplayBuffer_Add(benchmark::State& state) {
    ReplayBuffer buf(1'000'000);
    TrainingSample s{};
    s.target = 0.5;
    for (auto _ : state) {
        buf.add(s);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ReplayBuffer_Add);

// ---------------------------------------------------------------------------
// BM_ReplayBuffer_AddBatch — throughput of add_batch (256 samples/call)
// ---------------------------------------------------------------------------

static void BM_ReplayBuffer_AddBatch(benchmark::State& state) {
    ReplayBuffer buf(1'000'000);
    std::vector<TrainingSample> samples(256);
    for (auto& s : samples) s.target = 0.5f;

    for (auto _ : state) {
        buf.add_batch(samples.data(), 256);
    }
    state.SetItemsProcessed(state.iterations() * 256);
}
BENCHMARK(BM_ReplayBuffer_AddBatch);

// ---------------------------------------------------------------------------
// BM_ReplayBuffer_SampleBatch — throughput of sampling 512 samples
// ---------------------------------------------------------------------------

static void BM_ReplayBuffer_SampleBatch(benchmark::State& state) {
    ReplayBuffer buf(1'000'000);
    TrainingSample fill{};
    fill.target = 0.5;
    for (int i = 0; i < 500'000; ++i) buf.add(fill);

    RNG rng(42);
    std::vector<TrainingSample> out(512);

    for (auto _ : state) {
        buf.sample_batch(out.data(), 512, rng);
    }
    state.SetItemsProcessed(state.iterations() * 512);
}
BENCHMARK(BM_ReplayBuffer_SampleBatch);

// ---------------------------------------------------------------------------
// BM_ReplayBuffer_SaveLoad — save/load a 100K sample buffer
// ---------------------------------------------------------------------------

static void BM_ReplayBuffer_Save(benchmark::State& state) {
    ReplayBuffer buf(200'000);
    TrainingSample fill{};
    fill.target = 0.5;
    for (int i = 0; i < 100'000; ++i) buf.add(fill);

    const std::string path = "/tmp/bench_replay.bin";

    for (auto _ : state) {
        buf.save(path);
    }
    std::filesystem::remove(path);
    state.SetBytesProcessed(state.iterations() *
                             static_cast<int64_t>(100'000) * sizeof(TrainingSample));
}
BENCHMARK(BM_ReplayBuffer_Save);

static void BM_ReplayBuffer_Load(benchmark::State& state) {
    ReplayBuffer buf(200'000);
    TrainingSample fill{};
    fill.target = 0.5;
    for (int i = 0; i < 100'000; ++i) buf.add(fill);
    const std::string path = "/tmp/bench_replay_load.bin";
    buf.save(path);

    for (auto _ : state) {
        ReplayBuffer buf2(200'000);
        buf2.load(path);
    }
    std::filesystem::remove(path);
    state.SetBytesProcessed(state.iterations() *
                             static_cast<int64_t>(100'000) * sizeof(TrainingSample));
}
BENCHMARK(BM_ReplayBuffer_Load);

// ---------------------------------------------------------------------------
// BM_LogMetrics — CSV append throughput
// ---------------------------------------------------------------------------

static void BM_LogMetrics(benchmark::State& state) {
    const std::string path = "/tmp/bench_log_metrics.csv";
    std::filesystem::remove(path);

    TrainingMetrics m;
    m.training_step = 100;
    m.loss = 0.05;
    m.temperature = 0.9;

    for (auto _ : state) {
        log_metrics(path, m);
        ++m.training_step;
    }
    std::filesystem::remove(path);
}
BENCHMARK(BM_LogMetrics);
