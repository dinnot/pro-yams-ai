#include <gtest/gtest.h>

#include "distil/replay_buffer.h"
#include "engine/game_traits.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using BufferT = DistilReplayBuffer;            // Yams1v1 specialisation
using Sample  = BufferT::Sample;
static constexpr int kT = Yams1v1::kTensorSize;

// Tag the state[0] with the target's integer value so the test can recover
// each sample's identity after random sampling.
static Sample make_sample(int id) {
    Sample s{};
    s.state[0] = static_cast<float>(id);
    s.target   = static_cast<double>(id);
    return s;
}

// ---------------------------------------------------------------------------
// Construction / initial state
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, InitiallyEmpty) {
    BufferT b(/*capacity=*/1024, /*min_to_start=*/256, /*K=*/1.0);
    EXPECT_EQ(b.size(), 0);
    EXPECT_EQ(b.total_drawn(), 0);
    EXPECT_EQ(b.capacity(), 1024);
    EXPECT_DOUBLE_EQ(b.samples_per_train(), 1.0);
}

TEST(ReplayBufferTest, AddBatchGrowsSize) {
    BufferT b(1024, 256, 1.0);
    std::vector<Sample> samples(50);
    for (int i = 0; i < 50; ++i) samples[i] = make_sample(i);
    b.add_batch(samples.data(), 50);
    EXPECT_EQ(b.size(), 50);
}

// ---------------------------------------------------------------------------
// Blocking / warm-up semantics
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, DrawBlocksUntilWarmupFilled) {
    BufferT b(1024, /*min_to_start=*/100, 1.0);
    std::vector<Sample> samples(50);
    for (int i = 0; i < 50; ++i) samples[i] = make_sample(i);
    b.add_batch(samples.data(), 50);

    std::atomic<bool> draw_done{false};
    std::vector<float>  states(32 * kT);
    std::vector<double> targets(32);
    std::thread t([&]() {
        b.draw_batch(states.data(), targets.data(), 32);
        draw_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(draw_done.load());

    std::vector<Sample> more(60);
    for (int i = 0; i < 60; ++i) more[i] = make_sample(100 + i);
    b.add_batch(more.data(), 60);

    t.join();
    EXPECT_TRUE(draw_done.load());
}

TEST(ReplayBufferTest, StopUnblocksEmptyDraw) {
    BufferT b(1024, 100, 1.0);
    std::atomic<int> result{-1};
    std::vector<float>  states(32 * kT);
    std::vector<double> targets(32);
    std::thread t([&]() {
        result.store(b.draw_batch(states.data(), targets.data(), 32));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(result.load(), -1);

    b.stop();
    t.join();
    EXPECT_EQ(result.load(), 0);
}

TEST(ReplayBufferTest, StopDrainsRemainingBelowMinToStart) {
    BufferT b(1024, /*min_to_start=*/1024, 1.0);
    std::vector<Sample> samples(50);
    for (int i = 0; i < 50; ++i) samples[i] = make_sample(i);
    b.add_batch(samples.data(), 50);

    b.stop();  // signals the partial-drain path

    std::vector<float>  states(64 * kT);
    std::vector<double> targets(64);
    int total = 0;
    while (true) {
        int n = b.draw_batch(states.data(), targets.data(), 64);
        if (n == 0) break;
        total += n;
    }
    EXPECT_EQ(total, 50);  // all 50 samples drained as final partial batch(es)
    EXPECT_EQ(b.size(), 0);
}

// ---------------------------------------------------------------------------
// Eviction rate: K=1 evicts B per draw, K=2 evicts B/2, etc.
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, KOneEvictsFullBatchPerDraw) {
    BufferT b(/*capacity=*/200, /*min_to_start=*/100, /*K=*/1.0);
    std::vector<Sample> samples(200);
    for (int i = 0; i < 200; ++i) samples[i] = make_sample(i);
    b.add_batch(samples.data(), 200);
    EXPECT_EQ(b.size(), 200);

    std::vector<float>  states(50 * kT);
    std::vector<double> targets(50);
    ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 50), 50);

    // K=1, B=50 ⇒ evict 50 per draw.
    EXPECT_EQ(b.size(), 150);

    ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 50), 50);
    EXPECT_EQ(b.size(), 100);
}

TEST(ReplayBufferTest, KTwoEvictsHalfBatchPerDraw) {
    BufferT b(/*capacity=*/200, /*min_to_start=*/100, /*K=*/2.0);
    std::vector<Sample> samples(200);
    for (int i = 0; i < 200; ++i) samples[i] = make_sample(i);
    b.add_batch(samples.data(), 200);
    EXPECT_EQ(b.size(), 200);

    std::vector<float>  states(50 * kT);
    std::vector<double> targets(50);
    ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 50), 50);

    // K=2, B=50 ⇒ evict 25 per draw.
    EXPECT_EQ(b.size(), 175);

    ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 50), 50);
    EXPECT_EQ(b.size(), 150);
}

TEST(ReplayBufferTest, FractionalKAccumulatesEvictions) {
    // K=1.5, B=10 ⇒ evict 10/1.5 = 6.667 per draw. Over 3 draws, total
    // evicted should be 20 (not 18, not 21).
    BufferT b(/*capacity=*/200, /*min_to_start=*/10, /*K=*/1.5);
    std::vector<Sample> samples(200);
    for (int i = 0; i < 200; ++i) samples[i] = make_sample(i);
    b.add_batch(samples.data(), 200);

    std::vector<float>  states(10 * kT);
    std::vector<double> targets(10);

    const int start = b.size();
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 10), 10);
    }
    EXPECT_EQ(start - b.size(), 20);
}

// ---------------------------------------------------------------------------
// Producer backpressure
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, AddBlocksWhenAtCapacity) {
    BufferT b(/*capacity=*/100, /*min_to_start=*/50, /*K=*/2.0);

    std::vector<Sample> initial(100);
    for (int i = 0; i < 100; ++i) initial[i] = make_sample(i);
    b.add_batch(initial.data(), 100);
    EXPECT_EQ(b.size(), 100);

    std::atomic<bool> second_done{false};
    std::vector<Sample> more(40);
    for (int i = 0; i < 40; ++i) more[i] = make_sample(200 + i);
    std::thread producer([&]() {
        b.add_batch(more.data(), 40);
        second_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(second_done.load()) << "add_batch should block when full";

    // A draw evicts B/K = 50/2 = 25 ⇒ frees 25 slots. Producer may still
    // need a second draw to flush all 40 samples.
    std::vector<float>  states(50 * kT);
    std::vector<double> targets(50);
    ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 50), 50);
    ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 50), 50);

    producer.join();
    EXPECT_TRUE(second_done.load());
    // 100 initial - 50 evicted + 40 added = 90.
    EXPECT_EQ(b.size(), 90);
}

TEST(ReplayBufferTest, StopUnblocksBlockedProducer) {
    BufferT b(/*capacity=*/100, /*min_to_start=*/50, /*K=*/1.0);
    std::vector<Sample> initial(100);
    for (int i = 0; i < 100; ++i) initial[i] = make_sample(i);
    b.add_batch(initial.data(), 100);

    std::atomic<bool> producer_done{false};
    std::vector<Sample> more(40);
    for (int i = 0; i < 40; ++i) more[i] = make_sample(200 + i);
    std::thread producer([&]() {
        b.add_batch(more.data(), 40);
        producer_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(producer_done.load());

    b.stop();
    producer.join();
    EXPECT_TRUE(producer_done.load());

    // Blocked batch was discarded; size is unchanged.
    EXPECT_EQ(b.size(), 100);
}

// ---------------------------------------------------------------------------
// Sampling correctness — uniform with replacement
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, SampledTargetsAreFromBuffer) {
    BufferT b(/*capacity=*/100, /*min_to_start=*/100, /*K=*/2.0);
    std::vector<Sample> samples(100);
    for (int i = 0; i < 100; ++i) samples[i] = make_sample(i);
    b.add_batch(samples.data(), 100);

    std::vector<float>  states(64 * kT);
    std::vector<double> targets(64);
    ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 64), 64);

    // Every drawn id must be in [0, 100). state[0] must match target.
    for (int i = 0; i < 64; ++i) {
        int id = static_cast<int>(targets[i]);
        EXPECT_GE(id, 0);
        EXPECT_LT(id, 100);
        EXPECT_FLOAT_EQ(states[static_cast<size_t>(i) * kT], static_cast<float>(id));
    }
}

TEST(ReplayBufferTest, SteadyStateReuseMatchesK) {
    // Steady-state check: with continuous backfill at the producer side, the
    // ratio (total_picks / total_produced) should converge to K.
    //
    // We pre-fill, then alternate add_batch(B/K) + draw_batch(B) for many
    // rounds — this keeps size_ pinned near capacity, which is the steady-
    // state regime the K guarantee applies to.
    const int N = 1000;
    const double K = 2.0;
    const int B = 100;
    const int evict_per_draw = static_cast<int>(B / K);  // 50

    BufferT b(/*capacity=*/N, /*min_to_start=*/N, /*K=*/K);

    int next_id = 0;
    std::vector<Sample> first(N);
    for (int i = 0; i < N; ++i) first[i] = make_sample(next_id++);
    b.add_batch(first.data(), N);

    std::vector<float>  states(B * kT);
    std::vector<double> targets(B);

    long total_picks    = 0;
    long total_produced = N;
    const int rounds = 100;
    for (int r = 0; r < rounds; ++r) {
        // Drawer consumes a batch (evicts B/K = 50).
        ASSERT_EQ(b.draw_batch(states.data(), targets.data(), B), B);
        total_picks += B;

        // Producer backfills exactly what was evicted, keeping size_ ≈ N.
        std::vector<Sample> next(evict_per_draw);
        for (int i = 0; i < evict_per_draw; ++i) next[i] = make_sample(next_id++);
        b.add_batch(next.data(), evict_per_draw);
        total_produced += evict_per_draw;
    }

    // Buffer should be at (or near) capacity throughout — verify size_ is
    // pinned at N and didn't run dry.
    EXPECT_EQ(b.size(), N);

    // Average uses per produced sample ≈ K. With finite rounds there's some
    // boundary effect (early-fill samples get extra draws before being
    // evicted), so allow a small slack.
    const double observed_K = static_cast<double>(total_picks) / total_produced;
    EXPECT_NEAR(observed_K, K, 0.5)
        << "observed K=" << observed_K << " (picks=" << total_picks
        << ", produced=" << total_produced << ")";
}

// ---------------------------------------------------------------------------
// Concurrent producers
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, ConcurrentProducersThreadSafe) {
    const int n_threads = 4;
    const int per_thread = 250;
    const int total = n_threads * per_thread;

    BufferT b(/*capacity=*/total, /*min_to_start=*/total, /*K=*/1.0);

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<Sample> local(per_thread);
            for (int i = 0; i < per_thread; ++i) {
                local[i] = make_sample(t * per_thread + i);
            }
            for (int i = 0; i < per_thread; i += 5) {
                b.add_batch(local.data() + i, 5);
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(b.size(), total);

    // Drain via repeated draws (with replacement); call stop() so the final
    // partial-batch draw doesn't block waiting for samples that never arrive.
    b.stop();
    std::vector<int> seen(total, 0);
    std::vector<float>  states(64 * kT);
    std::vector<double> targets(64);
    while (true) {
        int n = b.draw_batch(states.data(), targets.data(), 64);
        if (n == 0) break;
        for (int i = 0; i < n; ++i) {
            int id = static_cast<int>(targets[i]);
            ASSERT_GE(id, 0);
            ASSERT_LT(id, total);
            ++seen[id];
        }
    }
    // With uniform-with-replacement sampling over the lifetime of each sample,
    // most ids should appear at least once. Sanity bound only.
    int unique = 0;
    for (int c : seen) if (c > 0) ++unique;
    EXPECT_GT(unique, total / 4);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, TotalDrawnTracksDraws) {
    BufferT b(/*capacity=*/200, /*min_to_start=*/40, /*K=*/2.0);
    std::vector<Sample> samples(200);
    for (int i = 0; i < 200; ++i) samples[i] = make_sample(i);
    b.add_batch(samples.data(), 200);

    std::vector<float>  states(40 * kT);
    std::vector<double> targets(40);
    b.draw_batch(states.data(), targets.data(), 40);
    EXPECT_EQ(b.total_drawn(), 40);

    b.draw_batch(states.data(), targets.data(), 40);
    EXPECT_EQ(b.total_drawn(), 80);
}

// ---------------------------------------------------------------------------
// Ring wraparound: produce > capacity total across multiple draw cycles
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, RingWrapsAroundCorrectly) {
    BufferT b(/*capacity=*/100, /*min_to_start=*/100, /*K=*/1.0);

    // First fill.
    std::vector<Sample> first(100);
    for (int i = 0; i < 100; ++i) first[i] = make_sample(i);
    b.add_batch(first.data(), 100);

    // Drain all 100 (K=1, B=100 ⇒ evict everything).
    std::vector<float>  states(100 * kT);
    std::vector<double> targets(100);
    ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 100), 100);
    EXPECT_EQ(b.size(), 0);

    // Refill — internal head_/tail_ have wrapped (or are at 0); either way,
    // subsequent operations must remain correct.
    std::vector<Sample> second(100);
    for (int i = 0; i < 100; ++i) second[i] = make_sample(1000 + i);
    b.add_batch(second.data(), 100);
    EXPECT_EQ(b.size(), 100);

    ASSERT_EQ(b.draw_batch(states.data(), targets.data(), 100), 100);
    // Every drawn id must be from the second batch.
    for (int i = 0; i < 100; ++i) {
        int id = static_cast<int>(targets[i]);
        EXPECT_GE(id, 1000);
        EXPECT_LT(id, 1100);
    }
}
