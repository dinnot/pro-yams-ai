#include <gtest/gtest.h>

#include "distil/shuffle_queue.h"
#include "engine/game_traits.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using QueueT = ShuffleQueue;            // Yams1v1 specialisation
using Sample = QueueT::Sample;
static constexpr int kT = Yams1v1::kTensorSize;

// Tag the state with the target's integer value so the test can recover
// each sample's identity after permutation.
static Sample make_sample(int id) {
    Sample s{};
    s.state[0] = static_cast<float>(id);
    s.target   = static_cast<double>(id);
    return s;
}

// ---------------------------------------------------------------------------
// Construction / initial state
// ---------------------------------------------------------------------------

TEST(ShuffleQueueTest, InitiallyEmpty) {
    QueueT q(1024, 256);
    EXPECT_EQ(q.accumulating_size(), 0);
    EXPECT_EQ(q.serving_remaining(), 0);
    EXPECT_EQ(q.total_drawn(), 0);
}

TEST(ShuffleQueueTest, AddBatchIncrementsAccumulating) {
    QueueT q(1024, 256);
    std::vector<Sample> samples(50);
    for (int i = 0; i < 50; ++i) samples[i] = make_sample(i);
    q.add_batch(samples.data(), 50);
    EXPECT_EQ(q.accumulating_size(), 50);
    EXPECT_EQ(q.serving_remaining(), 0);
}

// ---------------------------------------------------------------------------
// Blocking / unblock semantics
// ---------------------------------------------------------------------------

TEST(ShuffleQueueTest, DrawBlocksUntilFirstChunkReady) {
    QueueT q(1024, 100);

    // Push fewer than min_chunk_size_to_start — draw must block.
    std::vector<Sample> samples(50);
    for (int i = 0; i < 50; ++i) samples[i] = make_sample(i);
    q.add_batch(samples.data(), 50);

    std::atomic<bool> draw_done{false};
    std::vector<float>  states(32 * kT);
    std::vector<double> targets(32);
    std::thread t([&]() {
        q.draw_batch(states.data(), targets.data(), 32);
        draw_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(draw_done.load());

    // Push enough to cross the threshold — drawer must unblock.
    std::vector<Sample> more(60);
    for (int i = 0; i < 60; ++i) more[i] = make_sample(100 + i);
    q.add_batch(more.data(), 60);

    t.join();
    EXPECT_TRUE(draw_done.load());
}

TEST(ShuffleQueueTest, StopUnblocksEmptyDraw) {
    QueueT q(1024, 100);
    std::atomic<int> result{-1};
    std::vector<float>  states(32 * kT);
    std::vector<double> targets(32);
    std::thread t([&]() {
        result.store(q.draw_batch(states.data(), targets.data(), 32));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(result.load(), -1);

    q.stop();
    t.join();
    EXPECT_EQ(result.load(), 0);
}

TEST(ShuffleQueueTest, StopDrainsRemainingBelowThreshold) {
    QueueT q(1000, 1000);
    std::vector<Sample> samples(50);
    for (int i = 0; i < 50; ++i) samples[i] = make_sample(i);
    q.add_batch(samples.data(), 50);

    // Below threshold — without stop, draw would block. With stop, drain.
    q.stop();

    std::vector<int> count(50, 0);
    std::vector<float>  states(64 * kT);
    std::vector<double> targets(64);
    int drawn = 0;
    while (true) {
        int n = q.draw_batch(states.data(), targets.data(), 64);
        if (n == 0) break;
        for (int i = 0; i < n; ++i) {
            int id = static_cast<int>(targets[i]);
            ASSERT_GE(id, 0);
            ASSERT_LT(id, 50);
            count[id]++;
        }
        drawn += n;
    }
    EXPECT_EQ(drawn, 50);
    for (int i = 0; i < 50; ++i) EXPECT_EQ(count[i], 1) << "id=" << i;
}

// ---------------------------------------------------------------------------
// Once-and-only-once semantics
// ---------------------------------------------------------------------------

TEST(ShuffleQueueTest, EachSampleDrawnExactlyOnce_MultipleChunks) {
    const int chunk = 100;
    const int total = 500;
    QueueT q(chunk, chunk);

    std::vector<Sample> samples(total);
    for (int i = 0; i < total; ++i) samples[i] = make_sample(i);

    std::vector<int>    count(total, 0);
    std::vector<float>  states(50 * kT);
    std::vector<double> targets(50);

    // Interleave pushes and draws: push 100, draw 50 — forces multiple
    // rotations as chunks fill and partially empty.
    int pushed = 0, drawn = 0;
    while (pushed < total) {
        q.add_batch(samples.data() + pushed, 100);
        pushed += 100;
        int n = q.draw_batch(states.data(), targets.data(), 50);
        for (int i = 0; i < n; ++i) {
            int id = static_cast<int>(targets[i]);
            ASSERT_GE(id, 0);
            ASSERT_LT(id, total);
            count[id]++;
        }
        drawn += n;
    }

    // Drain everything left.
    q.stop();
    while (true) {
        int n = q.draw_batch(states.data(), targets.data(), 50);
        if (n == 0) break;
        for (int i = 0; i < n; ++i) {
            int id = static_cast<int>(targets[i]);
            count[id]++;
        }
        drawn += n;
    }

    EXPECT_EQ(drawn, total);
    for (int i = 0; i < total; ++i) {
        EXPECT_EQ(count[i], 1) << "id=" << i;
    }
}

TEST(ShuffleQueueTest, ConcurrentProducersThreadSafe) {
    const int n_threads = 4;
    const int per_thread = 250;
    const int total = n_threads * per_thread;

    QueueT q(total, total);

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<Sample> local(per_thread);
            for (int i = 0; i < per_thread; ++i) {
                local[i] = make_sample(t * per_thread + i);
            }
            // Push in small batches to maximise contention.
            for (int i = 0; i < per_thread; i += 5) {
                q.add_batch(local.data() + i, 5);
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(q.accumulating_size(), total);

    q.stop();
    std::vector<int>    count(total, 0);
    std::vector<float>  states(64 * kT);
    std::vector<double> targets(64);
    int drawn = 0;
    while (true) {
        int n = q.draw_batch(states.data(), targets.data(), 64);
        if (n == 0) break;
        for (int i = 0; i < n; ++i) {
            int id = static_cast<int>(targets[i]);
            ASSERT_GE(id, 0);
            ASSERT_LT(id, total);
            count[id]++;
        }
        drawn += n;
    }
    EXPECT_EQ(drawn, total);
    for (int i = 0; i < total; ++i) EXPECT_EQ(count[i], 1) << "id=" << i;
}

// ---------------------------------------------------------------------------
// Cross-chunk batch filling
// ---------------------------------------------------------------------------

TEST(ShuffleQueueTest, DrawSplitsAcrossChunks) {
    const int chunk = 30;
    QueueT q(chunk, chunk);

    std::vector<Sample> first(chunk);
    for (int i = 0; i < chunk; ++i) first[i] = make_sample(i);
    q.add_batch(first.data(), chunk);

    std::vector<float>  states(50 * kT);
    std::vector<double> targets(50);

    // Draw 20 from chunk 1 (10 remaining in serving_).
    int n = q.draw_batch(states.data(), targets.data(), 20);
    EXPECT_EQ(n, 20);

    // Push chunk 2.
    std::vector<Sample> second(chunk);
    for (int i = 0; i < chunk; ++i) second[i] = make_sample(100 + i);
    q.add_batch(second.data(), chunk);

    // Draw 25: consumes remaining 10 from chunk 1, rotates, takes 15 from chunk 2.
    n = q.draw_batch(states.data(), targets.data(), 25);
    EXPECT_EQ(n, 25);

    int from_first = 0, from_second = 0;
    for (int i = 0; i < 25; ++i) {
        int id = static_cast<int>(targets[i]);
        if (id < 100) ++from_first;
        else          ++from_second;
    }
    EXPECT_EQ(from_first, 10);
    EXPECT_EQ(from_second, 15);
}

// ---------------------------------------------------------------------------
// Order is actually permuted (not FIFO)
// ---------------------------------------------------------------------------

TEST(ShuffleQueueTest, OrderIsShuffled) {
    const int n = 1000;
    QueueT q(n, n);

    std::vector<Sample> samples(n);
    for (int i = 0; i < n; ++i) samples[i] = make_sample(i);
    q.add_batch(samples.data(), n);

    std::vector<float>  states(n * kT);
    std::vector<double> targets(n);
    int got = q.draw_batch(states.data(), targets.data(), n);
    ASSERT_EQ(got, n);

    // For a uniform random permutation of {0..n-1}, the expected number of
    // fixed points (targets[i] == i) is 1. FIFO would give n. A threshold
    // of n/10 is enormously conservative.
    int fixed = 0;
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(targets[i]) == i) ++fixed;
    }
    EXPECT_LT(fixed, n / 10) << "order looks FIFO: " << fixed << " fixed points";

    // Sanity: every id appears exactly once.
    std::vector<int> count(n, 0);
    for (int i = 0; i < n; ++i) count[static_cast<int>(targets[i])]++;
    for (int i = 0; i < n; ++i) ASSERT_EQ(count[i], 1) << "id=" << i;
}

// ---------------------------------------------------------------------------
// Diagnostics counters
// ---------------------------------------------------------------------------

TEST(ShuffleQueueTest, TotalDrawnTracksDraws) {
    QueueT q(100, 100);
    std::vector<Sample> samples(100);
    for (int i = 0; i < 100; ++i) samples[i] = make_sample(i);
    q.add_batch(samples.data(), 100);

    std::vector<float>  states(40 * kT);
    std::vector<double> targets(40);
    q.draw_batch(states.data(), targets.data(), 40);
    EXPECT_EQ(q.total_drawn(), 40);

    q.draw_batch(states.data(), targets.data(), 40);
    EXPECT_EQ(q.total_drawn(), 80);
}

// ---------------------------------------------------------------------------
// Back-pressure: add_batch blocks when buffered_size >= cap, unblocks when
// the drawer consumes samples.
// ---------------------------------------------------------------------------

TEST(ShuffleQueueTest, AddBlocksWhenCapReached) {
    // chunk=50, min_to_start=50, cap=100.
    QueueT q(50, 50, /*max_buffered_samples=*/100);

    std::vector<Sample> initial(100);
    for (int i = 0; i < 100; ++i) initial[i] = make_sample(i);
    q.add_batch(initial.data(), 100);
    EXPECT_EQ(q.buffered_size(), 100);

    // A second producer trying to add more must block.
    std::atomic<bool> second_done{false};
    std::vector<Sample> more(40);
    for (int i = 0; i < 40; ++i) more[i] = make_sample(200 + i);
    std::thread producer([&]() {
        q.add_batch(more.data(), 40);
        second_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(second_done.load()) << "add_batch should have blocked at cap";

    // Drawer consumes some samples — should free space and unblock producer.
    std::vector<float>  states(50 * kT);
    std::vector<double> targets(50);
    int n = q.draw_batch(states.data(), targets.data(), 50);
    EXPECT_EQ(n, 50);

    producer.join();
    EXPECT_TRUE(second_done.load());
}

TEST(ShuffleQueueTest, StopUnblocksBlockedProducer) {
    QueueT q(50, 50, /*max_buffered_samples=*/100);

    std::vector<Sample> initial(100);
    for (int i = 0; i < 100; ++i) initial[i] = make_sample(i);
    q.add_batch(initial.data(), 100);

    std::atomic<bool> producer_done{false};
    std::vector<Sample> more(40);
    for (int i = 0; i < 40; ++i) more[i] = make_sample(200 + i);
    std::thread producer([&]() {
        q.add_batch(more.data(), 40);
        producer_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(producer_done.load());

    q.stop();
    producer.join();
    EXPECT_TRUE(producer_done.load());

    // The blocked batch should have been discarded (no overshoot).
    EXPECT_EQ(q.buffered_size(), 100);
}

TEST(ShuffleQueueTest, UnboundedDefaultDoesNotBlock) {
    // Default constructor (no cap arg) means kUnbounded — no blocking.
    QueueT q(100, 100);
    std::vector<Sample> samples(10'000);
    for (int i = 0; i < 10'000; ++i) samples[i] = make_sample(i);
    q.add_batch(samples.data(), 10'000);
    EXPECT_EQ(q.buffered_size(), 10'000);
    EXPECT_EQ(q.max_buffered_samples(), QueueT::kUnbounded);
}
