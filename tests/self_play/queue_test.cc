#include <gtest/gtest.h>
#include "self_play/game_queues.h"
#include "self_play/game_instance.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// Use pointer values as stand-ins for real GameInstances (no game logic here).
// We cast small integers to GameInstance* for identity testing.
static GameInstance* fake(int id) {
    return reinterpret_cast<GameInstance*>(static_cast<uintptr_t>(id + 1));
}

// ---------------------------------------------------------------------------
// Basic push / pop
// ---------------------------------------------------------------------------
TEST(QueueTest, PushPop_Single) {
    GameQueue q;
    q.push(fake(42));
    GameInstance* g = q.try_pop();
    ASSERT_EQ(g, fake(42));
    EXPECT_EQ(q.try_pop(), nullptr);
}

TEST(QueueTest, PushPop_FIFO_Order) {
    GameQueue q;
    for (int i = 0; i < 10; ++i) q.push(fake(i));
    for (int i = 0; i < 10; ++i) {
        GameInstance* g = q.try_pop();
        EXPECT_EQ(g, fake(i)) << "FIFO order violated at index " << i;
    }
    EXPECT_EQ(q.try_pop(), nullptr);
}

TEST(QueueTest, Size) {
    GameQueue q;
    EXPECT_EQ(q.size(), 0);
    q.push(fake(0));
    EXPECT_EQ(q.size(), 1);
    q.push(fake(1));
    EXPECT_EQ(q.size(), 2);
    q.try_pop();
    EXPECT_EQ(q.size(), 1);
}

// ---------------------------------------------------------------------------
// push_front
// ---------------------------------------------------------------------------
TEST(QueueTest, PushFront_MaintainsPriority) {
    GameQueue q;
    q.push(fake(1));
    q.push_front(fake(0));
    EXPECT_EQ(q.try_pop(), fake(0));
    EXPECT_EQ(q.try_pop(), fake(1));
}

// ---------------------------------------------------------------------------
// push_batch
// ---------------------------------------------------------------------------
TEST(QueueTest, PushBatch) {
    GameQueue q;
    GameInstance* games[5] = {fake(0), fake(1), fake(2), fake(3), fake(4)};
    q.push_batch(games, 5);
    EXPECT_EQ(q.size(), 5);
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(q.try_pop(), fake(i));
}

// ---------------------------------------------------------------------------
// Thread safety — many producers, one consumer
// ---------------------------------------------------------------------------
TEST(QueueTest, ThreadSafety_ManyProducers) {
    GameQueue q;
    constexpr int kThreads = 4;
    constexpr int kPerThread = 100;

    std::vector<std::thread> producers;
    producers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        producers.emplace_back([&q, t] {
            for (int i = 0; i < kPerThread; ++i)
                q.push(fake(t * kPerThread + i));
        });
    }
    for (auto& th : producers) th.join();

    EXPECT_EQ(q.size(), kThreads * kPerThread);

    // Drain — just verify count, not order (multi-producer has no defined order)
    int count = 0;
    while (q.try_pop() != nullptr) ++count;
    EXPECT_EQ(count, kThreads * kPerThread);
}

// ---------------------------------------------------------------------------
// collect — returns available items up to max within timeout
// ---------------------------------------------------------------------------
TEST(QueueTest, Collect_ReturnsAvailableItems) {
    GameQueue q;
    q.push(fake(0));
    q.push(fake(1));
    q.push(fake(2));

    GameInstance* out[5] = {};
    int n = q.collect(out, 5, 100);  // ask for 5, only 3 available
    EXPECT_EQ(n, 3);
    EXPECT_EQ(out[0], fake(0));
    EXPECT_EQ(out[1], fake(1));
    EXPECT_EQ(out[2], fake(2));
}

TEST(QueueTest, Collect_Timeout_OnEmpty) {
    GameQueue q;
    GameInstance* out[4] = {};
    auto t0 = std::chrono::steady_clock::now();
    int n = q.collect(out, 4, 50);  // 50 ms timeout
    auto elapsed = std::chrono::steady_clock::now() - t0;

    EXPECT_EQ(n, 0);
    // Should have waited roughly 50ms (allow generous slack).
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 40);
}

// ---------------------------------------------------------------------------
// Blocking pop — woken by push from another thread
// ---------------------------------------------------------------------------
TEST(QueueTest, BlockingPop_WokenByPush) {
    GameQueue q;
    std::atomic<bool> received{false};

    std::thread consumer([&] {
        GameInstance* g = q.pop();
        EXPECT_EQ(g, fake(99));
        received = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(received.load());
    q.push(fake(99));

    consumer.join();
    EXPECT_TRUE(received.load());
}

// ---------------------------------------------------------------------------
// Nullptr sentinel passes through normally
// ---------------------------------------------------------------------------
TEST(QueueTest, NullptrSentinel) {
    GameQueue q;
    q.push(nullptr);
    EXPECT_EQ(q.pop(), nullptr);
}
