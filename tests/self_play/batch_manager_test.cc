#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "self_play/batch_manager.h"
#include "self_play/game_instance.h"
#include "engine/tensor.h"

// ---------------------------------------------------------------------------
// BatchManager stress test
//
// Hammers BatchManager with many "worker" threads doing reserve+commit and
// a few "coordinator" threads doing pop+recycle, while keeping the timeout
// short so that pop_ready_batch() flushes via timeout frequently. This is
// the regime where the flush-vs-commit double-push race triggers.
//
// What the test verifies:
//   1. No tensor data corruption: each worker writes a sentinel pattern
//      derived from its commit slot; a coord later reads it back and
//      compares. A racy reset() of an in-use batch corrupts the sentinel.
//   2. No commit is lost: total tensors processed == total committed.
//   3. No commit is double-processed: each unique commit slot lands in
//      exactly one InferenceBatch::Entry seen by exactly one coordinator.
//
// The original bug: flush_active_batch_locked() and commit() could both
// push the same batch to ready_batches_ when they concurrently observed
// reserved_count == committed_count. The duplicate push then propagated
// into free_batches_, letting two paths share a batch, which corrupts
// data and double-distributes entries. Without the fix this test fails
// (corruption count > 0 or duplicate process counts > 0).
// ---------------------------------------------------------------------------

namespace {

// BatchManager treats GameInstance* as an opaque token (it stores the pointer
// in Entry and never dereferences it). We piggy-back on that to use small
// int slots instead of full ~640KB GameInstance objects, since the test
// needs tens of thousands of distinct tokens.
GameInstance* slot_to_token(int* slot) {
    return reinterpret_cast<GameInstance*>(slot);
}
int token_to_slot(GameInstance* g) {
    return *reinterpret_cast<int*>(g);
}

}  // namespace

TEST(BatchManagerStressTest, NoDoubleProcessOrLossUnderContention) {
    constexpr int kNumWorkers       = 16;
    constexpr int kNumCoords        = 3;
    constexpr int kMaxTensors       = 256;
    constexpr int kNumBatches       = kNumCoords * 2;
    constexpr int kCommitsPerWorker = 2000;
    constexpr int kTimeoutMs        = 1;     // tight: forces frequent timeout-flushes
    constexpr int kTotalCommits     = kNumWorkers * kCommitsPerWorker;

    BatchManager bm(kNumBatches, kMaxTensors, /*use_pinned=*/false, kTimeoutMs);

    // Per-commit token storage: tokens[slot] holds the slot index, and
    // &tokens[slot] is passed as the GameInstance* to commit().
    std::vector<int> tokens(kTotalCommits);
    for (int i = 0; i < kTotalCommits; ++i) tokens[i] = i;

    auto process_counts = std::make_unique<std::atomic<int>[]>(kTotalCommits);
    for (int i = 0; i < kTotalCommits; ++i) process_counts[i].store(0);

    std::atomic<int64_t> total_reserved{0};
    std::atomic<int64_t> total_committed{0};
    std::atomic<int64_t> total_processed{0};
    std::atomic<int>     corruption_count{0};

    // -- Workers: reserve, write sentinel, commit ----------------------------
    std::vector<std::thread> workers;
    workers.reserve(kNumWorkers);
    for (int w = 0; w < kNumWorkers; ++w) {
        workers.emplace_back([&, w]() {
            std::mt19937 rng(static_cast<uint32_t>(w * 31 + 7));
            std::uniform_int_distribution<int> req_dist(1, 32);
            for (int i = 0; i < kCommitsPerWorker; ++i) {
                const int req_count = req_dist(rng);
                const int slot = w * kCommitsPerWorker + i;

                InferenceBatch* batch = nullptr;
                int offset = 0;
                float* dst = bm.reserve(req_count, batch, offset);
                if (!dst) return;  // shutdown signaled

                // Sentinel: tensor j's first float = slot*1000 + j. The full
                // tensor doesn't need writing — we only read the first float
                // back in the coord.
                for (int j = 0; j < req_count; ++j) {
                    dst[j * kTensorSize + 0] = static_cast<float>(slot * 1000 + j);
                }

                total_reserved.fetch_add(req_count, std::memory_order_relaxed);
                bm.commit(batch, slot_to_token(&tokens[slot]), offset, req_count);
                total_committed.fetch_add(req_count, std::memory_order_relaxed);
            }
        });
    }

    // -- Coordinators: pop, verify sentinel, count, recycle ------------------
    std::vector<std::thread> coords;
    coords.reserve(kNumCoords);
    for (int c = 0; c < kNumCoords; ++c) {
        coords.emplace_back([&]() {
            while (true) {
                InferenceBatch* batch = bm.pop_ready_batch();
                if (!batch) break;

                for (const auto& e : batch->entries) {
                    const int slot = token_to_slot(e.game);
                    process_counts[slot].fetch_add(1, std::memory_order_relaxed);

                    for (int j = 0; j < e.count; ++j) {
                        const float expected = static_cast<float>(slot * 1000 + j);
                        const float actual =
                            batch->data_ptr[(e.start_idx + j) * kTensorSize + 0];
                        if (expected != actual) {
                            corruption_count.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    total_processed.fetch_add(e.count, std::memory_order_relaxed);
                }

                bm.recycle_batch(batch);
            }
        });
    }

    for (auto& w : workers) w.join();

    // After workers join, the last partial batch is still active and waiting
    // on the timeout flush. Give pop_ready_batch() time to drain it.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (total_processed.load(std::memory_order_relaxed) <
               total_committed.load(std::memory_order_relaxed) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    bm.shutdown();
    for (auto& c : coords) c.join();

    EXPECT_EQ(total_reserved.load(), total_committed.load());
    EXPECT_EQ(total_committed.load(), total_processed.load())
        << "Tensor count mismatch — entries were lost or duplicated.";
    EXPECT_EQ(corruption_count.load(), 0)
        << "Sentinel mismatch — a batch was reset or written concurrently with "
           "coordinator processing.";

    int doubled = 0;
    int missed  = 0;
    for (int i = 0; i < kTotalCommits; ++i) {
        const int c = process_counts[i].load();
        if (c == 0)      ++missed;
        else if (c > 1)  ++doubled;
    }
    EXPECT_EQ(doubled, 0) << doubled << " commits were processed more than once";
    EXPECT_EQ(missed,  0) << missed  << " commits were never processed";
}

// ---------------------------------------------------------------------------
// Regression for the specific double-push race.
//
// This is a stricter version of the stress test using small batches and
// many workers per batch slot, which historically reproduced the race in
// a few hundred ms. Kept separate so a CI failure points at the right
// invariant.
// ---------------------------------------------------------------------------
TEST(BatchManagerStressTest, FlushVsCommitRace_NoDoublePush) {
    constexpr int kNumWorkers       = 32;
    constexpr int kNumCoords        = 2;
    constexpr int kMaxTensors       = 64;    // small → frequent fill-flushes
    constexpr int kNumBatches       = 4;
    constexpr int kCommitsPerWorker = 1500;
    constexpr int kTimeoutMs        = 1;
    constexpr int kTotalCommits     = kNumWorkers * kCommitsPerWorker;

    BatchManager bm(kNumBatches, kMaxTensors, /*use_pinned=*/false, kTimeoutMs);

    std::vector<int> tokens(kTotalCommits);
    for (int i = 0; i < kTotalCommits; ++i) tokens[i] = i;

    auto process_counts = std::make_unique<std::atomic<int>[]>(kTotalCommits);
    for (int i = 0; i < kTotalCommits; ++i) process_counts[i].store(0);

    std::atomic<int64_t> total_committed{0};
    std::atomic<int64_t> total_processed{0};

    std::vector<std::thread> workers;
    workers.reserve(kNumWorkers);
    for (int w = 0; w < kNumWorkers; ++w) {
        workers.emplace_back([&, w]() {
            std::mt19937 rng(static_cast<uint32_t>(w * 1009 + 1));
            // Bias toward small reservations so committed often hits reserved
            // exactly while the active batch is still partially full — the
            // exact regime that triggered the flush-vs-commit race.
            std::uniform_int_distribution<int> req_dist(1, 8);
            for (int i = 0; i < kCommitsPerWorker; ++i) {
                const int req_count = req_dist(rng);
                const int slot = w * kCommitsPerWorker + i;

                InferenceBatch* batch = nullptr;
                int offset = 0;
                float* dst = bm.reserve(req_count, batch, offset);
                if (!dst) return;

                // Touch the first float so a stale processing path reading
                // post-reset memory would observe a different value.
                for (int j = 0; j < req_count; ++j) {
                    dst[j * kTensorSize + 0] = static_cast<float>(slot * 1000 + j);
                }
                bm.commit(batch, slot_to_token(&tokens[slot]), offset, req_count);
                total_committed.fetch_add(req_count, std::memory_order_relaxed);
            }
        });
    }

    std::vector<std::thread> coords;
    coords.reserve(kNumCoords);
    for (int c = 0; c < kNumCoords; ++c) {
        coords.emplace_back([&]() {
            while (true) {
                InferenceBatch* batch = bm.pop_ready_batch();
                if (!batch) break;
                for (const auto& e : batch->entries) {
                    const int slot = token_to_slot(e.game);
                    process_counts[slot].fetch_add(1, std::memory_order_relaxed);
                    total_processed.fetch_add(e.count, std::memory_order_relaxed);
                }
                bm.recycle_batch(batch);
            }
        });
    }

    for (auto& w : workers) w.join();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (total_processed.load(std::memory_order_relaxed) <
               total_committed.load(std::memory_order_relaxed) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    bm.shutdown();
    for (auto& c : coords) c.join();

    EXPECT_EQ(total_committed.load(), total_processed.load());
    int doubled = 0, missed = 0;
    for (int i = 0; i < kTotalCommits; ++i) {
        const int c = process_counts[i].load();
        if (c == 0) ++missed;
        else if (c > 1) ++doubled;
    }
    EXPECT_EQ(doubled, 0) << "double-push race fired " << doubled << " times";
    EXPECT_EQ(missed,  0) << "missed " << missed << " commits";
}
