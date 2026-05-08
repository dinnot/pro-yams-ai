#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

#include <torch/torch.h>

#include "self_play/game_instance.h"
#include "engine/tensor.h"

// ---------------------------------------------------------------------------
// InferenceBatch
//
// Represents a single contiguous PyTorch batch tensor allocated in pinned
// memory. Workers concurrently reserve and write their tensors directly into
// this memory.
// ---------------------------------------------------------------------------
class InferenceBatch {
public:
    InferenceBatch(int max_tensors, bool use_pinned);

    struct Entry {
        GameInstance* game;
        int start_idx;
        int count;
    };

    torch::Tensor storage;
    float* data_ptr;
    int max_tensors;

    std::atomic<int> reserved_count{0};
    std::atomic<int> committed_count{0};

    std::mutex entries_mtx;
    std::vector<Entry> entries;

    // Single-pusher flag: protected by BatchManager::active_mtx_. The first
    // thread (committer or flusher) to observe reserved==committed flips it
    // and is the sole pusher to ready_batches_. Reset when the batch is
    // pulled back out of the free pool.
    bool push_claimed = false;

    void reset();
};

// ---------------------------------------------------------------------------
// BatchManager
//
// Manages the active InferenceBatch that workers write to, and a queue of
// ready batches for Coordinators to process.
// ---------------------------------------------------------------------------
class BatchManager {
public:
    BatchManager(int num_batches, int max_tensors, bool use_pinned, int timeout_ms);
    ~BatchManager();

    // -----------------------------------------------------------------------
    // Worker API
    // -----------------------------------------------------------------------

    // Reserve slots for tensor generation. Returns the pointer to the exactly
    // allocated memory block. If the active batch is full, it blocks until a
    // new batch is available.
    // Also returns the batch and offset needed for the commit call.
    float* reserve(int req_count, InferenceBatch*& out_batch, int& out_offset);

    // Commit the generated tensors to the batch. When all reserved slots in
    // the batch are committed, the batch is pushed to the ready queue.
    void commit(InferenceBatch* batch, GameInstance* game, int offset, int req_count);

    // -----------------------------------------------------------------------
    // Coordinator API
    // -----------------------------------------------------------------------

    // Blocks until a full or timed-out batch is ready for inference.
    InferenceBatch* pop_ready_batch();

    // Recycles an inferred batch back into the free pool.
    void recycle_batch(InferenceBatch* batch);

    // Shutdown signal
    void shutdown();

private:
    InferenceBatch* flush_active_batch_locked();

    int max_tensors_;
    int timeout_ms_;
    std::atomic<bool> shutdown_{false};

    std::vector<std::unique_ptr<InferenceBatch>> all_batches_;

    std::mutex pool_mtx_;
    std::condition_variable pool_cv_;
    std::vector<InferenceBatch*> free_batches_;

    std::mutex active_mtx_;
    InferenceBatch* active_batch_{nullptr};
    std::chrono::steady_clock::time_point active_batch_start_;

    std::mutex ready_mtx_;
    std::condition_variable ready_cv_;
    std::deque<InferenceBatch*> ready_batches_;
};
