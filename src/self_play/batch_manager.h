#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include <torch/torch.h>

#include "self_play/game_instance.h"
#include "engine/tensor.h"

// ---------------------------------------------------------------------------
// InferenceBatchT<Traits>
//
// Represents a single contiguous PyTorch batch tensor allocated in pinned
// memory. Workers concurrently reserve and write their tensors directly into
// this memory.
// ---------------------------------------------------------------------------
template <typename Traits>
class InferenceBatchT {
public:
    using Instance = GameInstanceT<Traits>;

    InferenceBatchT(int max_t, bool use_pinned) : max_tensors(max_t) {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        if (use_pinned) {
            opts = opts.pinned_memory(true);
        }
        storage = torch::empty({max_tensors, Traits::kTensorSize}, opts);
        data_ptr = storage.data_ptr<float>();
        entries.reserve(static_cast<size_t>(max_tensors));
    }

    struct Entry {
        Instance* game;
        int       start_idx;
        int       count;
    };

    torch::Tensor storage;
    float*        data_ptr;
    int           max_tensors;

    std::atomic<int> reserved_count{0};
    std::atomic<int> committed_count{0};

    std::mutex            entries_mtx;
    std::vector<Entry>    entries;

    // Single-pusher flag: protected by BatchManagerT::active_mtx_. The first
    // thread (committer or flusher) to observe reserved==committed flips it
    // and is the sole pusher to ready_batches_. Reset when the batch is
    // pulled back out of the free pool.
    bool push_claimed = false;

    void reset() {
        reserved_count.store(0, std::memory_order_relaxed);
        committed_count.store(0, std::memory_order_relaxed);
        entries.clear();
        push_claimed = false;
    }
};

// ---------------------------------------------------------------------------
// BatchManagerT<Traits>
//
// Manages the active InferenceBatchT that workers write to, and a queue of
// ready batches for Coordinators to process.
// ---------------------------------------------------------------------------
template <typename Traits>
class BatchManagerT {
public:
    using Batch    = InferenceBatchT<Traits>;
    using Instance = GameInstanceT<Traits>;

    BatchManagerT(int num_batches, int max_tensors, bool use_pinned, int timeout_ms)
        : max_tensors_(max_tensors), timeout_ms_(timeout_ms) {
        all_batches_.reserve(static_cast<size_t>(num_batches));
        for (int i = 0; i < num_batches; ++i) {
            all_batches_.push_back(std::make_unique<Batch>(max_tensors, use_pinned));
            free_batches_.push_back(all_batches_.back().get());
        }
    }

    ~BatchManagerT() {
        shutdown();
    }

    // -----------------------------------------------------------------------
    // Worker API
    // -----------------------------------------------------------------------

    /// Reserve slots for tensor generation. Returns the pointer to the exactly
    /// allocated memory block. If the active batch is full, it blocks until a
    /// new batch is available. Also returns the batch and offset needed for
    /// the commit call.
    float* reserve(int req_count, Batch*& out_batch, int& out_offset) {
        assert(req_count > 0 && req_count <= max_tensors_);

        while (!shutdown_.load(std::memory_order_relaxed)) {
            Batch* batch_to_push = nullptr;
            {
                std::lock_guard<std::mutex> active_lock(active_mtx_);
                if (active_batch_) {
                    int current = active_batch_->reserved_count.load(std::memory_order_relaxed);
                    if (current + req_count <= max_tensors_) {
                        out_offset = current;
                        active_batch_->reserved_count.fetch_add(req_count, std::memory_order_relaxed);
                        out_batch = active_batch_;
                        return active_batch_->data_ptr + (out_offset * Traits::kTensorSize);
                    }
                    batch_to_push = flush_active_batch_locked();
                }
            }

            if (batch_to_push) {
                std::lock_guard<std::mutex> ready_lock(ready_mtx_);
                ready_batches_.push_back(batch_to_push);
                ready_cv_.notify_one();
            }

            // Phase 2: get a free batch from the pool. Must NOT hold active_mtx_ here.
            Batch* candidate = nullptr;
            {
                std::unique_lock<std::mutex> pool_lock(pool_mtx_);
                pool_cv_.wait(pool_lock, [this]() {
                    return !free_batches_.empty() || shutdown_.load(std::memory_order_relaxed);
                });
                if (shutdown_.load(std::memory_order_relaxed)) return nullptr;
                candidate = free_batches_.back();
                free_batches_.pop_back();
            }

            // Phase 3: install the candidate as the active batch.
            bool installed = false;
            {
                std::lock_guard<std::mutex> active_lock(active_mtx_);
                if (!active_batch_) {
                    candidate->reset();
                    active_batch_ = candidate;
                    active_batch_start_ = std::chrono::steady_clock::now();
                    installed = true;
                }
            }

            if (installed) {
                // Wake any coordinator parked on ready_cv_ so it can compute
                // the new timeout deadline against active_batch_start_.
                ready_cv_.notify_one();
            } else {
                std::lock_guard<std::mutex> pool_lock(pool_mtx_);
                free_batches_.push_back(candidate);
                pool_cv_.notify_one();
            }
        }

        return nullptr;
    }

    /// Commit the generated tensors to the batch. When all reserved slots in
    /// the batch are committed, the batch is pushed to the ready queue.
    void commit(Batch* batch, Instance* game, int offset, int req_count) {
        if (!batch) return;

        {
            std::lock_guard<std::mutex> lock(batch->entries_mtx);
            batch->entries.push_back({game, offset, req_count});
        }

        int prev_committed = batch->committed_count.fetch_add(req_count, std::memory_order_relaxed);
        int new_committed = prev_committed + req_count;

        bool should_push = false;

        if (new_committed == max_tensors_) {
            std::lock_guard<std::mutex> lock(active_mtx_);
            if (active_batch_ == batch) {
                active_batch_ = nullptr;
            }
            if (!batch->push_claimed) {
                batch->push_claimed = true;
                should_push = true;
            }
        } else {
            std::lock_guard<std::mutex> lock(active_mtx_);
            if (active_batch_ != batch &&
                new_committed == batch->reserved_count.load(std::memory_order_relaxed) &&
                !batch->push_claimed) {
                batch->push_claimed = true;
                should_push = true;
            }
        }

        if (should_push) {
            std::lock_guard<std::mutex> ready_lock(ready_mtx_);
            ready_batches_.push_back(batch);
            ready_cv_.notify_one();
        }
    }

    // -----------------------------------------------------------------------
    // Coordinator API
    // -----------------------------------------------------------------------

    /// Blocks until a full or timed-out batch is ready for inference.
    Batch* pop_ready_batch() {
        std::unique_lock<std::mutex> lock(ready_mtx_);
        while (ready_batches_.empty() && !shutdown_.load(std::memory_order_relaxed)) {
            lock.unlock();

            bool has_deadline = false;
            std::chrono::steady_clock::time_point deadline;
            Batch* flushed_batch = nullptr;

            {
                std::unique_lock<std::mutex> act_lock(active_mtx_, std::try_to_lock);
                if (act_lock.owns_lock() &&
                    active_batch_ &&
                    active_batch_->reserved_count.load(std::memory_order_relaxed) > 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - active_batch_start_).count();
                    if (elapsed >= timeout_ms_) {
                        flushed_batch = flush_active_batch_locked();
                    } else {
                        deadline = active_batch_start_ +
                                   std::chrono::milliseconds(timeout_ms_);
                        has_deadline = true;
                    }
                }
            }

            lock.lock();

            if (flushed_batch) {
                ready_batches_.push_back(flushed_batch);
                continue;
            }

            if (has_deadline) {
                // Sleep until either a ready batch arrives or the active
                // batch's timeout is reached.
                ready_cv_.wait_until(lock, deadline);
            } else {
                // No active batch (or active_mtx_ was contended). Park until
                // a batch is pushed or installed. The timeout_ms_ fallback
                // closes the small window where reserve()'s install-notify
                // races with this thread's wait (and as a safety net if the
                // try_lock above missed an in-flight install).
                ready_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms_));
            }
        }

        if (shutdown_.load(std::memory_order_relaxed) && ready_batches_.empty()) {
            return nullptr;
        }

        Batch* b = ready_batches_.front();
        ready_batches_.pop_front();
        return b;
    }

    /// Recycles an inferred batch back into the free pool.
    void recycle_batch(Batch* batch) {
        if (!batch) return;
        std::lock_guard<std::mutex> lock(pool_mtx_);
        free_batches_.push_back(batch);
        pool_cv_.notify_one();
    }

    /// Shutdown signal.
    void shutdown() {
        shutdown_.store(true, std::memory_order_relaxed);
        pool_cv_.notify_all();
        ready_cv_.notify_all();
    }

private:
    Batch* flush_active_batch_locked() {
        if (!active_batch_) return nullptr;

        Batch* b = active_batch_;
        active_batch_ = nullptr;

        int reserved = b->reserved_count.load(std::memory_order_relaxed);
        int committed = b->committed_count.load(std::memory_order_relaxed);

        if (reserved == 0) {
            std::lock_guard<std::mutex> pool_lock(pool_mtx_);
            free_batches_.push_back(b);
            pool_cv_.notify_one();
            return nullptr;
        } else if (reserved == committed) {
            if (b->push_claimed) return nullptr;
            b->push_claimed = true;
            return b;
        }
        return nullptr;
    }

    int max_tensors_;
    int timeout_ms_;
    std::atomic<bool> shutdown_{false};

    std::vector<std::unique_ptr<Batch>> all_batches_;

    std::mutex pool_mtx_;
    std::condition_variable pool_cv_;
    std::vector<Batch*> free_batches_;

    std::mutex active_mtx_;
    Batch* active_batch_{nullptr};
    std::chrono::steady_clock::time_point active_batch_start_;

    std::mutex ready_mtx_;
    std::condition_variable ready_cv_;
    std::deque<Batch*> ready_batches_;
};

// Backward-compat aliases.
using InferenceBatch    = InferenceBatchT<Yams1v1>;
using InferenceBatch2v2 = InferenceBatchT<Yams2v2>;
using BatchManager      = BatchManagerT<Yams1v1>;
using BatchManager2v2   = BatchManagerT<Yams2v2>;
