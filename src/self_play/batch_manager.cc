#include "self_play/batch_manager.h"

#include <cassert>
#include <chrono>

// ---------------------------------------------------------------------------
// InferenceBatch
// ---------------------------------------------------------------------------
InferenceBatch::InferenceBatch(int max_t, bool use_pinned) : max_tensors(max_t) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (use_pinned) {
        opts = opts.pinned_memory(true);
    }
    storage = torch::empty({max_tensors, kTensorSize}, opts);
    data_ptr = storage.data_ptr<float>();
    entries.reserve(max_tensors);
}

void InferenceBatch::reset() {
    reserved_count.store(0, std::memory_order_relaxed);
    committed_count.store(0, std::memory_order_relaxed);
    entries.clear();
}

// ---------------------------------------------------------------------------
// BatchManager
// ---------------------------------------------------------------------------
BatchManager::BatchManager(int num_batches, int max_tensors, bool use_pinned, int timeout_ms)
    : max_tensors_(max_tensors), timeout_ms_(timeout_ms) {
    
    all_batches_.reserve(num_batches);
    for (int i = 0; i < num_batches; ++i) {
        all_batches_.push_back(std::make_unique<InferenceBatch>(max_tensors, use_pinned));
        free_batches_.push_back(all_batches_.back().get());
    }
}

BatchManager::~BatchManager() {
    shutdown();
}

void BatchManager::shutdown() {
    shutdown_.store(true, std::memory_order_relaxed);
    pool_cv_.notify_all();
    ready_cv_.notify_all();
}

float* BatchManager::reserve(int req_count, InferenceBatch*& out_batch, int& out_offset) {
    // A request larger than a whole batch would loop forever below.
    assert(req_count > 0 && req_count <= max_tensors_);

    std::unique_lock<std::mutex> lock(active_mtx_);

    while (!shutdown_.load(std::memory_order_relaxed)) {
        if (!active_batch_) {
            // Get a free batch from the pool
            std::unique_lock<std::mutex> pool_lock(pool_mtx_);
            pool_cv_.wait(pool_lock, [this]() { return !free_batches_.empty() || shutdown_.load(std::memory_order_relaxed); });
            if (shutdown_.load(std::memory_order_relaxed)) return nullptr;

            active_batch_ = free_batches_.back();
            free_batches_.pop_back();
            pool_lock.unlock();

            active_batch_->reset();
            active_batch_start_ = std::chrono::steady_clock::now();
        }

        // Try to reserve
        int current_reserved = active_batch_->reserved_count.load(std::memory_order_relaxed);
        if (current_reserved + req_count <= max_tensors_) {
            // Success
            out_offset = current_reserved;
            active_batch_->reserved_count.fetch_add(req_count, std::memory_order_relaxed);
            out_batch = active_batch_;
            return active_batch_->data_ptr + (out_offset * kTensorSize);
        }

        // Active batch is full, flush it and try again in the next loop
        InferenceBatch* flushed = flush_active_batch_locked();
        if (flushed) {
            std::lock_guard<std::mutex> ready_lock(ready_mtx_);
            ready_batches_.push_back(flushed);
            ready_cv_.notify_one();
        }
    }

    return nullptr;
}

InferenceBatch* BatchManager::flush_active_batch_locked() {
    if (!active_batch_) return nullptr;
    
    // We just take it out of active. The LAST worker to commit will actually push it to ready_batches_.
    // However, if there are NO outstanding commits (rare but possible if a timeout occurred and it was empty),
    // we handle that.
    InferenceBatch* b = active_batch_;
    active_batch_ = nullptr;

    int reserved = b->reserved_count.load(std::memory_order_relaxed);
    int committed = b->committed_count.load(std::memory_order_relaxed);

    if (reserved == 0) {
        // Empty batch, just recycle
        std::lock_guard<std::mutex> pool_lock(pool_mtx_);
        free_batches_.push_back(b);
        pool_cv_.notify_one();
        return nullptr;
    } else if (reserved == committed) {
        // All reserved slots are already committed. Ready to push!
        return b;
    }
    // Else: there are outstanding commits. The worker calling commit() will push it.
    return nullptr;
}

void BatchManager::commit(InferenceBatch* batch, GameInstance* game, int offset, int req_count) {
    if (!batch) return;

    {
        std::lock_guard<std::mutex> lock(batch->entries_mtx);
        batch->entries.push_back({game, offset, req_count});
    }

    int prev_committed = batch->committed_count.fetch_add(req_count, std::memory_order_relaxed);
    int new_committed = prev_committed + req_count;

    // Check if we are the LAST commit for a batch that is NO LONGER the active batch.
    // If the batch is still active, we DO NOT push it (unless we want to push full active batches immediately).
    // Actually, if new_committed == max_tensors_, it's definitively full!
    
    bool should_push = false;
    
    if (new_committed == max_tensors_) {
        // It's perfectly full. If it's still active, we must flush it.
        std::lock_guard<std::mutex> lock(active_mtx_);
        if (active_batch_ == batch) {
            active_batch_ = nullptr;
        }
        should_push = true;
    } else {
        // It's not perfectly full. Did it get flushed due to timeout or being nearly full?
        // If it's not the active batch AND all reserved are committed, push it.
        std::lock_guard<std::mutex> lock(active_mtx_);
        if (active_batch_ != batch && new_committed == batch->reserved_count.load(std::memory_order_relaxed)) {
            should_push = true;
        }
    }

    if (should_push) {
        std::lock_guard<std::mutex> ready_lock(ready_mtx_);
        ready_batches_.push_back(batch);
        ready_cv_.notify_one();
    }
}

InferenceBatch* BatchManager::pop_ready_batch() {
    std::unique_lock<std::mutex> lock(ready_mtx_);
    while (ready_batches_.empty() && !shutdown_.load(std::memory_order_relaxed)) {
        // Unlock ready_mtx_ before checking active_batch_ to prevent lock inversion deadlock
        lock.unlock();

        auto now = std::chrono::steady_clock::now();
        bool flushed = false;
        
        InferenceBatch* flushed_batch = nullptr;
        {
            std::lock_guard<std::mutex> act_lock(active_mtx_);
            if (active_batch_ && active_batch_->reserved_count.load(std::memory_order_relaxed) > 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - active_batch_start_).count();
                if (elapsed >= timeout_ms_) {
                    flushed_batch = flush_active_batch_locked();
                    flushed = true;
                }
            }
        }

        // Re-lock ready_mtx_ before modifying ready_batches_ or calling wait_for
        lock.lock();

        if (flushed_batch) {
            ready_batches_.push_back(flushed_batch);
            // No need to notify, since we are the thread waiting! But we can just continue to pick it up!
        }

        if (flushed) {
            // We just flushed, it might have been pushed to ready_batches_ immediately,
            // or we might need to wait for a worker to finish committing.
            // Continue the loop to check ready_batches_.
            continue;
        }

        ready_cv_.wait_for(lock, std::chrono::milliseconds(1));
    }

    if (shutdown_.load(std::memory_order_relaxed) && ready_batches_.empty()) {
        return nullptr;
    }

    InferenceBatch* b = ready_batches_.front();
    ready_batches_.pop_front();
    return b;
}

void BatchManager::recycle_batch(InferenceBatch* batch) {
    if (!batch) return;
    std::lock_guard<std::mutex> lock(pool_mtx_);
    free_batches_.push_back(batch);
    pool_cv_.notify_one();
}
