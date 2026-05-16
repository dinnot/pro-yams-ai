#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

#include "engine/game_traits.h"
#include "self_play/game_instance.h"

// ---------------------------------------------------------------------------
// GameQueueT — thread-safe FIFO queue of GameInstanceT<Traits>* pointers.
//
// Uses mutex + condition variable for blocking push/pop. Supports batch
// collection with timeout for the coordinator thread.
//
// Templated on Traits so 1v1 and 2v2 queues are distinct types (the queue
// never dereferences the GameInstance pointers, so the bodies are identical;
// the type parameter prevents accidental cross-variant wiring).
// ---------------------------------------------------------------------------
template <typename Traits>
class GameQueueT {
public:
    using Instance = GameInstanceT<Traits>;

    /// Push a game onto the queue (nullptr is a valid sentinel for shutdown).
    void push(Instance* game) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(game);
        }
        cv_.notify_one();
    }

    /// Push a game to the front of the queue (used for returning rejected items).
    void push_front(Instance* game) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_front(game);
        }
        cv_.notify_one();
    }

    /// Push multiple games at once (reduces lock acquisitions).
    void push_batch(Instance** games, int count) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int i = 0; i < count; ++i)
                queue_.push_back(games[i]);
        }
        // Wake exactly `count` waiters — notify_all would thunder-herd every
        // sleeping worker, contend on mutex_, and then send most back to sleep.
        for (int i = 0; i < count; ++i)
            cv_.notify_one();
    }

    /// Push multiple games to the front of the queue, preserving their order.
    /// After this call, games[0] will be at the very front.
    void push_front_batch(Instance** games, int count) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int i = count - 1; i >= 0; --i)
                queue_.push_front(games[i]);
        }
        for (int i = 0; i < count; ++i)
            cv_.notify_one();
    }

    /// Pop a game. Blocks until one is available.
    /// May return nullptr if nullptr was explicitly pushed (shutdown sentinel).
    Instance* pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
        Instance* g = queue_.front();
        queue_.pop_front();
        return g;
    }

    /// Try to pop a game. Returns nullptr immediately if queue is empty.
    Instance* try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return nullptr;
        Instance* g = queue_.front();
        queue_.pop_front();
        return g;
    }

    /// Pop a game, waiting at most timeout_ms milliseconds.
    /// Returns nullptr on timeout or if nullptr was pushed as a sentinel.
    Instance* pop_with_timeout(int timeout_ms) {
        auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(timeout_ms);
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_until(lock, deadline, [this] { return !queue_.empty(); });
        if (queue_.empty()) return nullptr;
        Instance* g = queue_.front();
        queue_.pop_front();
        return g;
    }

    /// Collect up to max_count games, waiting up to timeout_ms milliseconds.
    /// Returns the number of games collected (may be 0 on timeout).
    int collect(Instance** out, int max_count, int timeout_ms) {
        auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(timeout_ms);
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            cv_.wait_until(lock, deadline, [this] { return !queue_.empty(); });
        }
        int count = 0;
        while (!queue_.empty() && count < max_count) {
            out[count++] = queue_.front();
            queue_.pop_front();
        }
        return count;
    }

    /// Number of games currently in the queue.
    int size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(queue_.size());
    }

private:
    std::deque<Instance*>     queue_;
    mutable std::mutex        mutex_;
    std::condition_variable   cv_;
};

// Backward-compat aliases.
using GameQueue    = GameQueueT<Yams1v1>;
using GameQueue2v2 = GameQueueT<Yams2v2>;
