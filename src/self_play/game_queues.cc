#include "self_play/game_queues.h"

#include <chrono>

void GameQueue::push(GameInstance* game) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(game);
    }
    cv_.notify_one();
}

void GameQueue::push_front(GameInstance* game) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_front(game);
    }
    cv_.notify_one();
}

void GameQueue::push_batch(GameInstance** games, int count) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < count; ++i)
            queue_.push_back(games[i]);
    }
    cv_.notify_all();
}

void GameQueue::push_front_batch(GameInstance** games, int count) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Insert in reverse so games[0] ends up at the front.
        for (int i = count - 1; i >= 0; --i)
            queue_.push_front(games[i]);
    }
    cv_.notify_all();
}

GameInstance* GameQueue::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    GameInstance* g = queue_.front();
    queue_.pop_front();
    return g;
}

GameInstance* GameQueue::pop_with_timeout(int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_until(lock, deadline, [this] { return !queue_.empty(); });
    if (queue_.empty()) return nullptr;
    GameInstance* g = queue_.front();
    queue_.pop_front();
    return g;
}

GameInstance* GameQueue::try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return nullptr;
    GameInstance* g = queue_.front();
    queue_.pop_front();
    return g;
}

int GameQueue::collect(GameInstance** out, int max_count, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(mutex_);

    // Block until at least one item is available, or the timeout expires.
    if (queue_.empty()) {
        cv_.wait_until(lock, deadline, [this] { return !queue_.empty(); });
    }

    // Drain up to max_count items without further waiting.
    int count = 0;
    while (!queue_.empty() && count < max_count) {
        out[count++] = queue_.front();
        queue_.pop_front();
    }
    return count;
}

int GameQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(queue_.size());
}
