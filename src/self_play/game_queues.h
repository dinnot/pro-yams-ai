#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>

struct GameInstance;

// ---------------------------------------------------------------------------
// GameQueue — thread-safe FIFO queue of GameInstance pointers.
//
// Uses mutex + condition variable for blocking push/pop. Supports batch
// collection with timeout for the coordinator thread.
// ---------------------------------------------------------------------------
class GameQueue {
public:
    /// Push a game onto the queue (nullptr is a valid sentinel for shutdown).
    void push(GameInstance* game);

    /// Push a game to the front of the queue (used for returning rejected items).
    void push_front(GameInstance* game);

    /// Push multiple games at once (reduces lock acquisitions).
    void push_batch(GameInstance** games, int count);

    /// Push multiple games to the front of the queue, preserving their order.
    /// After this call, games[0] will be at the very front.
    void push_front_batch(GameInstance** games, int count);

    /// Pop a game. Blocks until one is available.
    /// May return nullptr if nullptr was explicitly pushed (shutdown sentinel).
    GameInstance* pop();

    /// Try to pop a game. Returns nullptr immediately if queue is empty.
    GameInstance* try_pop();

    /// Pop a game, waiting at most timeout_ms milliseconds.
    /// Returns nullptr on timeout or if nullptr was pushed as a sentinel.
    GameInstance* pop_with_timeout(int timeout_ms);

    /// Collect up to max_count games, waiting up to timeout_ms milliseconds.
    /// Returns the number of games collected (may be 0 on timeout).
    int collect(GameInstance** out, int max_count, int timeout_ms);

    /// Number of games currently in the queue.
    int size() const;

private:
    std::deque<GameInstance*> queue_;
    mutable std::mutex        mutex_;
    std::condition_variable   cv_;
};
