#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "engine/rng.h"
#include "self_play/training_data.h"

// ---------------------------------------------------------------------------
// ReplayBuffer — thread-safe ring buffer for training samples.
//
// Fixed capacity: when full, oldest samples are overwritten. Binary
// serialization (save/load) supports warm-starting from a checkpoint.
// ---------------------------------------------------------------------------
class ReplayBuffer {
public:
    explicit ReplayBuffer(int capacity);

    /// Add a single sample (thread-safe).
    void add(const TrainingSample& sample);

    /// Add multiple samples at once (thread-safe).
    void add_batch(const TrainingSample* samples, int count);

    /// Sample `count` items uniformly at random WITH replacement.
    ///
    /// @param out   Pre-allocated output array of at least `count` elements.
    /// @param count Number of samples requested.
    /// @param rng   Caller-supplied RNG (not mutex-protected — caller must not
    ///              share the same rng across threads).
    /// @return      Actual number written (min(count, size())).
    int sample_batch(TrainingSample* out, int count, RNG& rng) const;

    /// Current number of samples stored.
    int size() const;

    /// Maximum capacity.
    int capacity() const { return capacity_; }

    /// Save to binary file. Throws std::runtime_error on I/O failure.
    void save(const std::string& path) const;

    /// Load from binary file.
    /// @return false if the file does not exist; throws on corrupt data.
    bool load(const std::string& path);

private:
    int                         capacity_;
    std::vector<TrainingSample> data_;
    int                         write_pos_   = 0;
    int                         total_added_ = 0;
    mutable std::mutex          mutex_;
};
