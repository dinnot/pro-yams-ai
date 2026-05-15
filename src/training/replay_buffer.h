#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/game_traits.h"
#include "engine/rng.h"
#include "self_play/training_data.h"

// ---------------------------------------------------------------------------
// ReplayBufferT<Traits> — thread-safe ring buffer for training samples.
//
// Fixed capacity: when full, oldest samples are overwritten. Binary
// serialization (save/load) supports warm-starting from a checkpoint.
//
// Templated so that the underlying TrainingSampleT<Traits> tensor size is
// known at compile time and 1v1/2v2 buffers cannot be cross-loaded.
// ---------------------------------------------------------------------------
template <typename Traits>
class ReplayBufferT {
public:
    using Sample = TrainingSampleT<Traits>;

    explicit ReplayBufferT(int capacity)
        : capacity_(capacity), data_(static_cast<size_t>(capacity)) {}

    /// Add a single sample (thread-safe).
    void add(const Sample& sample) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_[static_cast<size_t>(write_pos_)] = sample;
        write_pos_ = (write_pos_ + 1) % capacity_;
        if (total_added_ < capacity_) {
            ++total_added_;
        }
    }

    /// Add multiple samples at once (thread-safe).
    void add_batch(const Sample* samples, int count) {
        if (count <= 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < count; ++i) {
            data_[static_cast<size_t>(write_pos_)] = samples[i];
            write_pos_ = (write_pos_ + 1) % capacity_;
        }
        if (total_added_ < capacity_) {
            total_added_ += count;
            if (total_added_ > capacity_) total_added_ = capacity_;
        }
    }

    /// Sample `count` items uniformly at random WITH replacement.
    int sample_batch(Sample* out, int count, RNG& rng) const {
        std::lock_guard<std::mutex> lock(mutex_);
        int n = std::min(total_added_, capacity_);
        if (n <= 0) return 0;
        int actual = std::min(count, n);
        for (int i = 0; i < actual; ++i) {
            int idx = rng.uniform_int(0, n - 1);
            out[i] = data_[static_cast<size_t>(idx)];
        }
        return actual;
    }

    /// Current number of samples stored.
    int size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::min(total_added_, capacity_);
    }

    /// Maximum capacity.
    int capacity() const { return capacity_; }

    // -- Binary file format -------------------------------------------------
    //   [0]   uint32_t  magic    = 0x52504C42 ("RPLB")
    //   [4]   uint32_t  version  = 1
    //   [8]   int32_t   capacity
    //   [12]  int32_t   size
    //   [16]  int32_t   write_pos
    //   [20]  int32_t   padding (reserved)
    //   [24]  Sample[size]  — stored in logical order (oldest first)
    static constexpr uint32_t kMagic   = 0x52504C42u;
    static constexpr uint32_t kVersion = 1u;

    /// Save to binary file. Throws std::runtime_error on I/O failure.
    void save(const std::string& path) const {
        std::lock_guard<std::mutex> lock(mutex_);
        int n = std::min(total_added_, capacity_);

        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("ReplayBuffer::save: cannot open " + path);

        uint32_t magic   = kMagic;
        uint32_t version = kVersion;
        int32_t  cap     = static_cast<int32_t>(capacity_);
        int32_t  size_v  = static_cast<int32_t>(n);
        int32_t  wpos    = static_cast<int32_t>(write_pos_);
        int32_t  pad     = 0;

        f.write(reinterpret_cast<const char*>(&magic),   sizeof(magic));
        f.write(reinterpret_cast<const char*>(&version), sizeof(version));
        f.write(reinterpret_cast<const char*>(&cap),     sizeof(cap));
        f.write(reinterpret_cast<const char*>(&size_v),  sizeof(size_v));
        f.write(reinterpret_cast<const char*>(&wpos),    sizeof(wpos));
        f.write(reinterpret_cast<const char*>(&pad),     sizeof(pad));

        if (n < capacity_) {
            f.write(reinterpret_cast<const char*>(data_.data()),
                    static_cast<std::streamsize>(n) * sizeof(Sample));
        } else {
            int tail = capacity_ - write_pos_;
            f.write(reinterpret_cast<const char*>(data_.data() + write_pos_),
                    static_cast<std::streamsize>(tail) * sizeof(Sample));
            f.write(reinterpret_cast<const char*>(data_.data()),
                    static_cast<std::streamsize>(write_pos_) * sizeof(Sample));
        }

        if (!f) throw std::runtime_error("ReplayBuffer::save: write error to " + path);
    }

    /// Load from binary file.
    /// @return false if the file does not exist; throws on corrupt data.
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        uint32_t magic, version;
        int32_t  cap, size_v, wpos, pad;

        f.read(reinterpret_cast<char*>(&magic),   sizeof(magic));
        f.read(reinterpret_cast<char*>(&version), sizeof(version));
        f.read(reinterpret_cast<char*>(&cap),     sizeof(cap));
        f.read(reinterpret_cast<char*>(&size_v),  sizeof(size_v));
        f.read(reinterpret_cast<char*>(&wpos),    sizeof(wpos));
        f.read(reinterpret_cast<char*>(&pad),     sizeof(pad));

        if (!f || magic != kMagic || version != kVersion)
            throw std::runtime_error("ReplayBuffer::load: corrupt file " + path);

        std::lock_guard<std::mutex> lock(mutex_);

        int load_count = static_cast<int>(size_v);
        if (load_count > capacity_) load_count = capacity_;

        f.read(reinterpret_cast<char*>(data_.data()),
               static_cast<std::streamsize>(load_count) * sizeof(Sample));
        if (!f)
            throw std::runtime_error("ReplayBuffer::load: data truncated in " + path);

        write_pos_   = load_count % capacity_;
        total_added_ = load_count;
        return true;
    }

private:
    int                          capacity_;
    std::vector<Sample>          data_;
    int                          write_pos_   = 0;
    int                          total_added_ = 0;
    mutable std::mutex           mutex_;
};

// Backward-compat aliases.
using ReplayBuffer    = ReplayBufferT<Yams1v1>;
using ReplayBuffer2v2 = ReplayBufferT<Yams2v2>;
