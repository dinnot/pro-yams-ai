#include "training/replay_buffer.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "engine/tensor.h"  // kTensorSize (via TrainingSample)

// Binary file format:
//   [0]   uint32_t  magic    = 0x52504C42 ("RPLB")
//   [4]   uint32_t  version  = 1
//   [8]   int32_t   capacity
//   [12]  int32_t   size     (number of valid samples stored)
//   [16]  int32_t   write_pos
//   [20]  int32_t   padding (reserved)
//   [24]  TrainingSample[size]  — stored in logical order (oldest first)

static constexpr uint32_t kMagic   = 0x52504C42u;
static constexpr uint32_t kVersion = 1u;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ReplayBuffer::ReplayBuffer(int capacity)
    : capacity_(capacity), data_(static_cast<size_t>(capacity)) {}

// ---------------------------------------------------------------------------
// add / add_batch
// ---------------------------------------------------------------------------

void ReplayBuffer::add(const TrainingSample& sample) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_[static_cast<size_t>(write_pos_)] = sample;
    write_pos_ = (write_pos_ + 1) % capacity_;
    if (total_added_ < capacity_) {
        ++total_added_;
    }
}

void ReplayBuffer::add_batch(const TrainingSample* samples, int count) {
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

// ---------------------------------------------------------------------------
// size
// ---------------------------------------------------------------------------

int ReplayBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::min(total_added_, capacity_);
}

// ---------------------------------------------------------------------------
// sample_batch — uniform random with replacement
// ---------------------------------------------------------------------------

int ReplayBuffer::sample_batch(TrainingSample* out, int count, RNG& rng) const {
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

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------

void ReplayBuffer::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int n = std::min(total_added_, capacity_);

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("ReplayBuffer::save: cannot open " + path);

    uint32_t magic   = kMagic;
    uint32_t version = kVersion;
    int32_t  cap     = static_cast<int32_t>(capacity_);
    int32_t  size    = static_cast<int32_t>(n);
    int32_t  wpos    = static_cast<int32_t>(write_pos_);
    int32_t  pad     = 0;

    f.write(reinterpret_cast<const char*>(&magic),   sizeof(magic));
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));
    f.write(reinterpret_cast<const char*>(&cap),     sizeof(cap));
    f.write(reinterpret_cast<const char*>(&size),    sizeof(size));
    f.write(reinterpret_cast<const char*>(&wpos),    sizeof(wpos));
    f.write(reinterpret_cast<const char*>(&pad),     sizeof(pad));

    // Write samples in logical order: oldest first.
    // If not yet full, data is in [0, size-1] with write_pos_ == size.
    // If full, oldest sample is at write_pos_, wrapping around.
    if (n < capacity_) {
        f.write(reinterpret_cast<const char*>(data_.data()),
                static_cast<std::streamsize>(n) * sizeof(TrainingSample));
    } else {
        // write_pos_ points to oldest
        int tail = capacity_ - write_pos_;
        f.write(reinterpret_cast<const char*>(data_.data() + write_pos_),
                static_cast<std::streamsize>(tail) * sizeof(TrainingSample));
        f.write(reinterpret_cast<const char*>(data_.data()),
                static_cast<std::streamsize>(write_pos_) * sizeof(TrainingSample));
    }

    if (!f) throw std::runtime_error("ReplayBuffer::save: write error to " + path);
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

bool ReplayBuffer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;  // file not found

    uint32_t magic, version;
    int32_t  cap, size, wpos, pad;

    f.read(reinterpret_cast<char*>(&magic),   sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    f.read(reinterpret_cast<char*>(&cap),     sizeof(cap));
    f.read(reinterpret_cast<char*>(&size),    sizeof(size));
    f.read(reinterpret_cast<char*>(&wpos),    sizeof(wpos));
    f.read(reinterpret_cast<char*>(&pad),     sizeof(pad));

    if (!f || magic != kMagic || version != kVersion)
        throw std::runtime_error("ReplayBuffer::load: corrupt file " + path);

    std::lock_guard<std::mutex> lock(mutex_);

    // Clamp to our capacity: if saved capacity differs, just load as many as fit.
    int load_count = static_cast<int>(size);
    if (load_count > capacity_) load_count = capacity_;

    f.read(reinterpret_cast<char*>(data_.data()),
           static_cast<std::streamsize>(load_count) * sizeof(TrainingSample));
    if (!f)
        throw std::runtime_error("ReplayBuffer::load: data truncated in " + path);

    write_pos_   = load_count % capacity_;
    total_added_ = load_count;
    return true;
}
