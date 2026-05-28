#pragma once

#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "engine/game_traits.h"
#include "engine/rng.h"
#include "self_play/training_data.h"  // TrainingSampleT

// ---------------------------------------------------------------------------
// DistilReplayBufferT<Traits> — fixed-capacity, ring-buffered replay queue for the
// distillation pipeline.
//
// Replaces the previous one-pass ShuffleQueueT. The teacher is fixed, so
// there is no policy drift; reuse is therefore safe and amortises expensive
// teacher inference / worker MCTS over multiple gradient updates.
//
// Semantics:
//   - Producers (workers) append to the ring via add_batch().
//   - Consumer (trainer) calls draw_batch() to receive `batch_size` samples
//     drawn uniformly at random WITH REPLACEMENT from the current buffer.
//   - After each draw, FIFO-evict ⌊eviction_credit⌋ of the oldest samples,
//     where eviction_credit accumulates `batch_size / samples_per_train` per
//     draw (fractional remainder carried across batches). This targets an
//     expected `samples_per_train` (K) draws per sample over its lifetime
//     in the buffer.
//       * K=1.0 ⇒ evict B per draw  ⇒ each sample used ≈ once before eviction
//                                      (parity with the old one-pass queue).
//       * K=2.0 ⇒ evict B/2 per draw ⇒ each sample used ≈ twice on average.
//       * K=N   ⇒ evict B/N per draw ⇒ each sample used ≈ N times.
//   - draw_batch returns 0 only after stop() has been called and the buffer
//     is fully drained.
//
// Steady-state math (for the curious): if capacity is N, batch size B, and
// the system is fully balanced (producers keep up), each sample lives in the
// buffer for N·K/B batches and is drawn with probability 1/N per individual
// sample draw → expected draws per sample = (B/N)·(N·K/B) = K. ✓
//
// Thread-safety: a single mutex covers all member access. Two condition
// variables: cv_consumer_ wakes the (single) drawer; cv_producers_ wakes
// producers blocked on the capacity cap.
//
// Memory / backpressure: add_batch() blocks once `size_ == capacity_`. Slots
// only free up when the drawer evicts. This is the natural back-pressure
// signal — if producers consistently block, either workers are too fast for
// the configured K (good problem: lower K) or the buffer is too small.
//
// Lock-holding policy: the per-batch memcpy (B × kTensorSize floats — easily
// 100+ MB for a 16K batch in 2v2) happens OUTSIDE the mutex. Under the lock
// we sample the indices; once unlocked, producers can keep appending into
// the free region but cannot overwrite our sampled indices (those slots are
// still "live" until the post-memcpy eviction advances the tail).
//
// First-draw warm-up: `min_samples_to_start` blocks the first draw until the
// buffer has accumulated enough samples for good decorrelation. After the
// first batch is served, subsequent draws only require `size_ >= batch_size`.
// ---------------------------------------------------------------------------
template <typename Traits>
class DistilReplayBufferT {
public:
    using Sample = TrainingSampleT<Traits>;

    /// @param capacity              Fixed ring-buffer capacity (in samples).
    ///                              Producers block when the ring is full.
    /// @param min_samples_to_start  Minimum fill required for the very first
    ///                              draw — gives the trainer a decorrelated
    ///                              initial batch. Subsequent draws only need
    ///                              `size_ >= batch_size`.
    /// @param samples_per_train     Target expected uses per sample, ≥ 1.0.
    ///                              1.0 → one-pass parity; 2.0 → ~2× reuse; ...
    /// @param seed                  RNG seed for uniform sampling.
    DistilReplayBufferT(int capacity,
                  int min_samples_to_start,
                  double samples_per_train,
                  uint64_t seed = 0xC0FFEE12345678ULL)
        : capacity_(capacity),
          min_samples_to_start_(min_samples_to_start),
          samples_per_train_(samples_per_train),
          rng_(seed) {
        buffer_.resize(static_cast<size_t>(capacity_));
    }

    /// Producer side. Append `n` samples (thread-safe). Blocks while the
    /// buffer is full; bails out without appending if stop() has been called.
    void add_batch(const Sample* s, int n) {
        if (n <= 0) return;
        bool crossed_start_threshold = false;
        int written_total = 0;
        while (written_total < n) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_producers_.wait(lk, [this]() {
                return stop_.load(std::memory_order_relaxed) ||
                       size_ < capacity_;
            });
            if (stop_.load(std::memory_order_relaxed)) return;

            // Append as much as fits without exceeding capacity.
            const int free_slots = capacity_ - size_;
            const int to_write   = std::min(free_slots, n - written_total);
            const int old_size   = size_;
            for (int i = 0; i < to_write; ++i) {
                buffer_[static_cast<size_t>(head_)] = s[written_total + i];
                head_ = (head_ + 1) % capacity_;
            }
            size_         += to_write;
            written_total += to_write;

            // Only wake the consumer when we cross the warm-up threshold for
            // the first time — otherwise notify on every meaningful change
            // (any append that takes size_ above batch-fill is enough; we
            // wake unconditionally below only on the warm-up transition).
            if (!started_ &&
                old_size < min_samples_to_start_ &&
                size_ >= min_samples_to_start_) {
                crossed_start_threshold = true;
            }
            // We always notify the consumer when adding samples, because the
            // consumer may be blocked waiting on size_ >= batch_size (post
            // warm-up) and we don't track that threshold separately. Cost is
            // a single notify_one per add_batch call.
            cv_consumer_.notify_one();
            (void)crossed_start_threshold;
        }
    }

    /// Consumer side. Blocks until either a full `batch_size` can be filled
    /// or stop() has been called and the buffer is fully drained.
    ///
    /// Writes up to `batch_size` samples (uniformly sampled with replacement)
    /// directly into the trainer's flat arrays: `states` is
    /// [batch_size × kTensorSize], `targets` is [batch_size]. Returns the
    /// number actually written.
    int draw_batch(float* states, double* targets, int batch_size) {
        if (batch_size <= 0) return 0;
        constexpr int kT = Traits::kTensorSize;

        // Indices into buffer_ chosen under the lock; memcpy'd after release.
        std::vector<int> picks;
        picks.reserve(static_cast<size_t>(batch_size));

        int actual_batch = batch_size;
        bool draining = false;

        {
            std::unique_lock<std::mutex> lk(mu_);

            const int required = started_
                ? batch_size
                : std::max(batch_size, min_samples_to_start_);

            cv_consumer_.wait(lk, [&]() {
                return stop_.load(std::memory_order_relaxed) ||
                       size_ >= required;
            });

            if (size_ < required) {
                // stop() fired while we were blocked. Drain whatever's left
                // as a partial final batch; if nothing remains, return 0.
                if (size_ == 0) return 0;
                actual_batch = size_;
                draining = true;
            }

            started_ = true;

            // Sample `actual_batch` indices uniformly with replacement from
            // the current [tail_, tail_ + size_) live window.
            for (int i = 0; i < actual_batch; ++i) {
                const int offset = rng_.uniform_int(0, size_ - 1);
                picks.push_back((tail_ + offset) % capacity_);
            }
        }
        // Lock released. Producers may now append into the free region (size_
        // can grow), but they cannot overwrite our sampled indices because
        // eviction hasn't moved tail_ yet — the sampled slots are still live.

        for (int i = 0; i < actual_batch; ++i) {
            const Sample& src = buffer_[static_cast<size_t>(picks[i])];
            std::memcpy(states + static_cast<size_t>(i) * kT,
                        src.state, kT * sizeof(float));
            targets[i] = src.target;
        }

        // Post-memcpy: evict from the front. Producers blocked on the cap
        // will wake once we free slots.
        int evicted_now = 0;
        bool freed_space = false;
        {
            std::lock_guard<std::mutex> lk(mu_);

            if (draining) {
                // Final-drain mode: evict everything we just drew so the
                // buffer drains monotonically and the next draw_batch can
                // return 0.
                evicted_now = actual_batch;
            } else {
                eviction_credit_ += static_cast<double>(batch_size) /
                                    samples_per_train_;
                evicted_now = static_cast<int>(eviction_credit_);
                if (evicted_now > size_) evicted_now = size_;
                eviction_credit_ -= static_cast<double>(evicted_now);
            }

            if (evicted_now > 0) {
                tail_ = (tail_ + evicted_now) % capacity_;
                size_ -= evicted_now;
                freed_space = true;
            }
            total_drawn_ += actual_batch;
        }
        if (freed_space) cv_producers_.notify_all();

        return actual_batch;
    }

    /// Wake any blocked drawer and signal end-of-stream. Subsequent
    /// draw_batch() calls drain everything left in the buffer (as smaller
    /// final batches), then return 0. Blocked producers wake and discard.
    void stop() {
        stop_.store(true, std::memory_order_relaxed);
        cv_consumer_.notify_all();
        cv_producers_.notify_all();
    }

    // -- Diagnostics --------------------------------------------------------

    int size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return size_;
    }

    long total_drawn() const {
        std::lock_guard<std::mutex> lk(mu_);
        return total_drawn_;
    }

    int  capacity()           const { return capacity_; }
    double samples_per_train() const { return samples_per_train_; }

private:
    int    capacity_;
    int    min_samples_to_start_;
    double samples_per_train_;
    RNG    rng_;

    std::vector<Sample> buffer_;   // ring storage, fixed at capacity_
    int  head_ = 0;                // next write index
    int  tail_ = 0;                // oldest live index
    int  size_ = 0;                // current count of live samples

    bool   started_         = false;  // crossed min_samples_to_start once
    double eviction_credit_ = 0.0;    // accumulated fractional evictions
    long   total_drawn_     = 0;

    mutable std::mutex      mu_;
    std::condition_variable cv_consumer_;
    std::condition_variable cv_producers_;
    std::atomic<bool>       stop_{false};
};

// Backward-compat aliases.
using DistilReplayBuffer    = DistilReplayBufferT<Yams1v1>;
using DistilReplayBuffer2v2 = DistilReplayBufferT<Yams2v2>;
