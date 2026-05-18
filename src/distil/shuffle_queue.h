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
// ShuffleQueueT<Traits> — one-pass, shuffled producer/consumer queue for the
// distillation pipeline.
//
// Replaces ReplayBufferT for distill mode. The teacher is fixed, so there is
// no policy drift and no benefit to mixing samples across time — each sample
// should be trained on exactly once. But consecutive samples from one worker
// are heavily correlated (same board, different placements), so we still
// shuffle within bounded chunks before serving them to the trainer.
//
// Semantics:
//   - Producers (workers) append to `accumulating_` via add_batch().
//   - Consumer (trainer) calls draw_batch() to receive shuffled samples.
//   - When `serving_` is empty and `accumulating_` has at least one chunk's
//     worth (or `min_chunk_size_to_start_` for the very first rotation),
//     a rotation atomically moves accumulating_ → serving_ and generates a
//     fresh random permutation over it.
//   - Each sample appears in exactly one draw_batch result.
//   - draw_batch returns 0 only after stop() has been called and everything
//     in both buffers has been drained.
//
// Thread-safety: a single mutex covers all member access. cv_ wakes a
// blocked drawer when producers add samples or when stop() fires.
//
// Memory / backpressure: rotation moves all of accumulating_ (not just
// chunk_size_), so chunks can be larger than chunk_size_ if production
// briefly outpaces consumption. To stop unbounded growth — and to stop
// workers from burning CPU / teacher inference on samples that will never
// be trained on — add_batch() blocks once
//   (accumulating + serving_remaining) >= max_buffered_samples.
// The cap is intentionally soft: blocked producers still flush their full
// batch once unblocked, so the buffered total can briefly overshoot by up
// to one batch per producer. Default is INT_MAX (unbounded); the distil
// pipeline passes a smaller cap (typically 2M samples).
// ---------------------------------------------------------------------------
template <typename Traits>
class ShuffleQueueT {
public:
    using Sample = TrainingSampleT<Traits>;

    static constexpr int kUnbounded = INT_MAX;

    /// @param chunk_size               Threshold for triggering a rotation
    ///                                 after the first chunk has been served.
    /// @param min_chunk_size_to_start  Threshold for the very first rotation
    ///                                 (typically smaller so the trainer can
    ///                                 begin while workers ramp up).
    /// @param max_buffered_samples     Soft cap on (accumulating + serving
    ///                                 remaining). add_batch blocks above it.
    ///                                 Default kUnbounded preserves the
    ///                                 prior behaviour for unit tests.
    /// @param seed                     RNG seed for per-chunk permutations.
    ShuffleQueueT(int chunk_size, int min_chunk_size_to_start,
                  int max_buffered_samples = kUnbounded,
                  uint64_t seed = 0xC0FFEE12345678ULL)
        : chunk_size_(chunk_size),
          min_chunk_size_to_start_(min_chunk_size_to_start),
          max_buffered_samples_(max_buffered_samples),
          rng_(seed) {
        accumulating_.reserve(static_cast<size_t>(chunk_size_));
    }

    /// Producer side. Append `n` samples (thread-safe). Blocks while the
    /// buffer is at the soft cap; bails out without appending if stop()
    /// has been called.
    void add_batch(const Sample* s, int n) {
        if (n <= 0) return;
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]() {
            return stop_.load(std::memory_order_relaxed) ||
                   buffered_size_locked() < max_buffered_samples_;
        });
        if (stop_.load(std::memory_order_relaxed)) return;  // discard on shutdown
        accumulating_.insert(accumulating_.end(), s, s + n);
        // Wake the drawer (and any other waiters re-checking conditions).
        cv_.notify_all();
    }

    /// Consumer side. Blocks until either a full `batch_size` can be filled
    /// or stop() has been called and the queue is fully drained.
    ///
    /// Writes up to `batch_size` samples directly into the trainer's flat
    /// arrays: `states` is [batch_size × kTensorSize], `targets` is
    /// [batch_size]. Returns the number actually written.
    ///
    /// Returns 0 only after stop() and full drain. Otherwise returns
    /// batch_size (or fewer only when stop drained a partial tail).
    ///
    /// Lock-holding policy: the big per-batch memcpy (batch_size × kTensorSize
    /// floats — easily 100+ MB for a 16K batch in 2v2) happens OUTSIDE the
    /// mutex. Under the lock we only claim sample indices and advance
    /// serve_pos_. This frees the producer threads to keep pushing while
    /// the trainer is copying. Safe because only this thread (the trainer)
    /// ever mutates serving_ (via rotate_locked), so the indices and the
    /// serving_ contents stay valid for the duration of the post-unlock copy.
    ///
    /// At chunk boundaries (rare — once per chunk_size samples) we materialize
    /// the small claim from the outgoing chunk in-lock before rotating, then
    /// the (large) claim from the new chunk is deferred to post-unlock as
    /// usual. So worst case the in-lock memcpy is bounded by ONE chunk's tail
    /// and the bulk always escapes the lock.
    int draw_batch(float* states, double* targets, int batch_size) {
        if (batch_size <= 0) return 0;
        constexpr int kT = Traits::kTensorSize;

        int written = 0;
        int deferred_start = -1;          // output index where deferred range begins
        std::vector<int> deferred_idx;    // sample indices into serving_ to memcpy later
        deferred_idx.reserve(static_cast<size_t>(batch_size));

        // Called inside the lock when we need to advance past the current chunk
        // (rotation, stop drain). Materializes whatever's deferred so the post-
        // unlock copy only has to handle the most-recent chunk's claim.
        auto flush_pending = [&]() {
            if (deferred_start < 0) return;
            for (size_t i = 0; i < deferred_idx.size(); ++i) {
                const Sample& src = serving_[static_cast<size_t>(deferred_idx[i])];
                std::memcpy(states + static_cast<size_t>(deferred_start) * kT +
                                static_cast<size_t>(i) * kT,
                            src.state, kT * sizeof(float));
                targets[deferred_start + static_cast<int>(i)] = src.target;
            }
            deferred_idx.clear();
            deferred_start = -1;
        };

        std::unique_lock<std::mutex> lk(mu_);
        while (written < batch_size) {
            int available = static_cast<int>(serving_.size()) - serve_pos_;
            if (available > 0) {
                int take = std::min(available, batch_size - written);
                // Only the most-recent claim is deferred; flush any prior range.
                if (deferred_start >= 0) flush_pending();
                deferred_start = written;
                for (int i = 0; i < take; ++i) {
                    deferred_idx.push_back(
                        perm_[static_cast<size_t>(serve_pos_ + i)]);
                }
                serve_pos_ += take;
                written += take;
                if (written == batch_size) break;
            }

            // serving_ exhausted — try to rotate.
            const int threshold = first_chunk_
                ? min_chunk_size_to_start_
                : chunk_size_;
            if (static_cast<int>(accumulating_.size()) >= threshold) {
                // Rotation will replace serving_'s contents; resolve deferred first.
                flush_pending();
                rotate_locked();
                continue;
            }

            // Stop fired — drain whatever's left unconditionally.
            if (stop_.load(std::memory_order_relaxed)) {
                if (!accumulating_.empty()) {
                    flush_pending();
                    rotate_locked();
                    continue;
                }
                break;  // truly empty
            }

            // Block until a producer adds more or stop fires.
            cv_.wait(lk);
        }
        total_drawn_ += written;
        // Drained samples may have freed slots — wake blocked producers.
        if (written > 0) cv_.notify_all();
        lk.unlock();

        // Post-unlock bulk memcpy. serving_ stays stable: only this thread
        // can mutate it (via rotate_locked, only called inside draw_batch),
        // and we won't re-enter draw_batch until the caller returns from us.
        if (deferred_start >= 0) {
            for (size_t i = 0; i < deferred_idx.size(); ++i) {
                const Sample& src = serving_[static_cast<size_t>(deferred_idx[i])];
                std::memcpy(states + static_cast<size_t>(deferred_start) * kT +
                                static_cast<size_t>(i) * kT,
                            src.state, kT * sizeof(float));
                targets[deferred_start + static_cast<int>(i)] = src.target;
            }
        }
        return written;
    }

    /// Wake any blocked drawer and signal end-of-stream. Subsequent
    /// draw_batch() calls drain everything left in both buffers, then return 0.
    void stop() {
        stop_.store(true, std::memory_order_relaxed);
        cv_.notify_all();
    }

    // -- Diagnostics --------------------------------------------------------

    int accumulating_size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<int>(accumulating_.size());
    }

    int serving_remaining() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<int>(serving_.size()) - serve_pos_;
    }

    long total_drawn() const {
        std::lock_guard<std::mutex> lk(mu_);
        return total_drawn_;
    }

    /// Combined buffered count (accumulating + serving_remaining). Cheap
    /// helper for orchestrator metrics.
    int buffered_size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return buffered_size_locked();
    }

    int max_buffered_samples() const { return max_buffered_samples_; }

private:
    int buffered_size_locked() const {
        return static_cast<int>(accumulating_.size()) +
               (static_cast<int>(serving_.size()) - serve_pos_);
    }


    /// Move accumulating_ → serving_ and produce a fresh Fisher-Yates
    /// permutation over the new serving_. Caller must hold mu_.
    void rotate_locked() {
        serving_ = std::move(accumulating_);
        accumulating_.clear();
        accumulating_.reserve(static_cast<size_t>(chunk_size_));

        const int n = static_cast<int>(serving_.size());
        perm_.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            perm_[static_cast<size_t>(i)] = i;
        }
        // Fisher-Yates on the index array (cheaper than shuffling samples).
        for (int i = n - 1; i > 0; --i) {
            int j = rng_.uniform_int(0, i);
            int tmp = perm_[static_cast<size_t>(i)];
            perm_[static_cast<size_t>(i)] = perm_[static_cast<size_t>(j)];
            perm_[static_cast<size_t>(j)] = tmp;
        }
        serve_pos_ = 0;
        first_chunk_ = false;
    }

    int  chunk_size_;
    int  min_chunk_size_to_start_;
    int  max_buffered_samples_;
    RNG  rng_;

    std::vector<Sample> accumulating_;
    std::vector<Sample> serving_;
    std::vector<int>    perm_;
    int  serve_pos_   = 0;
    bool first_chunk_ = true;
    long total_drawn_ = 0;

    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::atomic<bool>       stop_{false};
};

// Backward-compat aliases.
using ShuffleQueue    = ShuffleQueueT<Yams1v1>;
using ShuffleQueue2v2 = ShuffleQueueT<Yams2v2>;
