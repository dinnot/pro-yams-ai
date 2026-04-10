#pragma once

#include <cstdint>
#include <array>

// ---------------------------------------------------------------------------
// RNG — xoshiro256++ wrapped in a C++ class.
//
// Reference implementation by David Blackman and Sebastiano Vigna (public domain).
// See: https://prng.di.unimi.it/
//
// Properties:
//   - Passes all known statistical tests (BigCrush, PractRand)
//   - ~1 ns per call — significantly faster than std::mt19937 (~3 ns)
//   - 32-byte state (vs 2.5KB for mt19937)
//   - Deterministic: same seed → same sequence across platforms
//   - jump() skips 2^128 values (for non-overlapping parallel streams)
// ---------------------------------------------------------------------------
class RNG {
public:
    /// Construct with a 64-bit seed.
    /// Uses splitmix64 to expand the seed into the 256-bit state,
    /// as recommended by the xoshiro256++ authors.
    explicit RNG(uint64_t seed);

    /// Generate a raw 64-bit random value (core xoshiro256++ algorithm).
    uint64_t next();

    /// Generate a uniform integer in [min, max] inclusive.
    /// Uses Lemire's nearly divisionless method — nearly branch-free,
    /// extremely fast, and provably unbiased.
    int uniform_int(int min, int max);

    /// Generate a uniform double in [0, 1).
    double uniform_double();

    /// Shuffle an array in-place using Fisher-Yates.
    template<typename T, std::size_t N>
    void shuffle(std::array<T, N>& arr) {
        for (int i = static_cast<int>(N) - 1; i > 0; --i) {
            int j = uniform_int(0, i);
            T tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    /// Advance state by 2^128 calls (for generating non-overlapping streams).
    void jump();

    /// Advance state by 2^192 calls.
    void long_jump();

private:
    uint64_t state_[4];
};
