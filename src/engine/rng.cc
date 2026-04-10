#include "engine/rng.h"

#include <cassert>
#include <cstdint>

// ---------------------------------------------------------------------------
// splitmix64 — used to expand a single 64-bit seed into 256-bit xoshiro state.
// Public domain, by Sebastiano Vigna.
// ---------------------------------------------------------------------------
static inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// xoshiro256++ core
// ---------------------------------------------------------------------------
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

RNG::RNG(uint64_t seed) {
    state_[0] = splitmix64(seed);
    state_[1] = splitmix64(seed);
    state_[2] = splitmix64(seed);
    state_[3] = splitmix64(seed);
}

uint64_t RNG::next() {
    const uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
    const uint64_t t = state_[1] << 17;

    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];

    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);

    return result;
}

// ---------------------------------------------------------------------------
// Lemire's nearly divisionless method for unbiased bounded integers.
// Reference: https://lemire.me/blog/2019/06/06/nearly-divisionless-random-integer-generation-on-various-systems/
// ---------------------------------------------------------------------------
int RNG::uniform_int(int min, int max) {
    assert(min <= max);
    uint32_t range = static_cast<uint32_t>(max - min + 1);
    uint64_t x = next();
    uint32_t hi = static_cast<uint32_t>(x >> 32);
    uint64_t m = static_cast<uint64_t>(hi) * static_cast<uint64_t>(range);
    uint32_t l = static_cast<uint32_t>(m);
    if (l < range) {
        uint32_t t = static_cast<uint32_t>(-static_cast<int32_t>(range)) % range;
        while (l < t) {
            x = next();
            hi = static_cast<uint32_t>(x >> 32);
            m = static_cast<uint64_t>(hi) * static_cast<uint64_t>(range);
            l = static_cast<uint32_t>(m);
        }
    }
    return min + static_cast<int>(m >> 32);
}

// ---------------------------------------------------------------------------
// uniform_double — [0, 1) via 53-bit mantissa trick
// ---------------------------------------------------------------------------
double RNG::uniform_double() {
    // Use the top 53 bits of a 64-bit value for a double in [0, 1).
    return (next() >> 11) * (1.0 / (UINT64_C(1) << 53));
}

// ---------------------------------------------------------------------------
// jump() — advance state by 2^128 calls
// ---------------------------------------------------------------------------
void RNG::jump() {
    static const uint64_t JUMP[] = {
        UINT64_C(0x180ec6d33cfd0aba), UINT64_C(0xd5a61266f0c9392c),
        UINT64_C(0xa9582618e03fc9aa), UINT64_C(0x39abdc4529b1661c)
    };
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 64; b++) {
            if (JUMP[i] & (UINT64_C(1) << b)) {
                s0 ^= state_[0]; s1 ^= state_[1];
                s2 ^= state_[2]; s3 ^= state_[3];
            }
            next();
        }
    state_[0] = s0; state_[1] = s1;
    state_[2] = s2; state_[3] = s3;
}

// ---------------------------------------------------------------------------
// long_jump() — advance state by 2^192 calls
// ---------------------------------------------------------------------------
void RNG::long_jump() {
    static const uint64_t LONG_JUMP[] = {
        UINT64_C(0x76e15d3efefdcbbf), UINT64_C(0xc5004e441c522fb3),
        UINT64_C(0x77710069854ee241), UINT64_C(0x39109bb02acbe635)
    };
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & (UINT64_C(1) << b)) {
                s0 ^= state_[0]; s1 ^= state_[1];
                s2 ^= state_[2]; s3 ^= state_[3];
            }
            next();
        }
    state_[0] = s0; state_[1] = s1;
    state_[2] = s2; state_[3] = s3;
}
