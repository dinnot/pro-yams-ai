#include <gtest/gtest.h>
#include "engine/rng.h"

#include <array>
#include <algorithm>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------
TEST(RNG, Deterministic_SameSeedSameSequence) {
    RNG a(12345), b(12345);
    for (int i = 0; i < 1000; ++i)
        EXPECT_EQ(a.next(), b.next());
}

TEST(RNG, DifferentSeeds_DifferentSequence) {
    RNG a(1), b(2);
    bool any_diff = false;
    for (int i = 0; i < 100; ++i)
        if (a.next() != b.next()) { any_diff = true; break; }
    EXPECT_TRUE(any_diff);
}

// ---------------------------------------------------------------------------
// uniform_int
// ---------------------------------------------------------------------------
TEST(RNG, UniformInt_AlwaysInRange) {
    RNG rng(99);
    for (int i = 0; i < 10000; ++i) {
        int v = rng.uniform_int(1, 6);
        EXPECT_GE(v, 1);
        EXPECT_LE(v, 6);
    }
}

TEST(RNG, UniformInt_SingleValue) {
    RNG rng(7);
    for (int i = 0; i < 100; ++i)
        EXPECT_EQ(rng.uniform_int(42, 42), 42);
}

TEST(RNG, UniformInt_AllValuesReached) {
    // With enough draws, every value in [1,6] should appear
    RNG rng(0);
    std::unordered_set<int> seen;
    for (int i = 0; i < 10000; ++i)
        seen.insert(rng.uniform_int(1, 6));
    EXPECT_EQ(static_cast<int>(seen.size()), 6);
}

TEST(RNG, UniformInt_LargeRange) {
    RNG rng(555);
    for (int i = 0; i < 10000; ++i) {
        int v = rng.uniform_int(0, 999);
        EXPECT_GE(v, 0);
        EXPECT_LE(v, 999);
    }
}

// ---------------------------------------------------------------------------
// shuffle
// ---------------------------------------------------------------------------
TEST(RNG, Shuffle_IsPermutation) {
    RNG rng(77);
    std::array<int, 6> arr = {8, 10, 12, 14, 16, 18};
    auto sorted = arr;
    rng.shuffle(arr);
    auto shuffled_sorted = arr;
    std::sort(shuffled_sorted.begin(), shuffled_sorted.end());
    EXPECT_EQ(shuffled_sorted, sorted);
}

TEST(RNG, Shuffle_ChangesOrder) {
    // With 6! = 720 possible permutations, same seed rarely preserves order
    RNG rng(1);
    std::array<int, 6> orig = {1, 2, 3, 4, 5, 6};
    std::array<int, 6> arr = orig;
    rng.shuffle(arr);
    // Not identical to original (true for seed=1 — verified)
    bool changed = (arr != orig);
    EXPECT_TRUE(changed);
}

// ---------------------------------------------------------------------------
// jump / long_jump — just verify they change state
// ---------------------------------------------------------------------------
TEST(RNG, Jump_ChangesState) {
    RNG a(42), b(42);
    b.jump();
    // After jump, sequences should differ
    bool any_diff = false;
    for (int i = 0; i < 10; ++i)
        if (a.next() != b.next()) { any_diff = true; break; }
    EXPECT_TRUE(any_diff);
}

TEST(RNG, LongJump_ChangesState) {
    RNG a(42), b(42);
    b.long_jump();
    bool any_diff = false;
    for (int i = 0; i < 10; ++i)
        if (a.next() != b.next()) { any_diff = true; break; }
    EXPECT_TRUE(any_diff);
}
