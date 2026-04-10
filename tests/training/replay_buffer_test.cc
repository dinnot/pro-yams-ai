#include <gtest/gtest.h>
#include "training/replay_buffer.h"
#include "engine/rng.h"
#include "engine/tensor.h"

#include <cstring>
#include <filesystem>
#include <string>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static TrainingSample make_sample(float fill, double target) {
    TrainingSample s{};
    for (int i = 0; i < kTensorSize; ++i) s.state[i] = fill;
    s.target = target;
    return s;
}

// ---------------------------------------------------------------------------
// Basic add / size / capacity
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, InitiallyEmpty) {
    ReplayBuffer buf(100);
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.capacity(), 100);
}

TEST(ReplayBufferTest, AddIncreasesSize) {
    ReplayBuffer buf(10);
    buf.add(make_sample(1.0f, 0.5));
    EXPECT_EQ(buf.size(), 1);
    buf.add(make_sample(2.0f, 0.3));
    EXPECT_EQ(buf.size(), 2);
}

TEST(ReplayBufferTest, SizeCappsAtCapacity) {
    ReplayBuffer buf(5);
    for (int i = 0; i < 20; ++i)
        buf.add(make_sample(static_cast<float>(i), 0.0));
    EXPECT_EQ(buf.size(), 5);
}

TEST(ReplayBufferTest, AddBatch) {
    ReplayBuffer buf(100);
    TrainingSample samples[10];
    for (int i = 0; i < 10; ++i) samples[i] = make_sample(static_cast<float>(i), 0.5);
    buf.add_batch(samples, 10);
    EXPECT_EQ(buf.size(), 10);
}

// ---------------------------------------------------------------------------
// Ring-buffer overwrite behaviour
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, RingOverwrite) {
    // After writing 3 into a buffer of capacity 3, size == 3.
    // After writing 1 more, old sample is overwritten, size still 3.
    ReplayBuffer buf(3);
    buf.add(make_sample(1.0f, 0.1));
    buf.add(make_sample(2.0f, 0.2));
    buf.add(make_sample(3.0f, 0.3));
    EXPECT_EQ(buf.size(), 3);
    buf.add(make_sample(4.0f, 0.4));  // overwrites oldest
    EXPECT_EQ(buf.size(), 3);
}

// ---------------------------------------------------------------------------
// sample_batch
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, SampleBatch_EmptyReturnsZero) {
    ReplayBuffer buf(100);
    RNG rng(42);
    TrainingSample out[10];
    EXPECT_EQ(buf.sample_batch(out, 10, rng), 0);
}

TEST(ReplayBufferTest, SampleBatch_ReturnsCorrectCount) {
    ReplayBuffer buf(100);
    for (int i = 0; i < 50; ++i)
        buf.add(make_sample(static_cast<float>(i), static_cast<double>(i) / 100.0));

    RNG rng(1234);
    TrainingSample out[30];
    int n = buf.sample_batch(out, 30, rng);
    EXPECT_EQ(n, 30);
}

TEST(ReplayBufferTest, SampleBatch_SmallBuffer_Capped) {
    ReplayBuffer buf(100);
    buf.add(make_sample(1.0f, 0.5));
    buf.add(make_sample(2.0f, 0.6));

    RNG rng(99);
    TrainingSample out[10];
    int n = buf.sample_batch(out, 10, rng);
    EXPECT_EQ(n, 2);  // only 2 samples available
}

TEST(ReplayBufferTest, SampleBatch_TargetsInRange) {
    ReplayBuffer buf(100);
    for (int i = 0; i < 50; ++i)
        buf.add(make_sample(0.0f, static_cast<double>(i) / 50.0));

    RNG rng(7);
    TrainingSample out[50];
    buf.sample_batch(out, 50, rng);
    for (int i = 0; i < 50; ++i) {
        EXPECT_GE(out[i].target, 0.0);
        EXPECT_LE(out[i].target, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Save / load round-trip
// ---------------------------------------------------------------------------

TEST(ReplayBufferTest, SaveLoad_RoundTrip) {
    const std::string path = "/tmp/replay_buffer_test.bin";
    std::filesystem::remove(path);

    ReplayBuffer buf(50);
    for (int i = 0; i < 30; ++i)
        buf.add(make_sample(static_cast<float>(i), static_cast<double>(i) / 30.0));

    buf.save(path);

    ReplayBuffer buf2(50);
    bool loaded = buf2.load(path);
    ASSERT_TRUE(loaded);
    EXPECT_EQ(buf2.size(), 30);

    // Sample from both and verify targets span the same range.
    RNG rng(42);
    TrainingSample out1[30], out2[30];
    buf.sample_batch(out1, 30, rng);
    RNG rng2(42);
    buf2.sample_batch(out2, 30, rng2);
    for (int i = 0; i < 30; ++i) {
        EXPECT_NEAR(out1[i].target, out2[i].target, 1e-9);
        for (int j = 0; j < kTensorSize; ++j)
            EXPECT_FLOAT_EQ(out1[i].state[j], out2[i].state[j]);
    }

    std::filesystem::remove(path);
}

TEST(ReplayBufferTest, SaveLoad_FullBuffer) {
    const std::string path = "/tmp/replay_buffer_full_test.bin";
    std::filesystem::remove(path);

    ReplayBuffer buf(10);
    for (int i = 0; i < 25; ++i)  // wraps several times
        buf.add(make_sample(static_cast<float>(i), 0.5));

    EXPECT_EQ(buf.size(), 10);
    buf.save(path);

    ReplayBuffer buf2(10);
    ASSERT_TRUE(buf2.load(path));
    EXPECT_EQ(buf2.size(), 10);

    std::filesystem::remove(path);
}

TEST(ReplayBufferTest, Load_MissingFile_ReturnsFalse) {
    ReplayBuffer buf(100);
    EXPECT_FALSE(buf.load("/tmp/nonexistent_replay_buffer_xyz.bin"));
}
