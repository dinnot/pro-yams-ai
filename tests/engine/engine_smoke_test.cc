#include <gtest/gtest.h>

// Proves gtest is working and can link against libengine
TEST(EngineSmoke, Compiles) {
    EXPECT_EQ(1 + 1, 2);
}
