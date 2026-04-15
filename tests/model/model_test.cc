#include <gtest/gtest.h>
#include "model/pro_yams_net.h"
#include "model/model_config.h"
#include "engine/tensor.h"  // kTensorSize

// Use CPU for tests (no GPU required)
static torch::Device test_device() { return torch::Device(torch::kCPU); }

// ---------------------------------------------------------------------------
// Construction and parameter count
// ---------------------------------------------------------------------------
TEST(ModelTest, DefaultConfig_Construction) {
    ModelConfig cfg;
    ASSERT_NO_THROW(ProYamsNet net(cfg));
}

TEST(ModelTest, ParameterCount_DefaultConfig) {
    // Default: 3 hidden layers of width 256, architecture=resnet
    // Projection: 809*256 + 256                              = 207,360
    // ResBlock 0: lin1(256*256+256) + norm1(256*2)
    //           + lin2(256*256+256) + norm2(256*2)           = 132,608
    // ResBlock 1: same                                       = 132,608
    // Output:  256*1 + 1                                     =     257
    // Total:                                                 = 472,833
    ModelConfig cfg;
    ProYamsNet net(cfg);
    int64_t param_count = 0;
    for (const auto& p : net.parameters()) {
        param_count += p.numel();
    }
    EXPECT_EQ(param_count, 472833LL);
}

TEST(ModelTest, CustomConfig_ParameterCount) {
    // 2 hidden layers of width 128, architecture=resnet
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 128;
    // Projection: 809*128 + 128                             = 103,680
    // ResBlock 0: lin1(128*128+128) + norm1(128*2)
    //           + lin2(128*128+128) + norm2(128*2)          =  33,536
    // Output:  128*1 + 1                                    =     129
    // Total:                                                = 137,345
    ProYamsNet net(cfg);
    int64_t param_count = 0;
    for (const auto& p : net.parameters()) param_count += p.numel();
    EXPECT_EQ(param_count, 137345LL);
}

// ---------------------------------------------------------------------------
// Forward pass output shape
// ---------------------------------------------------------------------------
TEST(ModelTest, ForwardShape_BatchSize1) {
    ModelConfig cfg;
    ProYamsNet net(cfg);
    net.eval();
    torch::NoGradGuard no_grad;
    auto input = torch::rand({1, kTensorSize});
    auto output = net.forward(input);
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({1, 1}));
}

TEST(ModelTest, ForwardShape_BatchSize64) {
    ModelConfig cfg;
    ProYamsNet net(cfg);
    net.eval();
    torch::NoGradGuard no_grad;
    auto output = net.forward(torch::rand({64, kTensorSize}));
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({64, 1}));
}

TEST(ModelTest, ForwardShape_BatchSize512) {
    ModelConfig cfg;
    ProYamsNet net(cfg);
    net.eval();
    torch::NoGradGuard no_grad;
    auto output = net.forward(torch::rand({512, kTensorSize}));
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({512, 1}));
}

// ---------------------------------------------------------------------------
// Output range: tanh guarantees (-1, 1); sigmoid guarantees (0, 1)
// ---------------------------------------------------------------------------
TEST(ModelTest, OutputRange_Tanh) {
    ModelConfig cfg;
    cfg.output_activation = "tanh";
    ProYamsNet net(cfg);
    net.eval();
    torch::NoGradGuard no_grad;
    auto input  = torch::rand({256, kTensorSize});
    auto output = net.forward(input);
    auto min_val = output.min().item<float>();
    auto max_val = output.max().item<float>();
    EXPECT_GT(min_val, -1.0f);
    EXPECT_LT(max_val,  1.0f);
}

TEST(ModelTest, OutputRange_Sigmoid) {
    ModelConfig cfg;
    cfg.output_activation = "sigmoid";
    ProYamsNet net(cfg);
    net.eval();
    torch::NoGradGuard no_grad;
    auto input  = torch::rand({256, kTensorSize});
    auto output = net.forward(input);
    auto min_val = output.min().item<float>();
    auto max_val = output.max().item<float>();
    EXPECT_GT(min_val, 0.0f);
    EXPECT_LT(max_val, 1.0f);
}

// ---------------------------------------------------------------------------
// Determinism: same model, same input → same output
// ---------------------------------------------------------------------------
TEST(ModelTest, Determinism_SameInputSameOutput) {
    ModelConfig cfg;
    ProYamsNet net(cfg);
    net.eval();
    torch::NoGradGuard no_grad;
    auto input = torch::rand({32, kTensorSize});
    auto out1  = net.forward(input.clone());
    auto out2  = net.forward(input.clone());
    EXPECT_TRUE(out1.allclose(out2, /*rtol=*/1e-6, /*atol=*/1e-6));
}

// ---------------------------------------------------------------------------
// config() accessor round-trip
// ---------------------------------------------------------------------------
TEST(ModelTest, ConfigAccessor) {
    ModelConfig cfg;
    cfg.hidden_layers = 4;
    cfg.hidden_width  = 512;
    ProYamsNet net(cfg);
    EXPECT_EQ(net.config().hidden_layers, 4);
    EXPECT_EQ(net.config().hidden_width,  512);
    EXPECT_EQ(net.config().input_size,    809);
}

// ---------------------------------------------------------------------------
// Weight initialization: outputs near 0 for tanh (uninformed prior)
// ---------------------------------------------------------------------------
TEST(ModelTest, XavierInit_OutputsNearZero) {
    ModelConfig cfg;  // default output_activation = "tanh"
    ProYamsNet net(cfg);
    initialize_weights(net);
    net.eval();
    torch::NoGradGuard no_grad;
    // Use uniform [0,1] input (typical tensor values)
    auto input = torch::rand({1000, kTensorSize});
    auto output = net.forward(input);
    float mean = output.mean().item<float>();
    // After Xavier init with tanh, mean should be near 0.
    // Bounds are [-0.8, 0.8] to avoid spurious failures from random init variance.
    EXPECT_GT(mean, -0.8f) << "Mean output too low — init may be problematic";
    EXPECT_LT(mean,  0.8f) << "Mean output too high — init may be problematic";
}

// ---------------------------------------------------------------------------
// Weights not all zero after initialization
// ---------------------------------------------------------------------------
TEST(ModelTest, XavierInit_WeightsNonZero) {
    ModelConfig cfg;
    ProYamsNet net(cfg);
    initialize_weights(net);
    for (const auto& p : net.parameters()) {
        if (p.dim() > 1) {
            // Weight matrix: should NOT be all zeros
            EXPECT_GT(p.abs().max().item<float>(), 0.0f)
                << "Weight parameter is all zeros after Xavier init";
        }
    }
}
