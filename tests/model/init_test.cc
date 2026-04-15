#include <gtest/gtest.h>
#include "model/pro_yams_net.h"
#include "engine/tensor.h"

static torch::Device cpu_device() { return torch::Device(torch::kCPU); }

// ---------------------------------------------------------------------------
// Xavier init: weights not all zero
// ---------------------------------------------------------------------------
TEST(InitTest, XavierInit_WeightsNonZero) {
    ModelConfig cfg;
    ProYamsNet net(cfg);
    initialize_weights(net);
    for (const auto& p : net.parameters()) {
        if (p.dim() > 1) {
            EXPECT_GT(p.abs().max().item<float>(), 0.0f)
                << "Weight is all zeros after Xavier init";
        }
    }
}

// ---------------------------------------------------------------------------
// Biases zero after init
// ---------------------------------------------------------------------------
TEST(InitTest, XavierInit_BiasesZero) {
    ModelConfig cfg;
    ProYamsNet net(cfg);
    initialize_weights(net);
    // Bias tensors are 1D
    for (const auto& p : net.named_parameters()) {
        const std::string& name = p.key();
        if (name.find("bias") != std::string::npos) {
            EXPECT_NEAR(p.value().abs().max().item<float>(), 0.0f, 1e-7f)
                << "Bias '" << name << "' is not zero after init";
        }
    }
}

// ---------------------------------------------------------------------------
// Xavier variance: weights have reasonable scale
// ---------------------------------------------------------------------------
TEST(InitTest, XavierInit_ReasonableVariance) {
    ModelConfig cfg;
    ProYamsNet net(cfg);
    initialize_weights(net);
    // All weight matrices should have non-trivial std dev
    for (const auto& p : net.parameters()) {
        if (p.dim() > 1) {
            float std_dev = p.std().item<float>();
            EXPECT_GT(std_dev, 1e-4f) << "Weight std dev too small after Xavier init";
            EXPECT_LT(std_dev, 1.0f)  << "Weight std dev too large after Xavier init";
        }
    }
}

// ---------------------------------------------------------------------------
// Output near 0 after Xavier init (uninformed prior for tanh)
// ---------------------------------------------------------------------------
TEST(InitTest, XavierInit_OutputNearHalf) {
    // Run multiple seeds to verify outputs are bounded (tanh → near 0)
    ModelConfig cfg;
    cfg.hidden_layers = 3;
    cfg.hidden_width  = 256;

    for (int seed = 0; seed < 5; ++seed) {
        torch::manual_seed(seed);
        ProYamsNet net(cfg);
        initialize_weights(net);
        net.eval();
        torch::NoGradGuard no_grad;
        // Use uniform [0,1] input matching typical tensor features
        auto input  = torch::rand({500, kTensorSize});
        auto output = net.forward(input);
        float mean  = output.mean().item<float>();
        // After Xavier init with tanh, mean should be near 0 and well within [-1,1].
        // Bound is ±0.9: ResNet near-identity init collapses to ~1 hidden layer,
        // which has higher variance in mean than a 3-layer MLP.
        EXPECT_GT(mean, -0.9f)
            << "Mean output < -0.9 with seed " << seed << " (mean=" << mean << ")";
        EXPECT_LT(mean,  0.9f)
            << "Mean output > 0.9 with seed " << seed << " (mean=" << mean << ")";
    }
}

// ---------------------------------------------------------------------------
// Two different initializations produce different weights
// ---------------------------------------------------------------------------
TEST(InitTest, TwoInits_DifferentWeights) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;

    torch::manual_seed(1);
    ProYamsNet net_a(cfg);
    initialize_weights(net_a);

    torch::manual_seed(2);
    ProYamsNet net_b(cfg);
    initialize_weights(net_b);

    torch::NoGradGuard no_grad;
    auto input = torch::rand({16, kTensorSize});
    auto out_a = net_a.forward(input);
    auto out_b = net_b.forward(input);

    EXPECT_FALSE(out_a.allclose(out_b, /*rtol=*/1e-3, /*atol=*/1e-3))
        << "Two different inits produced identical outputs";
}

// ---------------------------------------------------------------------------
// get_device: returns valid device (CPU or CUDA)
// ---------------------------------------------------------------------------
TEST(InitTest, GetDevice_ReturnsValidDevice) {
    torch::Device dev = get_device();
    // Must be either CPU or CUDA
    bool valid = (dev.type() == torch::kCPU || dev.type() == torch::kCUDA);
    EXPECT_TRUE(valid) << "get_device() returned unexpected device type";
}
