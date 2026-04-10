#include <gtest/gtest.h>
#include "model/inference.h"
#include "model/pro_yams_net.h"
#include "engine/tensor.h"  // kTensorSize

#include <cmath>
#include <vector>

static torch::Device cpu_device() { return torch::Device(torch::kCPU); }

// ---------------------------------------------------------------------------
// batch_inference correctness: matches direct libtorch forward pass
// ---------------------------------------------------------------------------
TEST(InferenceTest, BatchInference_MatchesDirectForward) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    auto net = std::make_shared<ProYamsNet>(cfg);
    initialize_weights(*net);
    net->eval();

    auto device = cpu_device();
    InferenceEngine engine(net, device);

    int batch_size = 16;
    std::vector<float> input(batch_size * kTensorSize);
    for (auto& v : input) v = static_cast<float>(rand()) / RAND_MAX;

    // Via InferenceEngine
    std::vector<double> ie_output(batch_size);
    engine.batch_inference(input.data(), batch_size, ie_output.data());

    // Via direct libtorch
    torch::NoGradGuard no_grad;
    auto t_input  = torch::from_blob(input.data(), {batch_size, kTensorSize}, torch::kFloat32);
    auto t_output = net->forward(t_input).contiguous();
    const float* direct_ptr = t_output.data_ptr<float>();

    for (int i = 0; i < batch_size; ++i) {
        EXPECT_NEAR(ie_output[i], static_cast<double>(direct_ptr[i]), 1e-5)
            << "Mismatch at sample " << i;
    }
}

// ---------------------------------------------------------------------------
// Batch sizes: 1, 16, 64, 256
// ---------------------------------------------------------------------------
TEST(InferenceTest, VariousBatchSizes) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    auto net = std::make_shared<ProYamsNet>(cfg);
    initialize_weights(*net);

    auto device = cpu_device();
    InferenceEngine engine(net, device);

    for (int bs : {1, 16, 64, 256}) {
        std::vector<float>  input(bs * kTensorSize, 0.5f);
        std::vector<double> output(bs, 0.0);

        ASSERT_NO_THROW(engine.batch_inference(input.data(), bs, output.data()))
            << "batch_inference threw for batch_size=" << bs;

        for (int i = 0; i < bs; ++i) {
            EXPECT_GT(output[i], 0.0) << "Output below 0 for batch_size=" << bs;
            EXPECT_LT(output[i], 1.0) << "Output above 1 for batch_size=" << bs;
        }
    }
}

// ---------------------------------------------------------------------------
// All outputs in (0, 1) — sigmoid guarantee
// ---------------------------------------------------------------------------
TEST(InferenceTest, Outputs_InUnitRange) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    auto net = std::make_shared<ProYamsNet>(cfg);
    initialize_weights(*net);
    InferenceEngine engine(net, cpu_device());

    int bs = 100;
    std::vector<float>  input(bs * kTensorSize);
    for (auto& v : input) v = static_cast<float>(rand()) / RAND_MAX;
    std::vector<double> output(bs);
    engine.batch_inference(input.data(), bs, output.data());

    for (int i = 0; i < bs; ++i) {
        EXPECT_GT(output[i], 0.0) << "Output[" << i << "] not > 0";
        EXPECT_LT(output[i], 1.0) << "Output[" << i << "] not < 1";
    }
}

// ---------------------------------------------------------------------------
// swap_model: results change after swap
// ---------------------------------------------------------------------------
TEST(InferenceTest, SwapModel_ResultsChange) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;

    // Two models with different weights
    auto net_a = std::make_shared<ProYamsNet>(cfg);
    initialize_weights(*net_a);
    auto net_b = std::make_shared<ProYamsNet>(cfg);
    initialize_weights(*net_b);  // Different random weights

    InferenceEngine engine(net_a, cpu_device());

    std::vector<float>  input(kTensorSize);
    for (auto& v : input) v = static_cast<float>(rand()) / RAND_MAX;

    std::vector<double> output_a(1), output_b(1);
    engine.batch_inference(input.data(), 1, output_a.data());

    engine.swap_model(net_b);
    engine.batch_inference(input.data(), 1, output_b.data());

    // With high probability, two different random inits give different outputs
    // (This could theoretically fail but is extremely unlikely)
    EXPECT_NE(output_a[0], output_b[0])
        << "Outputs identical after model swap — swap may have failed";
}

// ---------------------------------------------------------------------------
// clone_model: clone produces identical outputs
// ---------------------------------------------------------------------------
TEST(InferenceTest, CloneModel_IdenticalOutputs) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    auto original = std::make_shared<ProYamsNet>(cfg);
    initialize_weights(*original);
    original->eval();

    auto cloned = clone_model(*original, cpu_device());
    cloned->eval();

    torch::NoGradGuard no_grad;
    auto input = torch::rand({32, kTensorSize});
    auto out_orig  = original->forward(input);
    auto out_clone = cloned->forward(input);

    EXPECT_TRUE(out_orig.allclose(out_clone, /*rtol=*/1e-5, /*atol=*/1e-5))
        << "Cloned model produces different outputs";
}

// ---------------------------------------------------------------------------
// clone_model: modifying clone doesn't affect original
// ---------------------------------------------------------------------------
TEST(InferenceTest, CloneModel_Independence) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    auto original = std::make_shared<ProYamsNet>(cfg);
    initialize_weights(*original);
    original->eval();

    auto cloned = clone_model(*original, cpu_device());

    // Zero out all cloned parameters
    torch::NoGradGuard no_grad;
    for (auto& p : cloned->parameters()) p.data().zero_();

    // Original should be unaffected
    auto input     = torch::rand({4, kTensorSize});
    auto out_orig  = original->forward(input);
    auto out_clone = cloned->forward(input);

    // They should NOT be the same after zeroing clone's weights
    EXPECT_FALSE(out_orig.allclose(out_clone, /*rtol=*/1e-3, /*atol=*/1e-3))
        << "Cloned model modification leaked into original";
}
