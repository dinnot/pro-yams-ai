#include <gtest/gtest.h>
#include "model/trainer.h"
#include "model/pro_yams_net.h"
#include "engine/tensor.h"

#include <cmath>
#include <filesystem>
#include <vector>

static torch::Device cpu_device() { return torch::Device(torch::kCPU); }

// ---------------------------------------------------------------------------
// Basic construction
// ---------------------------------------------------------------------------
TEST(TrainerTest, Construction_NoCrash) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    ASSERT_NO_THROW(ModelTrainer trainer(cfg, cpu_device()));
}

// ---------------------------------------------------------------------------
// train_step returns a finite loss
// ---------------------------------------------------------------------------
TEST(TrainerTest, TrainStep_ReturnsFiniteLoss) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    ModelTrainer trainer(cfg, cpu_device());

    int batch_size = 32;
    std::vector<float>  states(batch_size * kTensorSize);
    std::vector<double> targets(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < kTensorSize; ++j)
            states[i * kTensorSize + j] = static_cast<float>(rand()) / RAND_MAX;
        targets[i] = (i % 2 == 0) ? 1.0 : 0.0;
    }

    double loss = trainer.train_step(states.data(), targets.data(), batch_size);
    EXPECT_TRUE(std::isfinite(loss)) << "Loss is not finite: " << loss;
    EXPECT_GE(loss, 0.0) << "Loss is negative";
}

// ---------------------------------------------------------------------------
// Loss decreases over 100 steps on a simple dataset
// ---------------------------------------------------------------------------
TEST(TrainerTest, LossDecreases_Over100Steps) {
    torch::manual_seed(42);
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    cfg.learning_rate = 0.001;
    ModelTrainer trainer(cfg, cpu_device());

    int batch_size = 64;
    std::vector<float>  states(batch_size * kTensorSize);
    std::vector<double> targets(batch_size);

    // Fixed input; alternating 0/1 targets provide a clear gradient signal
    torch::manual_seed(100);
    auto state_t = torch::rand({batch_size, kTensorSize});
    std::memcpy(states.data(), state_t.data_ptr<float>(),
                batch_size * kTensorSize * sizeof(float));
    for (int i = 0; i < batch_size; ++i)
        targets[i] = (i % 2 == 0) ? 1.0 : 0.0;

    double first_loss = trainer.train_step(states.data(), targets.data(), batch_size);
    double last_loss  = first_loss;
    for (int i = 1; i < 200; ++i)
        last_loss = trainer.train_step(states.data(), targets.data(), batch_size);

    EXPECT_LT(last_loss, first_loss)
        << "Loss did not decrease after 200 steps (first=" << first_loss
        << ", last=" << last_loss << ")";
}

// ---------------------------------------------------------------------------
// Overfitting: tiny dataset memorised after 1000 steps
// ---------------------------------------------------------------------------
TEST(TrainerTest, Overfitting_TinyDataset) {
    torch::manual_seed(77);
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 128;
    cfg.learning_rate = 0.001;
    ModelTrainer trainer(cfg, cpu_device());

    // 8 samples, fixed data — use torch random for reproducibility
    int n = 8;
    std::vector<float>  states(n * kTensorSize);
    std::vector<double> targets(n);
    {
        auto state_t = torch::rand({n, kTensorSize});
        std::memcpy(states.data(), state_t.data_ptr<float>(),
                    n * kTensorSize * sizeof(float));
    }
    for (int i = 0; i < n; ++i)
        targets[i] = (i % 2 == 0) ? 1.0 : 0.0;

    double loss = 1.0;
    for (int step = 0; step < 3000; ++step)
        loss = trainer.train_step(states.data(), targets.data(), n);

    EXPECT_LT(loss, 0.01)
        << "Model failed to memorise tiny dataset (loss=" << loss << ")";
}

// ---------------------------------------------------------------------------
// training_step_count increments correctly
// ---------------------------------------------------------------------------
TEST(TrainerTest, TrainingStepCount) {
    ModelConfig cfg;
    cfg.hidden_layers = 1;
    cfg.hidden_width  = 32;
    ModelTrainer trainer(cfg, cpu_device());

    EXPECT_EQ(trainer.training_step_count(), 0);

    std::vector<float>  s(kTensorSize, 0.5f);
    std::vector<double> t(1, 0.5);
    trainer.train_step(s.data(), t.data(), 1);
    EXPECT_EQ(trainer.training_step_count(), 1);
    trainer.train_step(s.data(), t.data(), 1);
    EXPECT_EQ(trainer.training_step_count(), 2);
}

// ---------------------------------------------------------------------------
// clone_for_inference produces a copy
// ---------------------------------------------------------------------------
TEST(TrainerTest, CloneForInference_NoCrash) {
    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    ModelTrainer trainer(cfg, cpu_device());

    auto clone = trainer.clone_for_inference(cpu_device());
    ASSERT_NE(clone, nullptr);

    // Clone should produce valid output
    torch::NoGradGuard no_grad;
    auto input  = torch::rand({4, kTensorSize});
    auto output = clone->forward(input);
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({4, 1}));
}

// ---------------------------------------------------------------------------
// Checkpoint round-trip: save + load restores model and metadata
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 2v2 migration Task 7.3: checkpoint refuses to load into a process running
// the wrong game variant.
// ---------------------------------------------------------------------------
TEST(TrainerTest, Checkpoint_VariantMismatch_Throws) {
    namespace fs = std::filesystem;

    ModelConfig cfg_1v1;
    cfg_1v1.game_variant = kGameVariant1v1;
    cfg_1v1.hidden_layers = 2;
    cfg_1v1.hidden_width  = 64;

    const std::string tmpdir = "/tmp/pro_yams_ckpt_variant_test";
    fs::create_directories(tmpdir);
    const std::string ckpt_path = tmpdir + "/variant_ckpt";

    // Save a 1v1 checkpoint.
    ModelTrainer trainer_1v1(cfg_1v1, cpu_device());
    trainer_1v1.save_checkpoint(ckpt_path, 0, 1.0, 0.0);

    // Confirm metadata parses correctly.
    auto loaded_cfg = ModelTrainer::config_from_checkpoint(ckpt_path);
    EXPECT_EQ(loaded_cfg.game_variant, kGameVariant1v1);

    // Construct a 2v2-configured trainer and try to load the 1v1 checkpoint.
    ModelConfig cfg_2v2 = cfg_1v1;
    cfg_2v2.game_variant = kGameVariant2v2;
    ModelTrainer trainer_2v2(cfg_2v2, cpu_device());
    int step; double temp, eps;
    EXPECT_THROW(trainer_2v2.load_checkpoint(ckpt_path, step, temp, eps),
                 std::runtime_error);

    // Round-trip into a matching trainer still works.
    ModelTrainer trainer_1v1_b(cfg_1v1, cpu_device());
    EXPECT_NO_THROW(trainer_1v1_b.load_checkpoint(ckpt_path, step, temp, eps));

    fs::remove_all(tmpdir);
}

TEST(TrainerTest, Checkpoint_RoundTrip) {
    namespace fs = std::filesystem;

    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    cfg.learning_rate = 0.001;

    // Create a temp dir for the checkpoint
    const std::string tmpdir = "/tmp/pro_yams_ckpt_test";
    fs::create_directories(tmpdir);
    const std::string ckpt_path = tmpdir + "/test_ckpt";

    // Train a bit
    ModelTrainer trainer_a(cfg, cpu_device());
    std::vector<float>  states(64 * kTensorSize);
    std::vector<double> targets(64, 0.5);
    for (auto& v : states) v = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < 10; ++i)
        trainer_a.train_step(states.data(), targets.data(), 64);

    // Record output before save (use clone_for_inference to avoid const issue)
    torch::NoGradGuard no_grad;
    auto input   = torch::rand({8, kTensorSize});
    auto net_a   = trainer_a.clone_for_inference(cpu_device());
    auto out_a   = net_a->forward(input).clone();

    // Save
    trainer_a.save_checkpoint(ckpt_path, 42, 1.5, 0.01);

    // Load into a fresh trainer
    ModelTrainer trainer_b(cfg, cpu_device());
    int step_loaded;
    double temp_loaded, eps_loaded;
    trainer_b.load_checkpoint(ckpt_path, step_loaded, temp_loaded, eps_loaded);

    EXPECT_EQ(step_loaded,  42);
    EXPECT_NEAR(temp_loaded, 1.5,  1e-9);
    EXPECT_NEAR(eps_loaded,  0.01, 1e-9);

    // Model outputs should match
    auto net_b = trainer_b.clone_for_inference(cpu_device());
    auto out_b = net_b->forward(input);
    EXPECT_TRUE(out_a.allclose(out_b, /*rtol=*/1e-5, /*atol=*/1e-5))
        << "Loaded model produces different outputs than saved model";

    // Cleanup
    fs::remove_all(tmpdir);
}

// ---------------------------------------------------------------------------
// Checkpoint: training continues smoothly after load (loss doesn't spike)
// ---------------------------------------------------------------------------
TEST(TrainerTest, Checkpoint_ContinuedTraining) {
    namespace fs = std::filesystem;
    const std::string tmpdir = "/tmp/pro_yams_cont_test";
    fs::create_directories(tmpdir);

    ModelConfig cfg;
    cfg.hidden_layers = 2;
    cfg.hidden_width  = 64;
    cfg.learning_rate = 0.001;

    ModelTrainer trainer(cfg, cpu_device());
    std::vector<float>  states(32 * kTensorSize);
    std::vector<double> targets(32, 0.5);
    for (auto& v : states) v = static_cast<float>(rand()) / RAND_MAX;

    // Train 20 steps
    double loss_before = 0;
    for (int i = 0; i < 20; ++i)
        loss_before = trainer.train_step(states.data(), targets.data(), 32);

    // Save and reload
    const std::string ckpt = tmpdir + "/cont";
    trainer.save_checkpoint(ckpt, 20, 1.0, 0.05);

    ModelTrainer trainer2(cfg, cpu_device());
    int s; double t, e;
    trainer2.load_checkpoint(ckpt, s, t, e);

    // First step after load should give a reasonable loss, not far from the
    // pre-save value. BCE minimum with target=0.5 is -log(0.5) ≈ 0.693.
    double loss_after = trainer2.train_step(states.data(), targets.data(), 32);
    EXPECT_LT(loss_after, 0.75)
        << "Loss spiked after checkpoint load (may indicate optimizer state not restored)";

    fs::remove_all(tmpdir);
}
