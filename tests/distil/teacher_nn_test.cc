#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "distil/distil_config.h"
#include "distil/distil_loop.h"
#include "distil/teacher_nn.h"
#include "engine/game_traits.h"
#include "engine/rng.h"
#include "model/inference.h"
#include "model/model_config.h"
#include "model/pro_yams_net.h"
#include "model/trainer.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"
#include "engine/game_flow.h"
#include "engine/tensor.h"
#include "test_tables.h"

// Tables singleton — shared across distil test files.
static PrecomputedTables* g_tables = nullptr;
static void ensure_tables() { g_tables = &distil_test::tables(); }

// Save a tiny ProYamsNet to disk and return the checkpoint stem (no
// extension). The .model file is what NNTeacher reads.
static std::string write_tiny_teacher_ckpt(const std::string& stem,
                                           int game_variant = kGameVariant1v1,
                                           const std::string& act = "tanh",
                                           int hidden_layers = 1,
                                           int hidden_width  = 16,
                                           int tensor_version = kTensorVersionLatest) {
    std::filesystem::remove(stem + ".model");
    std::filesystem::remove(stem + ".optimizer");

    ModelConfig cfg;
    cfg.game_variant      = game_variant;
    // input_size must agree with tensor_version, else NNTeacher rejects the
    // checkpoint as inconsistent. Deriving keeps an older-version teacher
    // (e.g. V1, 986) self-consistent.
    cfg.tensor_version    = tensor_version;
    cfg.input_size        = (game_variant == kGameVariant1v1)
                            ? tensor_size_for_version<Yams1v1>(tensor_version)
                            : tensor_size_for_version<Yams2v2>(tensor_version);
    cfg.hidden_layers     = hidden_layers;
    cfg.hidden_width      = hidden_width;
    cfg.output_activation = act;
    cfg.loss_function     = "mse";
    cfg.architecture      = "mlp";

    ModelTrainer trainer(cfg, torch::Device(torch::kCPU));
    trainer.save_checkpoint(stem, /*step=*/0, /*temp=*/0.0, /*eps=*/0.0);
    return stem;
}

// ---------------------------------------------------------------------------
// Construction validation: rejects shape/variant/activation mismatches.
// ---------------------------------------------------------------------------

TEST(NNTeacherTest, RejectsMissingCheckpoint) {
    ensure_tables();
    EXPECT_THROW({
        NNTeacher<Yams1v1> t("/tmp/no_such_distil_teacher",
                             torch::kCPU, /*use_margin=*/true);
    }, std::exception);
}

TEST(NNTeacherTest, RejectsVariantMismatch) {
    ensure_tables();
    const std::string stem = "/tmp/distil_nn_test_2v2_ckpt";
    write_tiny_teacher_ckpt(stem, kGameVariant2v2, "tanh");

    // Build a 2v2 checkpoint, try to load it into a 1v1 NNTeacher.
    EXPECT_THROW({
        NNTeacher<Yams1v1> t(stem, torch::kCPU, /*use_margin=*/true);
    }, std::runtime_error);

    std::filesystem::remove(stem + ".model");
    std::filesystem::remove(stem + ".optimizer");
}

TEST(NNTeacherTest, RejectsActivationMismatch) {
    ensure_tables();
    const std::string stem = "/tmp/distil_nn_test_sigmoid_ckpt";
    write_tiny_teacher_ckpt(stem, kGameVariant1v1, "sigmoid");

    // sigmoid teacher with use_duel_margin_maximization=true (expects tanh).
    EXPECT_THROW({
        NNTeacher<Yams1v1> t(stem, torch::kCPU, /*use_margin=*/true);
    }, std::runtime_error);

    std::filesystem::remove(stem + ".model");
    std::filesystem::remove(stem + ".optimizer");
}

// ---------------------------------------------------------------------------
// evaluate(): matches a direct InferenceEngine call on the same model.
// ---------------------------------------------------------------------------

TEST(NNTeacherTest, EvaluateMatchesDirectInference) {
    ensure_tables();
    const std::string stem = "/tmp/distil_nn_test_match_ckpt";
    write_tiny_teacher_ckpt(stem);

    NNTeacher<Yams1v1> teacher(stem, torch::kCPU, /*use_margin=*/true);

    // Generate a real afterstate batch from an initial board.
    GameState gs; GameContext ctx; SolverBuffers buffers{};
    RNG rng(123);
    init_game<Yams1v1>(gs, ctx, rng);
    solver_get_requests<Yams1v1>(gs, ctx, *g_tables, buffers);
    ASSERT_GT(buffers.request_count, 0);

    const int n  = buffers.request_count;
    const int kT = Yams1v1::kTensorSize;
    std::vector<float> tensors(static_cast<size_t>(n) * kT);
    generate_tensor_batch<Yams1v1>(
        gs.board, ctx, gs.board.current_player,
        buffers.requests, n, *g_tables, tensors.data());

    // Teacher path.
    std::vector<double> targets_teacher(n, 0.0);
    std::vector<double> solver_evs_teacher(n, 0.0);
    teacher.evaluate(gs.board, ctx, buffers.requests, n,
                     tensors.data(), /*tensor_stride=*/kT,
                     targets_teacher.data(), solver_evs_teacher.data());

    // Direct path through a fresh engine wrapping the teacher's model.
    InferenceEngine direct(
        std::shared_ptr<ProYamsNet>(&teacher.model(), [](ProYamsNet*){/*no-op*/}),
        torch::kCPU);
    std::vector<double> evs_direct(n, 0.0);
    direct.batch_inference(tensors.data(), n, evs_direct.data());

    for (int i = 0; i < n; ++i) {
        // For an NN teacher, the prediction IS the action value: targets and
        // solver_evs are the same numbers, both equal to direct inference.
        EXPECT_NEAR(targets_teacher[i],    evs_direct[i], 1e-9) << "i=" << i;
        EXPECT_NEAR(solver_evs_teacher[i], evs_direct[i], 1e-9) << "i=" << i;
    }

    std::filesystem::remove(stem + ".model");
    std::filesystem::remove(stem + ".optimizer");
}

// ---------------------------------------------------------------------------
// Cross-version distillation: a V1 teacher (input_size 986) supervising a
// student on the latest layout reads the leading 986 columns of each (wider)
// student row — the append-only prefix. This is the path that makes "distil an
// old model into a new-tensor student" work without a second tensor generation.
// ---------------------------------------------------------------------------
TEST(NNTeacherTest, EvaluateReadsV1PrefixOfLatestTensor) {
    ensure_tables();
    // Skip trivially if there is no version gap to exercise (V1 == latest).
    if (Yams1v1::kTensorSizeV1 == Yams1v1::kTensorSize) GTEST_SKIP();

    const std::string stem = "/tmp/distil_nn_test_v1_prefix_ckpt";
    write_tiny_teacher_ckpt(stem, kGameVariant1v1, "tanh",
                            /*hidden_layers=*/1, /*hidden_width=*/16,
                            /*tensor_version=*/kTensorVersionV1);

    NNTeacher<Yams1v1> teacher(stem, torch::kCPU, /*use_margin=*/true);
    ASSERT_EQ(teacher.tensor_version(), kTensorVersionV1);
    ASSERT_EQ(teacher.input_size(), Yams1v1::kTensorSizeV1);

    GameState gs; GameContext ctx; SolverBuffers buffers{};
    RNG rng(321);
    init_game<Yams1v1>(gs, ctx, rng);
    solver_get_requests<Yams1v1>(gs, ctx, *g_tables, buffers);
    ASSERT_GT(buffers.request_count, 0);

    const int n  = buffers.request_count;
    const int kT = Yams1v1::kTensorSize;          // latest (student) width
    const int v1 = Yams1v1::kTensorSizeV1;        // teacher width
    std::vector<float> student(static_cast<size_t>(n) * kT);
    generate_tensor_batch<Yams1v1>(
        gs.board, ctx, gs.board.current_player,
        buffers.requests, n, *g_tables, student.data());

    // Teacher reads the prefix internally (stride kT, width v1).
    std::vector<double> evs_teacher(n, 0.0);
    teacher.evaluate(gs.board, ctx, buffers.requests, n,
                     student.data(), /*tensor_stride=*/kT,
                     evs_teacher.data(), nullptr);

    // Reference: pack the leading v1 columns of each row ourselves, infer.
    std::vector<float> packed(static_cast<size_t>(n) * v1);
    for (int i = 0; i < n; ++i) {
        std::memcpy(packed.data() + static_cast<size_t>(i) * v1,
                    student.data() + static_cast<size_t>(i) * kT,
                    static_cast<size_t>(v1) * sizeof(float));
    }
    InferenceEngine direct(
        std::shared_ptr<ProYamsNet>(&teacher.model(), [](ProYamsNet*){}),
        torch::kCPU);
    std::vector<double> evs_direct(n, 0.0);
    direct.batch_inference(packed.data(), n, evs_direct.data());

    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(evs_teacher[i], evs_direct[i], 1e-9) << "i=" << i;

    std::filesystem::remove(stem + ".model");
    std::filesystem::remove(stem + ".optimizer");
}

TEST(NNTeacherTest, NeedsTensorInputTrue) {
    ensure_tables();
    const std::string stem = "/tmp/distil_nn_test_needs_input_ckpt";
    write_tiny_teacher_ckpt(stem);
    NNTeacher<Yams1v1> teacher(stem, torch::kCPU, /*use_margin=*/true);
    EXPECT_TRUE(teacher.needs_tensor_input());
    std::filesystem::remove(stem + ".model");
    std::filesystem::remove(stem + ".optimizer");
}

// ---------------------------------------------------------------------------
// End-to-end DistilLoopT with an NN teacher runs a few steps and writes
// the final report.
// ---------------------------------------------------------------------------

TEST(NNTeacherTest, DistilLoopRunsWithNNTeacher) {
    ensure_tables();
    const std::string stem = "/tmp/distil_nn_loop_teacher";
    write_tiny_teacher_ckpt(stem);

    const std::string ckpt_dir = "/tmp/distil_nn_loop_test_ckpt";
    const std::string log_dir  = "/tmp/distil_nn_loop_test_logs";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove_all(log_dir);

    DistilConfig cfg;
    cfg.teacher_kind            = TeacherKind::kNN;
    cfg.teacher_checkpoint_path = stem;
    cfg.reference_heuristic_version = 2;

    cfg.self_play.num_workers = 2;
    cfg.self_play.num_games   = 4;
    cfg.self_play.max_inference_batch = 1024;
    cfg.self_play.min_games_per_batch = 1;
    cfg.self_play.batch_timeout_ms    = 10;

    cfg.train_batch_size        = 32;
    cfg.replay_buffer_capacity  = 1024;
    cfg.min_samples_to_start    = 64;
    cfg.samples_per_train       = 1.0;

    cfg.checkpoint_interval = 1'000'000;
    cfg.max_steps           = 10;
    cfg.min_steps           = 0;
    cfg.eval_interval       = 1'000'000;
    cfg.eval_games          = 4;
    cfg.final_eval_games    = 8;
    cfg.convergence_win_rate_delta = -1.0;
    cfg.convergence_patience       = 3;

    cfg.use_duel_margin_maximization   = true;
    cfg.duel_margin_maximization_scale = 4000.0;

    cfg.checkpoint_dir = ckpt_dir;
    cfg.log_dir        = log_dir;
    cfg.log_path       = log_dir + "/distil_log.csv";

    cfg.student_model.hidden_layers     = 1;
    cfg.student_model.hidden_width      = 16;
    cfg.student_model.output_activation = "tanh";
    cfg.student_model.architecture      = "mlp";
    // Fresh student on the latest tensor layout (the loader derives these; this
    // test builds DistilConfig directly so it must set them explicitly).
    cfg.student_model.tensor_version    = kTensorVersionLatest;
    cfg.student_model.input_size        = Yams1v1::kTensorSize;

    DistilLoop loop(cfg, *g_tables, torch::kCPU, torch::kCPU);
    loop.run();

    EXPECT_EQ(loop.stop_reason(), DistilLoop::StopReason::kMaxSteps);
    EXPECT_GT(loop.metrics().total_samples_trained, 0);
    EXPECT_TRUE(std::isfinite(loop.metrics().loss));
    // final_report ran and wrote the headline numbers.
    EXPECT_TRUE(std::isfinite(loop.final_teacher_win_rate()));
    EXPECT_TRUE(std::isfinite(loop.final_student_win_rate()));
    ASSERT_TRUE(std::filesystem::exists(log_dir + "/distil_final.csv"));

    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::remove_all(log_dir);
    std::filesystem::remove(stem + ".model");
    std::filesystem::remove(stem + ".optimizer");
}
