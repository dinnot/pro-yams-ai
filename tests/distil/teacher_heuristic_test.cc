#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "distil/teacher_heuristic.h"
#include "engine/game_flow.h"
#include "engine/game_traits.h"
#include "engine/rng.h"
#include "heuristic/heuristic_bot.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// Fixture: share precomputed tables across the suite.
// ---------------------------------------------------------------------------
class HeuristicTeacherTest : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) { init_precomputed_tables(tables); initialised = true; }
    }
    void SetUp() override {
        RNG rng(seed_++);
        init_game<Yams1v1>(gs, ctx, rng);
        solver_get_requests<Yams1v1>(gs, ctx, tables, buffers);
        ASSERT_GT(buffers.request_count, 0)
            << "init_game produced 0 requests — solver setup broken";
    }
    GameState     gs;
    GameContext   ctx;
    SolverBuffers buffers;
    static int seed_;
};
PrecomputedTables HeuristicTeacherTest::tables;
bool HeuristicTeacherTest::initialised = false;
int  HeuristicTeacherTest::seed_       = 7000;

// ---------------------------------------------------------------------------
// Interface contract
// ---------------------------------------------------------------------------

TEST_F(HeuristicTeacherTest, NeedsTensorInputFalse) {
    HeuristicTeacher<Yams1v1> t(HeuristicVersion::V2, tables,
                                /*use_margin=*/true);
    EXPECT_FALSE(t.needs_tensor_input());
}

TEST_F(HeuristicTeacherTest, VersionAccessor) {
    HeuristicTeacher<Yams1v1> t(HeuristicVersion::V3, tables, true);
    EXPECT_EQ(t.version(), HeuristicVersion::V3);
}

// ---------------------------------------------------------------------------
// Normalisation matches worker.cc's blending branches exactly.
// We compute the raw heuristic value with the in-tree evaluator and compare
// to the teacher's output to verify each squash branch.
// ---------------------------------------------------------------------------

TEST_F(HeuristicTeacherTest, V2_MarginMode_TanhSquash) {
    const double scale = 4000.0;
    HeuristicTeacher<Yams1v1> teacher(HeuristicVersion::V2, tables,
                                      /*use_margin=*/true, scale);

    std::vector<double> targets(buffers.request_count, 0.0);
    std::vector<double> solver_evs(buffers.request_count, 0.0);
    teacher.evaluate(gs.board, ctx, buffers.requests, buffers.request_count,
                     /*tensors=*/nullptr, /*tensor_stride=*/0, targets.data(), solver_evs.data());

    std::vector<double> raw(buffers.request_count, 0.0);
    heuristic_evaluate_v2<Yams1v1>(gs.board, ctx, buffers.requests,
                                   buffers.request_count, raw.data(), tables);

    for (int i = 0; i < buffers.request_count; ++i) {
        double expected = std::tanh(raw[i] / scale);
        EXPECT_NEAR(targets[i],    expected, 1e-12) << "i=" << i;
        EXPECT_NEAR(solver_evs[i], raw[i],   1e-12) << "i=" << i;
        EXPECT_GE(targets[i], -1.0);
        EXPECT_LE(targets[i],  1.0);
    }
}

TEST_F(HeuristicTeacherTest, V2_WinProbMode_RescaledTanh) {
    const double scale = 4000.0;
    HeuristicTeacher<Yams1v1> teacher(HeuristicVersion::V2, tables,
                                      /*use_margin=*/false, scale);

    std::vector<double> targets(buffers.request_count, 0.0);
    std::vector<double> solver_evs(buffers.request_count, 0.0);
    teacher.evaluate(gs.board, ctx, buffers.requests, buffers.request_count,
                     nullptr, /*tensor_stride=*/0, targets.data(), solver_evs.data());

    std::vector<double> raw(buffers.request_count, 0.0);
    heuristic_evaluate_v2<Yams1v1>(gs.board, ctx, buffers.requests,
                                   buffers.request_count, raw.data(), tables);

    for (int i = 0; i < buffers.request_count; ++i) {
        double expected = (std::tanh(raw[i] / scale) + 1.0) / 2.0;
        EXPECT_NEAR(targets[i],    expected, 1e-12) << "i=" << i;
        EXPECT_NEAR(solver_evs[i], raw[i],   1e-12) << "i=" << i;
        EXPECT_GE(targets[i], 0.0);
        EXPECT_LE(targets[i], 1.0);
    }
}

TEST_F(HeuristicTeacherTest, V1_WinProbMode_LegacyClampOver1800) {
    HeuristicTeacher<Yams1v1> teacher(HeuristicVersion::V1, tables,
                                      /*use_margin=*/false);

    std::vector<double> targets(buffers.request_count, 0.0);
    std::vector<double> solver_evs(buffers.request_count, 0.0);
    teacher.evaluate(gs.board, ctx, buffers.requests, buffers.request_count,
                     nullptr, /*tensor_stride=*/0, targets.data(), solver_evs.data());

    std::vector<double> raw(buffers.request_count, 0.0);
    heuristic_evaluate<Yams1v1>(gs.board, ctx, buffers.requests,
                                buffers.request_count, raw.data());

    for (int i = 0; i < buffers.request_count; ++i) {
        double expected = std::max(0.0, std::min(1.0, raw[i] / 1800.0));
        EXPECT_NEAR(targets[i],    expected, 1e-12) << "i=" << i;
        EXPECT_NEAR(solver_evs[i], raw[i],   1e-12) << "i=" << i;
        EXPECT_GE(targets[i], 0.0);
        EXPECT_LE(targets[i], 1.0);
    }
}

TEST_F(HeuristicTeacherTest, V1_MarginMode_TanhSquash) {
    const double scale = 4000.0;
    HeuristicTeacher<Yams1v1> teacher(HeuristicVersion::V1, tables,
                                      /*use_margin=*/true, scale);

    std::vector<double> targets(buffers.request_count, 0.0);
    std::vector<double> solver_evs(buffers.request_count, 0.0);
    teacher.evaluate(gs.board, ctx, buffers.requests, buffers.request_count,
                     nullptr, /*tensor_stride=*/0, targets.data(), solver_evs.data());

    std::vector<double> raw(buffers.request_count, 0.0);
    heuristic_evaluate<Yams1v1>(gs.board, ctx, buffers.requests,
                                buffers.request_count, raw.data());

    for (int i = 0; i < buffers.request_count; ++i) {
        double expected = std::tanh(raw[i] / scale);
        EXPECT_NEAR(targets[i],    expected, 1e-12) << "i=" << i;
        EXPECT_NEAR(solver_evs[i], raw[i],   1e-12) << "i=" << i;
    }
}

// V16/V17 are the "odds bot" research configs (output_win_odds=true). Their
// raw output is a probability; the normalisation path is distinct from the
// margin-style versions and worth covering directly.
TEST_F(HeuristicTeacherTest, V16_OddsBot_MarginMode_MapsToMinusOneToOne) {
    HeuristicTeacher<Yams1v1> teacher(HeuristicVersion::V16, tables,
                                      /*use_margin=*/true);

    std::vector<double> targets(buffers.request_count, 0.0);
    std::vector<double> solver_evs(buffers.request_count, 0.0);
    teacher.evaluate(gs.board, ctx, buffers.requests, buffers.request_count,
                     nullptr, /*tensor_stride=*/0, targets.data(), solver_evs.data());

    std::vector<double> raw(buffers.request_count, 0.0);
    const ResearchConfig& cfg = get_research_config_for(HeuristicVersion::V16);
    ASSERT_TRUE(cfg.output_win_odds);
    heuristic_evaluate_research<Yams1v1>(gs.board, ctx, buffers.requests,
                                         buffers.request_count, raw.data(),
                                         tables, cfg);

    for (int i = 0; i < buffers.request_count; ++i) {
        double p = std::max(0.0, std::min(1.0, raw[i]));
        double expected = p * 2.0 - 1.0;
        EXPECT_NEAR(targets[i],    expected, 1e-12) << "i=" << i;
        EXPECT_NEAR(solver_evs[i], raw[i],   1e-12) << "i=" << i;
        EXPECT_GE(targets[i], -1.0);
        EXPECT_LE(targets[i],  1.0);
    }
}

TEST_F(HeuristicTeacherTest, V16_OddsBot_WinProbMode_ClampedProb) {
    HeuristicTeacher<Yams1v1> teacher(HeuristicVersion::V16, tables,
                                      /*use_margin=*/false);

    std::vector<double> targets(buffers.request_count, 0.0);
    std::vector<double> solver_evs(buffers.request_count, 0.0);
    teacher.evaluate(gs.board, ctx, buffers.requests, buffers.request_count,
                     nullptr, /*tensor_stride=*/0, targets.data(), solver_evs.data());

    std::vector<double> raw(buffers.request_count, 0.0);
    const ResearchConfig& cfg = get_research_config_for(HeuristicVersion::V16);
    heuristic_evaluate_research<Yams1v1>(gs.board, ctx, buffers.requests,
                                         buffers.request_count, raw.data(),
                                         tables, cfg);

    for (int i = 0; i < buffers.request_count; ++i) {
        double expected = std::max(0.0, std::min(1.0, raw[i]));
        EXPECT_NEAR(targets[i],    expected, 1e-12) << "i=" << i;
        EXPECT_NEAR(solver_evs[i], raw[i],   1e-12) << "i=" << i;
        EXPECT_GE(targets[i], 0.0);
        EXPECT_LE(targets[i], 1.0);
    }
}

// ---------------------------------------------------------------------------
// 2v2 specialisation compiles and runs.
// ---------------------------------------------------------------------------

TEST(HeuristicTeacher2v2Test, V2_MarginMode_RunsAndProducesInRangeOutputs) {
    static PrecomputedTables tables;
    static bool init = false;
    if (!init) { init_precomputed_tables(tables); init = true; }

    GameState2v2     gs;
    GameContext2v2   ctx;
    SolverBuffers    buffers;
    RNG              rng(9999);
    init_game<Yams2v2>(gs, ctx, rng);
    solver_get_requests<Yams2v2>(gs, ctx, tables, buffers);
    ASSERT_GT(buffers.request_count, 0);

    HeuristicTeacher<Yams2v2> teacher(HeuristicVersion::V2, tables,
                                      /*use_margin=*/true);
    std::vector<double> targets(buffers.request_count, 0.0);
    std::vector<double> solver_evs(buffers.request_count, 0.0);
    teacher.evaluate(gs.board, ctx, buffers.requests, buffers.request_count,
                     nullptr, /*tensor_stride=*/0, targets.data(), solver_evs.data());

    for (int i = 0; i < buffers.request_count; ++i) {
        EXPECT_GE(targets[i], -1.0) << "i=" << i;
        EXPECT_LE(targets[i],  1.0) << "i=" << i;
    }
}
