#include <gtest/gtest.h>
#include "engine/tensor.h"
#include "engine/board_init.h"
#include "engine/game_flow.h"
#include "engine/placement.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

#include <cmath>

// Shared fixture
class TensorBatchTest : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) { init_precomputed_tables(tables); initialised = true; }
    }
    void SetUp() override {
        RNG rng(seed_++);
        init_game(gs, ctx, rng);
    }
    GameState gs;
    GameContext ctx;
    static int seed_;
};
PrecomputedTables TensorBatchTest::tables;
bool TensorBatchTest::initialised = false;
int TensorBatchTest::seed_ = 7000;

// ---------------------------------------------------------------------------
// Helper: generate tensor for a manually cloned+modified board
// ---------------------------------------------------------------------------
static void tensor_for_request(const BoardState& base_board,
                                const GameContext& base_ctx,
                                int player,
                                const AfterstateRequest& req,
                                const PrecomputedTables& tables,
                                float* out) {
    // Full clone + apply_placement (mirrors generate_tensor_batch logic)
    BoardState board_clone = base_board;
    GameContext ctx_clone  = base_ctx;
    apply_placement(player,
                    req.placement.column, req.placement.row, req.score,
                    board_clone, ctx_clone);
    generate_tensor(board_clone, ctx_clone, player, tables, out);
}

// ---------------------------------------------------------------------------
// Batch vs individual consistency
// ---------------------------------------------------------------------------
TEST_F(TensorBatchTest, BatchConsistency_MatchesSingleTensor) {
    // Gather requests from current game state
    SolverBuffers buffers;
    solver_get_requests(gs, ctx, tables, buffers);
    ASSERT_GT(buffers.request_count, 0);

    // Set rolls_left=0 to force Turbo inclusion and avoid early game oddities
    gs.rolls_left = 0;
    solver_get_requests(gs, ctx, tables, buffers);
    ASSERT_GT(buffers.request_count, 0);

    int p = gs.board.current_player;
    int n = buffers.request_count;

    // Batch generate
    std::vector<float> batch_out(n * kTensorSize, 0.0f);
    generate_tensor_batch(gs.board, ctx, p, buffers.requests, n, tables, batch_out.data());

    // Compare each tensor in batch against individually generated
    for (int i = 0; i < n; ++i) {
        float single[kTensorSize];
        tensor_for_request(gs.board, ctx, p, buffers.requests[i], tables, single);

        const float* batch_tensor = batch_out.data() + i * kTensorSize;
        for (int f = 0; f < kTensorSize; ++f) {
            EXPECT_NEAR(batch_tensor[f], single[f], 1e-5f)
                << "Mismatch at request " << i << " feature " << f;
            if (std::abs(batch_tensor[f] - single[f]) > 1e-5f)
                goto next_request;  // report first mismatch per request only
        }
        next_request:;
    }
}

// ---------------------------------------------------------------------------
// Multiple afterstates: each tensor corresponds to a distinct placement
// ---------------------------------------------------------------------------
TEST_F(TensorBatchTest, MultipleAfterstates_CorrectPlacement) {
    gs.rolls_left = 0;
    SolverBuffers buffers;
    solver_get_requests(gs, ctx, tables, buffers);
    ASSERT_GE(buffers.request_count, 2);

    int p = gs.board.current_player;
    int n = std::min(buffers.request_count, 10);

    std::vector<float> batch_out(n * kTensorSize, 0.0f);
    generate_tensor_batch(gs.board, ctx, p, buffers.requests, n, tables, batch_out.data());

    // For each tensor, verify the cell corresponding to the placement is filled
    for (int i = 0; i < n; ++i) {
        const float* t = batch_out.data() + i * kTensorSize;
        int col = buffers.requests[i].placement.column;
        int row = buffers.requests[i].placement.row;
        // Group A: player perspective = pi=0 (me first), so offset within [0, 234)
        int cell_idx  = col * kNumRows + row;
        int feat_idx  = cell_idx * 3;  // is_filled feature
        EXPECT_NEAR(t[feat_idx], 1.0f, 1e-6f)
            << "Request " << i << " (col=" << col << ",row=" << row
            << "): is_filled should be 1 in batch tensor";
    }
}

// ---------------------------------------------------------------------------
// Batch with single request: same as single generate_tensor result
// ---------------------------------------------------------------------------
TEST_F(TensorBatchTest, BatchSingle_MatchesSingleTensor) {
    gs.rolls_left = 0;
    SolverBuffers buffers;
    solver_get_requests(gs, ctx, tables, buffers);
    ASSERT_GT(buffers.request_count, 0);

    int p = gs.board.current_player;

    float batch_out[kTensorSize] = {};
    generate_tensor_batch(gs.board, ctx, p, buffers.requests, 1, tables, batch_out);

    float single[kTensorSize] = {};
    tensor_for_request(gs.board, ctx, p, buffers.requests[0], tables, single);

    for (int f = 0; f < kTensorSize; ++f) {
        EXPECT_NEAR(batch_out[f], single[f], 1e-5f)
            << "Single-request batch mismatch at feature " << f;
    }
}

// ---------------------------------------------------------------------------
// Batch tensors: all features in [0, 1]
// ---------------------------------------------------------------------------
TEST_F(TensorBatchTest, BatchFeatures_InUnitRange) {
    gs.rolls_left = 0;
    SolverBuffers buffers;
    solver_get_requests(gs, ctx, tables, buffers);
    ASSERT_GT(buffers.request_count, 0);

    int p = gs.board.current_player;
    int n = buffers.request_count;

    std::vector<float> batch_out(n * kTensorSize, 0.0f);
    generate_tensor_batch(gs.board, ctx, p, buffers.requests, n, tables, batch_out.data());

    for (int i = 0; i < n; ++i) {
        const float* t = batch_out.data() + i * kTensorSize;
        for (int f = 0; f < kTensorSize; ++f) {
            EXPECT_GE(t[f], 0.0f - 1e-5f)
                << "Request " << i << " feature " << f << " below 0";
            EXPECT_LE(t[f], 1.0f + 1e-5f)
                << "Request " << i << " feature " << f << " above 1";
        }
    }
}
