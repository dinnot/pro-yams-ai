#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "engine/board_init.h"
#include "engine/game_traits.h"
#include "engine/placement.h"
#include "engine/rng.h"
#include "engine/tensor.h"
#include "solver/precomputed_tables.h"

// ---------------------------------------------------------------------------
// Tensor rotational-invariance tests (Task 5 validation).
//
// Canonical view: tensor for player p emits per-player groups in the order
//   [canonical[0]=p, canonical[1]=(p+1)%N, ..., canonical[N-1]=(p+N-1)%N]
// and per-pairing groups in Traits::kCanonicalPairing order.
//
// Invariant: rotating the four sheets cyclically by k and asking for the
// tensor from the (active+k)-th player must produce a BIT-IDENTICAL output
// to the original tensor from active. This proves the network sees no raw
// seat indices.
// ---------------------------------------------------------------------------

class TensorRotationTest : public ::testing::Test {
protected:
    static PrecomputedTables tables;
    static bool initialised;
    static void SetUpTestSuite() {
        if (!initialised) {
            init_precomputed_tables(tables);
            initialised = true;
        }
    }
};
PrecomputedTables TensorRotationTest::tables;
bool TensorRotationTest::initialised = false;

namespace {

// Rotate a 2v2 board+context by k seats. After rotation:
//   new player p holds what player (p - k + N) % N had before.
// Equivalently: old player p's data moves to new player (p + k) % N.
// All per-player fields of board+ctx are permuted accordingly.
void rotate_2v2(BoardState2v2& board, GameContext2v2& ctx, int k) {
    constexpr int N = Yams2v2::kNumPlayers;
    k = ((k % N) + N) % N;
    if (k == 0) return;

    BoardState2v2 nb = board;
    GameContext2v2 nc = ctx;
    for (int p = 0; p < N; ++p) {
        const int src = (p - k + N) % N;
        // Board cells
        for (int c = 0; c < kNumColumns; ++c)
            for (int r = 0; r < kNumRows; ++r)
                nb.cells[p][c][r] = board.cells[src][c][r];
        // Per-player ctx
        for (int c = 0; c < kNumColumns; ++c) {
            nc.upper_sum[p][c]         = ctx.upper_sum[src][c];
            nc.ss_scratched[p][c]      = ctx.ss_scratched[src][c];
            nc.ls_scratched[p][c]      = ctx.ls_scratched[src][c];
            nc.lower_has_scratch[p][c] = ctx.lower_has_scratch[src][c];
        }
        nc.legal_all[p]               = ctx.legal_all[src];
        nc.legal_no_turbo[p]          = ctx.legal_no_turbo[src];
        nc.non_turbo_cells_remaining[p] = ctx.non_turbo_cells_remaining[src];
    }
    // golden_max, coefficients, cells_filled are not per-player.
    board = nb;
    ctx = nc;
}

// Build a populated 2v2 board+ctx: each player has scored a handful of cells
// (different rows/columns per player) so the per-player slots in the tensor
// carry distinguishable signal.
void make_populated_2v2(BoardState2v2& board, GameContext2v2& ctx, uint64_t seed) {
    RNG rng(seed);
    init_board<Yams2v2>(board, rng);
    init_context<Yams2v2>(ctx, board);

    // Choose a few legal placements for each player. Use distinct rows so the
    // sheets are visibly different.
    const int rows_per_player[Yams2v2::kNumPlayers][3] = {
        {kRow1s, kRow6s, kRowFH},   // P0
        {kRow2s, kRowSS, kRowK},    // P1
        {kRow3s, kRowLS, kRowSTR},  // P2
        {kRow4s, kRow5s, kRowU8},   // P3
    };
    const int8_t scores[3] = {5, 28, 50};  // Distinguish placements
    for (int p = 0; p < Yams2v2::kNumPlayers; ++p) {
        for (int k = 0; k < 3; ++k) {
            const int col = kColFree;  // Free column — any order
            const int row = rows_per_player[p][k];
            int8_t score = scores[k];
            // Stagger scores across players so golden_max doesn't block.
            score = static_cast<int8_t>(std::min<int>(100, score + p * 2));
            apply_placement<Yams2v2>(p, col, row, score, board, ctx);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Rotation by 1: tensor(rot1(board), active=1) == tensor(board, active=0).
// The classic single-step cyclic shift.
// ---------------------------------------------------------------------------
TEST_F(TensorRotationTest, RotationBy1_BitEqual) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_populated_2v2(board, ctx, /*seed=*/12345);

    std::vector<float> t_active0(Yams2v2::kTensorSize);
    generate_tensor<Yams2v2>(board, ctx, /*player=*/0, tables, t_active0.data());

    BoardState2v2 rot_board = board;
    GameContext2v2 rot_ctx = ctx;
    rotate_2v2(rot_board, rot_ctx, /*k=*/1);

    std::vector<float> t_active1_rot(Yams2v2::kTensorSize);
    generate_tensor<Yams2v2>(rot_board, rot_ctx, /*player=*/1, tables, t_active1_rot.data());

    ASSERT_EQ(t_active0.size(), t_active1_rot.size());
    for (int i = 0; i < Yams2v2::kTensorSize; ++i) {
        EXPECT_FLOAT_EQ(t_active0[i], t_active1_rot[i])
            << "mismatch at feature " << i;
    }
}

// ---------------------------------------------------------------------------
// Rotation by 2: tensor(rot2(board), active=2) == tensor(board, active=0).
// This is the within-team-PAIR swap: P0↔P2 AND P1↔P3 simultaneously.
// (Swapping only P0↔P2 doesn't preserve the canonical view because P1, P3
//  retain their right/left-neighbor identities — see plan note.)
// ---------------------------------------------------------------------------
TEST_F(TensorRotationTest, RotationBy2_BitEqual) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_populated_2v2(board, ctx, /*seed=*/67890);

    std::vector<float> t_active0(Yams2v2::kTensorSize);
    generate_tensor<Yams2v2>(board, ctx, /*player=*/0, tables, t_active0.data());

    BoardState2v2 rot_board = board;
    GameContext2v2 rot_ctx = ctx;
    rotate_2v2(rot_board, rot_ctx, /*k=*/2);

    std::vector<float> t_active2_rot(Yams2v2::kTensorSize);
    generate_tensor<Yams2v2>(rot_board, rot_ctx, /*player=*/2, tables, t_active2_rot.data());

    for (int i = 0; i < Yams2v2::kTensorSize; ++i) {
        EXPECT_FLOAT_EQ(t_active0[i], t_active2_rot[i])
            << "mismatch at feature " << i;
    }
}

// ---------------------------------------------------------------------------
// Rotation by 3: tensor(rot3(board), active=3) == tensor(board, active=0).
// Covers the remaining cyclic step.
// ---------------------------------------------------------------------------
TEST_F(TensorRotationTest, RotationBy3_BitEqual) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_populated_2v2(board, ctx, /*seed=*/99999);

    std::vector<float> t_active0(Yams2v2::kTensorSize);
    generate_tensor<Yams2v2>(board, ctx, /*player=*/0, tables, t_active0.data());

    BoardState2v2 rot_board = board;
    GameContext2v2 rot_ctx = ctx;
    rotate_2v2(rot_board, rot_ctx, /*k=*/3);

    std::vector<float> t_active3_rot(Yams2v2::kTensorSize);
    generate_tensor<Yams2v2>(rot_board, rot_ctx, /*player=*/3, tables, t_active3_rot.data());

    for (int i = 0; i < Yams2v2::kTensorSize; ++i) {
        EXPECT_FLOAT_EQ(t_active0[i], t_active3_rot[i])
            << "mismatch at feature " << i;
    }
}

// ---------------------------------------------------------------------------
// Tensor size traits — make sure the layout sums match the declared sizes.
// ---------------------------------------------------------------------------
TEST(TensorTraits, Yams1v1_TensorSizes) {
    // V1 layout (frozen) is the prefix; Group G is appended for V2.
    EXPECT_EQ(Yams1v1::kTensorSizeV1, 986);
    EXPECT_EQ(Yams1v1::kGroupGSize, 24);          // 2 * 2 players * 6 cols
    EXPECT_EQ(Yams1v1::kTensorSize, 986 + 24);    // latest = 1010
    EXPECT_EQ(Yams1v1::kNumPairings, 1);
    EXPECT_EQ(tensor_size_for_version<Yams1v1>(kTensorVersionV1), 986);
    EXPECT_EQ(tensor_size_for_version<Yams1v1>(kTensorVersionLatest), 1010);
}

TEST(TensorTraits, Yams2v2_TensorSizes) {
    // V1: 624 (A) + 48 (B.1) + 336 (B.2) + 312 (C) + 14 (D) + 432 (E) + 360 (F)
    EXPECT_EQ(Yams2v2::kTensorSizeV1, 2126);
    EXPECT_EQ(Yams2v2::kGroupGSize, 48);          // 2 * 4 players * 6 cols
    EXPECT_EQ(Yams2v2::kTensorSize, 2126 + 48);   // latest = 2174
    EXPECT_EQ(Yams2v2::kNumPairings, 4);
    EXPECT_EQ(tensor_size_for_version<Yams2v2>(kTensorVersionV1), 2126);
    EXPECT_EQ(tensor_size_for_version<Yams2v2>(kTensorVersionLatest), 2174);
}

// Group G is appended after Group F, so the V1 layout is a byte-exact prefix
// of the V2 tensor. Phase 1: Group G is placeholder zeros. (When the real
// SS/LS interlock features land, replace the zero-check with their expected
// values — the prefix [0, kTensorSizeV1) stays pinned by the content tests in
// tensor_test.cc, which only assert offsets inside the V1 region.)
TEST_F(TensorRotationTest, GroupGIsAppendedZeros_Phase1) {
    BoardState2v2 board;
    GameContext2v2 ctx;
    make_populated_2v2(board, ctx, /*seed=*/7);

    std::vector<float> out(Yams2v2::kTensorSize, -1.0f);
    generate_tensor<Yams2v2>(board, ctx, /*player=*/0, tables, out.data());
    for (int i = Yams2v2::kTensorSizeV1; i < Yams2v2::kTensorSize; ++i) {
        EXPECT_EQ(out[i], 0.0f) << "Group G slot " << i << " should be zero in Phase 1";
    }
}
