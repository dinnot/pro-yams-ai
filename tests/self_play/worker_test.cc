#include <gtest/gtest.h>
#include "self_play/worker.h"
#include "self_play/game_instance.h"
#include "self_play/game_queues.h"
#include "solver/precomputed_tables.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"

#include <atomic>
#include <cstring>
#include <thread>

// Shared precomputed tables — expensive to build, so init once.
static PrecomputedTables* g_tables = nullptr;

static void ensure_tables() {
    if (!g_tables) {
        g_tables = new PrecomputedTables();
        init_precomputed_tables(*g_tables);
    }
}

static SolverConfig greedy_config() {
    return {0.0, 0.0, false};
}

static std::unique_ptr<GameInstance> make_game(int id = 0) {
    ensure_tables();
    auto g = std::make_unique<GameInstance>();
    g->game_id           = id;
    g->rng               = RNG(static_cast<uint64_t>(42 + id));
    g->trajectory_length = 0;
    g->result            = 0.0;
    init_game(g->state, g->ctx, g->rng);
    g->phase = GamePhase::kNeedRequests;
    return g;
}

// ---------------------------------------------------------------------------
// kNeedRequests → pending queue with tensors
// ---------------------------------------------------------------------------
TEST(WorkerTest, NeedRequests_PushesToPending) {
    ensure_tables();
    GameQueue available, completed;
    BatchManager bm(1, 4096, false, 5);
    std::atomic<bool> shutdown{false};

    auto game = make_game();
    available.push(game.get());
    available.push(nullptr);  // sentinel

    std::thread wt(worker_thread<Yams1v1>,
                   std::ref(available), std::ref(bm),
                   static_cast<BatchManager*>(nullptr),
                   std::ref(completed),
                   std::ref(*g_tables), greedy_config(), std::ref(shutdown));
    wt.join();

    EXPECT_EQ(game->phase, GamePhase::kWaitingInference);
    EXPECT_GT(game->solver_buffers.request_count, 0);
    EXPECT_LE(game->solver_buffers.request_count, kMaxAfterstateRequests);
    EXPECT_EQ(completed.size(), 0);
}

// ---------------------------------------------------------------------------
// kNeedResolve with EVs pre-set + rolls_left=0 → placement forced, trajectory recorded
// ---------------------------------------------------------------------------
TEST(WorkerTest, NeedResolve_RecordsTrajectory) {
    ensure_tables();
    GameQueue available, completed;
    BatchManager bm(1, 4096, false, 5);
    std::atomic<bool> shutdown{false};

    auto game = make_game(1);

    // Generate requests first.
    solver_get_requests(game->state, game->ctx, *g_tables, game->solver_buffers);

    // Fill EVs with constant 0.5.
    for (int i = 0; i < game->solver_buffers.request_count; ++i)
        game->solver_buffers.evs[i] = 0.5;

    // Force rolls_left=0 so solver must place (short-circuits reroll logic).
    game->state.rolls_left = 0;
    game->current_turn_start_ev = 0.5; // Simulate EV captured at start of turn
    game->phase = GamePhase::kNeedResolve;
    available.push(game.get());
    available.push(nullptr);  // sentinel

    std::thread wt(worker_thread<Yams1v1>,
                   std::ref(available), std::ref(bm),
                   static_cast<BatchManager*>(nullptr),
                   std::ref(completed),
                   std::ref(*g_tables), greedy_config(), std::ref(shutdown));
    wt.join();

    // With rolls_left=0 solver always places, so trajectory must have 1 step.
    EXPECT_EQ(game->trajectory_length, 1);
    // Game should be in exactly one queue (available=kNeedRequests or completed).
    int total_out = available.size() + completed.size();
    EXPECT_EQ(total_out, 1);
}

// ---------------------------------------------------------------------------
// Complete game trajectory — all 156 steps filled
// ---------------------------------------------------------------------------
TEST(WorkerTest, FullGame_TrajectoryLength) {
    ensure_tables();

    // Run a complete game by manually cycling the worker loop ourselves.
    auto game = make_game(7);
    SolverConfig cfg = greedy_config();

    // Manually simulate: generate requests, fill with constant EVs, resolve.
    // Repeat until the game completes. Limit iterations to avoid infinite loop.
    int iterations = 0;
    constexpr int kMaxIter = 10000;

    while (game->phase != GamePhase::kCompleted && iterations < kMaxIter) {
        if (game->phase == GamePhase::kNeedRequests) {
            game->solver_buffers.dp_computed = false;
            solver_get_requests(game->state, game->ctx, *g_tables,
                                game->solver_buffers);
            ASSERT_LE(game->solver_buffers.request_count, kMaxAfterstateRequests);
            // Simulate coordinator: fill EVs.
            for (int i = 0; i < game->solver_buffers.request_count; ++i)
                game->solver_buffers.evs[i] = 0.5;
            game->phase = GamePhase::kNeedResolve;

        } else if (game->phase == GamePhase::kNeedResolve) {
            SolverResult res = solver_resolve(game->state, game->ctx, *g_tables,
                                              game->solver_buffers, cfg, game->rng);

            if (game->state.rolls_left == 2) {
                game->current_turn_start_ev = res.expected_value;
            }

            if (res.should_place) {
                ASSERT_GE(res.chosen_request_idx, 0);
                TrajectoryStep& step = game->trajectory[game->trajectory_length];
                step.value  = game->current_turn_start_ev;
                step.player = static_cast<int8_t>(game->state.board.current_player);
                
                generate_tensor_batch(
                    game->state.board, game->ctx, game->state.board.current_player,
                    &game->solver_buffers.requests[res.chosen_request_idx], 1,
                    *g_tables, step.tensor
                );
                
                ++game->trajectory_length;

                perform_placement(game->state, game->ctx,
                                  res.placement.column, res.placement.row, game->rng);
                if (is_terminal(game->state.board)) {
                    int duel = get_game_result(game->state, game->ctx);
                    game->result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
                    game->phase  = GamePhase::kCompleted;
                } else {
                    game->phase = GamePhase::kNeedRequests;
                }
            } else {
                perform_reroll(game->state, res.hold_mask, game->rng);
                game->phase = GamePhase::kNeedResolve;
            }
        }
        ++iterations;
    }

    EXPECT_EQ(game->phase, GamePhase::kCompleted);
    EXPECT_EQ(game->trajectory_length, GameInstance::kMaxTrajectorySteps)
        << "Expected exactly 156 trajectory steps";
    double r = game->result;
    EXPECT_TRUE(r == 0.0 || r == 0.5 || r == 1.0)
        << "Invalid result: " << r;
}

TEST(WorkerTest, Caching_PreventsDoubleBlending) {
    ensure_tables();
    GameQueue available, completed;
    BatchManager bm(1, 4096, false, 5);
    std::atomic<bool> shutdown{false};

    auto game = make_game(42);
    SolverConfig config = greedy_config();
    config.heuristic_weight = 0.5;

    // Step 1: kNeedRequests sets flags
    game->phase = GamePhase::kNeedRequests;
    available.push(game.get());
    available.push(nullptr);
    worker_thread<Yams1v1>(available, bm, nullptr, completed, *g_tables, config, shutdown);
    
    EXPECT_FALSE(game->solver_buffers.dp_computed);
    EXPECT_FALSE(game->solver_buffers.evs_blended);

    // Step 2: kNeedResolve blends first time
    game->phase = GamePhase::kNeedResolve;
    for (int i = 0; i < game->solver_buffers.request_count; ++i) game->solver_buffers.evs[i] = 0.6;
    available.push(game.get());
    available.push(nullptr);
    worker_thread<Yams1v1>(available, bm, nullptr, completed, *g_tables, config, shutdown);

    EXPECT_TRUE(game->solver_buffers.evs_blended);
    double blended_ev = game->solver_buffers.evs[0];

    // Step 3: kNeedResolve again (simulating reroll) should NOT blend again
    game->phase = GamePhase::kNeedResolve;
    game->solver_buffers.dp_computed = true; // Pretend we already solved once
    // If it blends again, the EV will change. If it doesn't, it stays same.
    available.push(game.get());
    available.push(nullptr);
    worker_thread<Yams1v1>(available, bm, nullptr, completed, *g_tables, config, shutdown);

    EXPECT_EQ(game->solver_buffers.evs[0], blended_ev) << "EV should not change on second resolve (reroll)";
}
