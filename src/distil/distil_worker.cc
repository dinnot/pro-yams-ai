#include "distil/distil_worker.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "engine/game_flow.h"
#include "engine/scoring.h"
#include "engine/tensor.h"

namespace {

// Mix the worker seed with a monotonically increasing counter to derive a
// fresh per-game RNG seed on recycle. A simple multiply-XOR is plenty for
// decorrelating recycle streams within and across workers.
inline uint64_t derive_recycle_seed(uint64_t worker_seed, uint64_t counter) {
    return worker_seed ^ (counter * 0x9E3779B97F4A7C15ULL);
}

template <typename Traits>
inline void reset_game_in_place(GameInstanceT<Traits>& game,
                                uint64_t new_seed) {
    game.rng = RNG(new_seed);
    init_game<Traits>(game.state, game.ctx, game.rng);
    game.trajectory_length = 0;
    game.phase = GamePhase::kNeedRequests;
    game.use_past_opponent = false;
    game.past_opponent_player = -1;
}

}  // namespace

template <typename Traits>
void distil_worker_thread(GameQueueT<Traits>& available,
                          Teacher<Traits>& teacher,
                          ShuffleQueueT<Traits>& shuffle_queue,
                          const PrecomputedTables& tables,
                          const SolverConfig& solver_config,
                          uint64_t worker_seed,
                          DistilWorkerStats* stats) {
    using GI = GameInstanceT<Traits>;
    using Sample = TrainingSampleT<Traits>;
    constexpr int kT = Traits::kTensorSize;

    // Per-worker scratch buffers (avoid per-turn heap traffic).
    std::vector<float> tensor_scratch(
        static_cast<size_t>(kMaxAfterstateRequests) * kT);
    std::vector<Sample> sample_scratch(kMaxAfterstateRequests);

    // Pre-build the greedy solver config once — fields below override any
    // exploration / blending settings the caller may have set.
    SolverConfig greedy_cfg = solver_config;
    greedy_cfg.placement_temperature = 0.0;
    greedy_cfg.hold_temperature      = 0.0;
    greedy_cfg.exploration_enabled   = false;
    greedy_cfg.heuristic_weight      = 0.0;   // Teacher's evs are already final.
    greedy_cfg.compute_pre_roll_ev   = false;

    uint64_t recycle_counter = 0;

    while (true) {
        GI* game = available.pop();
        if (game == nullptr) break;   // shutdown sentinel

        // ---------------------------------------------------------------
        // Per-turn setup.
        // ---------------------------------------------------------------
        game->solver_buffers.dp_computed = false;
        game->solver_buffers.evs_blended = false;

        solver_get_requests<Traits>(game->state, game->ctx, tables,
                                    game->solver_buffers);

        bool became_terminal     = false;
        bool degenerate_terminal = false;  // True iff the "no placement, no
                                           // reroll" fallback ended the game
                                           // — skip get_game_result then.

        // ---------------------------------------------------------------
        // 0-requests fast paths (mirrors self_play/worker.cc 70..106).
        // ---------------------------------------------------------------
        if (game->solver_buffers.request_count == 0) {
            if (game->state.rolls_left <= 0) {
                const auto& all = game->ctx.legal_all[game->state.board.current_player];
                if (all.count > 0) {
                    perform_placement<Traits>(game->state, game->ctx,
                                              all.placements[0].column,
                                              all.placements[0].row, game->rng);
                    became_terminal = is_terminal<Traits>(game->state.board);
                } else {
                    // No legal placement, no rerolls — degenerate, score as draw.
                    game->final_duel_margin = 0;
                    game->result            = 0.5;
                    became_terminal         = true;
                    degenerate_terminal     = true;
                }
            } else {
                // No requests but rerolls available — reroll everything.
                perform_reroll<Traits>(game->state, 0, game->rng);
            }
        } else {
            // -----------------------------------------------------------
            // Normal path: tensors → teacher.evaluate → emit samples → resolve.
            //
            // Within a single turn the board state (and hence requests +
            // tensors + teacher evs) is invariant across rerolls — only the
            // dice change. So we only emit training samples on the FIRST
            // roll of the turn (rolls_left == 2). Emitting on every roll
            // would push 2..3× duplicate samples into the shuffle queue,
            // destroying SGD batch diversity. Tensors are skipped on
            // subsequent rolls when the teacher doesn't need them (heuristic).
            // -----------------------------------------------------------
            const int n = game->solver_buffers.request_count;
            assert(n <= kMaxAfterstateRequests);

            const bool emit_samples = (game->state.rolls_left == 2);
            const bool need_tensors = emit_samples || teacher.needs_tensor_input();

            if (need_tensors) {
                generate_tensor_batch<Traits>(
                    game->state.board, game->ctx,
                    game->state.board.current_player,
                    game->solver_buffers.requests, n,
                    tables,
                    tensor_scratch.data());
            }

            // Compute training targets (squashed) into a scratch array;
            // solver_resolve consumes the RAW evs written into buffers.evs.
            // Decoupling avoids the E[tanh(X)] ≠ tanh(E[X]) bias that
            // averaging squashed margins introduces into expectimax.
            double targets[kMaxAfterstateRequests];
            teacher.evaluate(game->state.board, game->ctx,
                             game->solver_buffers.requests, n,
                             need_tensors ? tensor_scratch.data() : nullptr,
                             emit_samples ? targets : nullptr,
                             game->solver_buffers.evs);

            if (emit_samples) {
                for (int i = 0; i < n; ++i) {
                    std::memcpy(sample_scratch[i].state,
                                tensor_scratch.data() + static_cast<size_t>(i) * kT,
                                kT * sizeof(float));
                    sample_scratch[i].target = targets[i];
                }
                shuffle_queue.add_batch(sample_scratch.data(), n);
                if (stats) {
                    stats->samples_emitted.fetch_add(n, std::memory_order_relaxed);
                }
            }

            // Resolve greedily — teacher's raw evs in buffers.evs drive the
            // action via expectimax.
            SolverResult result = solver_resolve<Traits>(
                game->state, game->ctx, tables, game->solver_buffers,
                greedy_cfg, game->rng);

            if (result.should_place) {
                perform_placement<Traits>(game->state, game->ctx,
                                          result.placement.column,
                                          result.placement.row, game->rng);
                if (stats) {
                    stats->turns_processed.fetch_add(1, std::memory_order_relaxed);
                }
                became_terminal = is_terminal<Traits>(game->state.board);
            } else {
                perform_reroll<Traits>(game->state, result.hold_mask, game->rng);
            }
        }

        // ---------------------------------------------------------------
        // Tail: either recycle (terminal) or requeue.
        // ---------------------------------------------------------------
        if (became_terminal) {
            if (!degenerate_terminal) {
                int duel = get_game_result<Traits>(game->state, game->ctx);
                game->final_duel_margin = duel;
                game->result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
            }
            if (stats) {
                stats->games_completed.fetch_add(1, std::memory_order_relaxed);
            }
            uint64_t new_seed = derive_recycle_seed(worker_seed, ++recycle_counter);
            reset_game_in_place<Traits>(*game, new_seed);
            available.push(game);
        } else {
            game->phase = GamePhase::kNeedRequests;
            available.push(game);
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template void distil_worker_thread<Yams1v1>(GameQueueT<Yams1v1>&,
                                            Teacher<Yams1v1>&,
                                            ShuffleQueueT<Yams1v1>&,
                                            const PrecomputedTables&,
                                            const SolverConfig&,
                                            uint64_t,
                                            DistilWorkerStats*);
template void distil_worker_thread<Yams2v2>(GameQueueT<Yams2v2>&,
                                            Teacher<Yams2v2>&,
                                            ShuffleQueueT<Yams2v2>&,
                                            const PrecomputedTables&,
                                            const SolverConfig&,
                                            uint64_t,
                                            DistilWorkerStats*);
