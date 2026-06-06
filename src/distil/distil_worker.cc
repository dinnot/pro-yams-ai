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
                          DistilReplayBufferT<Traits>& replay_buffer,
                          const PrecomputedTables& tables,
                          const SolverConfig& solver_config,
                          uint64_t worker_seed,
                          double samples_per_games_rate,
                          DistilWorkerStats* stats) {
    using GI = GameInstanceT<Traits>;
    using Sample = TrainingSampleT<Traits>;
    constexpr int kT = Traits::kTensorSize;

    // Per-worker scratch buffers (avoid per-turn heap traffic).
    std::vector<float> tensor_scratch(
        static_cast<size_t>(kMaxAfterstateRequests) * kT);
    std::vector<Sample> sample_scratch(kMaxAfterstateRequests);

    // Dedicated RNG for the keep-rate Bernoulli. Kept independent of
    // game->rng so subsampling decisions don't perturb gameplay (dice rolls
    // / placements stay identical to a rate=1.0 run with the same seed).
    RNG subsample_rng(worker_seed ^ 0xDEADC0FFEEBADF00ULL);
    // Treat values within float epsilon of 1.0 as "no subsampling" so we
    // skip the per-sample draw entirely on the default path.
    const bool subsample_enabled = (samples_per_games_rate < 1.0 - 1e-9);

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
        // Per-turn setup. We run a FULL turn inline (request → teacher
        // eval → emit → resolve-and-reroll loop → placement), instead of
        // pushing the game back to the queue between rerolls. The board
        // state is invariant across a turn's rerolls (only the dice
        // change), so the requests, tensors, teacher evs and DP table can
        // be computed once and reused across rerolls — solver_resolve
        // short-circuits its DP rebuild when dp_computed is true.
        // ---------------------------------------------------------------
        game->solver_buffers.dp_computed = false;
        game->solver_buffers.evs_blended = false;

        bool became_terminal     = false;
        bool degenerate_terminal = false;  // True iff the "no placement, no
                                           // reroll" fallback ended the game
                                           // — skip get_game_result then.

        solver_get_requests<Traits>(game->state, game->ctx, tables,
                                    game->solver_buffers);

        // The zero-request edge case may need to reroll and retry several
        // times before we either find a non-empty request set (and fall
        // into the normal path) or exhaust our rerolls (and force a
        // fallback placement). Looping inline avoids the queue ping-pong.
        bool turn_done = false;
        while (!turn_done) {
            if (game->solver_buffers.request_count == 0) {
                if (game->state.rolls_left <= 0) {
                    // No legal request set and no rerolls left — force a
                    // fallback placement at the first legal cell.
                    const auto& all = game->ctx.legal_all[game->state.board.current_player];
                    if (all.count > 0) {
                        perform_placement<Traits>(game->state, game->ctx,
                                                  all.placements[0].column,
                                                  all.placements[0].row, game->rng);
                        became_terminal = is_terminal<Traits>(game->state.board);
                    } else {
                        // No legal placement at all — degenerate, score as draw.
                        game->final_duel_margin = 0;
                        game->result            = 0.5;
                        became_terminal         = true;
                        degenerate_terminal     = true;
                    }
                    if (stats) {
                        stats->turns_processed.fetch_add(1, std::memory_order_relaxed);
                    }
                    turn_done = true;
                } else {
                    // No requests but rerolls available — reroll all dice
                    // and recompute requests for the new dice. Board is
                    // unchanged so dp_computed can stay reset; we never
                    // called solver_resolve yet anyway.
                    perform_reroll<Traits>(game->state, 0, game->rng);
                    solver_get_requests<Traits>(game->state, game->ctx, tables,
                                                game->solver_buffers);
                }
            } else {
                // -------------------------------------------------------
                // Normal path. Tensors + teacher.evaluate + sample emit
                // happen exactly ONCE per turn (board is invariant across
                // rerolls). The reroll loop below reuses the cached evs
                // and DP, so subsequent rerolls within this turn are just
                // a DP lookup.
                // -------------------------------------------------------
                const int n = game->solver_buffers.request_count;
                assert(n <= kMaxAfterstateRequests);

                generate_tensor_batch<Traits>(
                    game->state.board, game->ctx,
                    game->state.board.current_player,
                    game->solver_buffers.requests, n,
                    tables,
                    tensor_scratch.data());

                // Compute training targets (squashed) into a scratch array;
                // solver_resolve consumes the RAW evs written into buffers.evs.
                // Decoupling avoids the E[tanh(X)] ≠ tanh(E[X]) bias that
                // averaging squashed margins introduces into expectimax.
                double targets[kMaxAfterstateRequests];
                // tensor_scratch holds the STUDENT (latest) tensor at stride kT.
                // An NN teacher on an older append-only version reads the first
                // input_size <= kT columns of each row (a byte-exact prefix);
                // a heuristic teacher ignores the tensor entirely.
                teacher.evaluate(game->state.board, game->ctx,
                                 game->solver_buffers.requests, n,
                                 tensor_scratch.data(), kT,
                                 targets,
                                 game->solver_buffers.evs);

                int kept = 0;
                if (subsample_enabled) {
                    // Independent Bernoulli per request. Approximate fraction
                    // (binomial(n, rate)); cheaper than reservoir sampling
                    // and the user explicitly OK'd inexact retention.
                    for (int i = 0; i < n; ++i) {
                        if (subsample_rng.uniform_double() >= samples_per_games_rate) {
                            continue;
                        }
                        std::memcpy(sample_scratch[kept].state,
                                    tensor_scratch.data() + static_cast<size_t>(i) * kT,
                                    kT * sizeof(float));
                        sample_scratch[kept].target = targets[i];
                        ++kept;
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        std::memcpy(sample_scratch[i].state,
                                    tensor_scratch.data() + static_cast<size_t>(i) * kT,
                                    kT * sizeof(float));
                        sample_scratch[i].target = targets[i];
                    }
                    kept = n;
                }
                if (kept > 0) {
                    replay_buffer.add_batch(sample_scratch.data(), kept);
                    if (stats) {
                        stats->samples_emitted.fetch_add(kept, std::memory_order_relaxed);
                    }
                }

                // Resolve greedily — teacher's raw evs in buffers.evs drive
                // the action via expectimax. After a reroll, the dice
                // change but the board (and hence evs) doesn't, so the
                // next solver_resolve reuses the cached DP via
                // dp_computed=true (set inside the first solver_resolve
                // call).
                while (true) {
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
                        break;
                    } else {
                        perform_reroll<Traits>(game->state, result.hold_mask, game->rng);
                    }
                }
                turn_done = true;
            }
        }

        // ---------------------------------------------------------------
        // Tail: either recycle (terminal) or requeue for the next turn.
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
        } else {
            game->phase = GamePhase::kNeedRequests;
        }
        available.push(game);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template void distil_worker_thread<Yams1v1>(GameQueueT<Yams1v1>&,
                                            Teacher<Yams1v1>&,
                                            DistilReplayBufferT<Yams1v1>&,
                                            const PrecomputedTables&,
                                            const SolverConfig&,
                                            uint64_t,
                                            double,
                                            DistilWorkerStats*);
template void distil_worker_thread<Yams2v2>(GameQueueT<Yams2v2>&,
                                            Teacher<Yams2v2>&,
                                            DistilReplayBufferT<Yams2v2>&,
                                            const PrecomputedTables&,
                                            const SolverConfig&,
                                            uint64_t,
                                            double,
                                            DistilWorkerStats*);
