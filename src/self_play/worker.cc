#include "self_play/worker.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>

#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "self_play/game_instance.h"
#include "heuristic/heuristic_bot.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"
#include "engine/tensor.h"

namespace {

inline int64_t elapsed_us(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
}

inline std::chrono::steady_clock::time_point debug_now(SelfPlayDebugStats* stats) {
    return stats ? std::chrono::steady_clock::now()
                 : std::chrono::steady_clock::time_point{};
}

}  // namespace

template <typename Traits>
void worker_thread(GameQueueT<Traits>& available,
                   BatchManagerT<Traits>& batch_manager,
                   BatchManagerT<Traits>* opponent_batch_manager,
                   GameQueueT<Traits>& completed,
                   const PrecomputedTables& tables, const SolverConfig& config,
                   std::atomic<bool>& shutdown,
                   SelfPlayDebugStats* debug_stats) {
    (void)shutdown;  // checked via nullptr sentinel from available queue
    using GI = GameInstanceT<Traits>;
    using IB = InferenceBatchT<Traits>;
    using BM = BatchManagerT<Traits>;

    while (true) {
        GI* game = available.pop();
        if (game == nullptr) break;  // shutdown sentinel

        if (game->phase == GamePhase::kNeedRequests) {
            if (debug_stats) {
                debug_stats->worker_need_requests.fetch_add(1, std::memory_order_relaxed);
            }
            // -----------------------------------------------------------
            // Step 1: generate afterstate requests.
            // -----------------------------------------------------------
            game->solver_buffers.dp_computed = false;
            game->solver_buffers.evs_blended = false;

            auto t_req0 = debug_now(debug_stats);
            solver_get_requests<Traits>(game->state, game->ctx, tables,
                                        game->solver_buffers);
            auto t_req1 = debug_now(debug_stats);
            if (debug_stats) {
                debug_stats->solver_get_requests_us.fetch_add(elapsed_us(t_req0, t_req1),
                                                              std::memory_order_relaxed);
                debug_stats->worker_requests.fetch_add(game->solver_buffers.request_count,
                                                       std::memory_order_relaxed);
            }

            if (game->solver_buffers.request_count == 0) {
                if (game->state.rolls_left <= 0) {
                    const auto& all = game->ctx.legal_all[game->state.board.current_player];
                    if (all.count > 0) {
                        perform_placement<Traits>(game->state, game->ctx,
                                                  all.placements[0].column,
                                                  all.placements[0].row, game->rng);
                        if (debug_stats) {
                            debug_stats->worker_placements.fetch_add(1, std::memory_order_relaxed);
                        }
                        if (is_terminal<Traits>(game->state.board)) {
                            int duel = get_game_result<Traits>(game->state, game->ctx);
                            game->final_duel_margin = duel;
                            game->result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
                            game->phase = GamePhase::kCompleted;
                            if (debug_stats) {
                                debug_stats->worker_completed_games.fetch_add(1, std::memory_order_relaxed);
                            }
                            completed.push(game);
                        } else {
                            available.push(game);
                        }
                    } else {
                        game->phase = GamePhase::kCompleted;
                        game->result = 0.5;
                        completed.push(game);
                    }
                } else {
                    perform_reroll<Traits>(game->state, 0, game->rng);
                    if (debug_stats) {
                        debug_stats->worker_rerolls.fetch_add(1, std::memory_order_relaxed);
                    }
                    available.push(game);
                }
                continue;
            }

            assert(game->solver_buffers.request_count <= kMaxAfterstateRequests);

            // -----------------------------------------------------------
            // Step 2: generate tensors directly into the shared batch buffer.
            // -----------------------------------------------------------
            BM* target_bm = &batch_manager;
            if (opponent_batch_manager != nullptr &&
                game->use_past_opponent &&
                game->past_opponent_player >= 0 &&
                Traits::are_teammates(game->state.board.current_player,
                                      game->past_opponent_player)) {
                target_bm = opponent_batch_manager;
            }

            IB* reserved_batch = nullptr;
            int batch_offset = 0;
            auto t_reserve0 = debug_now(debug_stats);
            float* dest_ptr = target_bm->reserve(
                game->solver_buffers.request_count, reserved_batch, batch_offset);
            auto t_reserve1 = debug_now(debug_stats);
            if (debug_stats) {
                debug_stats->batch_reserve_us.fetch_add(elapsed_us(t_reserve0, t_reserve1),
                                                        std::memory_order_relaxed);
            }

            if (dest_ptr == nullptr) {
                break;
            }

            auto t_tensor0 = debug_now(debug_stats);
            generate_tensor_batch<Traits>(
                game->state.board, game->ctx,
                game->state.board.current_player,
                game->solver_buffers.requests,
                game->solver_buffers.request_count,
                tables,
                dest_ptr);
            auto t_tensor1 = debug_now(debug_stats);
            if (debug_stats) {
                debug_stats->tensor_batch_us.fetch_add(elapsed_us(t_tensor0, t_tensor1),
                                                       std::memory_order_relaxed);
                debug_stats->worker_tensors.fetch_add(game->solver_buffers.request_count,
                                                      std::memory_order_relaxed);
            }

            game->phase = GamePhase::kWaitingInference;
            auto t_commit0 = debug_now(debug_stats);
            target_bm->commit(reserved_batch, game, batch_offset,
                              game->solver_buffers.request_count);
            auto t_commit1 = debug_now(debug_stats);
            if (debug_stats) {
                debug_stats->batch_commit_us.fetch_add(elapsed_us(t_commit0, t_commit1),
                                                       std::memory_order_relaxed);
            }

        } else if (game->phase == GamePhase::kNeedResolve) {
            if (debug_stats) {
                debug_stats->worker_need_resolve.fetch_add(1, std::memory_order_relaxed);
            }
            bool is_first_resolve = !game->solver_buffers.dp_computed;

            if (is_first_resolve) {
                game->current_turn_is_exploratory = false;
            }

            if (is_first_resolve && config.heuristic_weight > 0.0001f) {
                auto t_pure0 = debug_now(debug_stats);
                SolverResult pure_res = solver_resolve_greedy<Traits>(
                    game->state, game->ctx, tables, game->solver_buffers, true);
                auto t_pure1 = debug_now(debug_stats);
                if (debug_stats) {
                    debug_stats->pure_resolve_us.fetch_add(elapsed_us(t_pure0, t_pure1),
                                                           std::memory_order_relaxed);
                }
                game->current_turn_start_ev = pure_res.pre_roll_ev;
                game->solver_buffers.dp_computed = false;
            }

            if (config.heuristic_weight > 0.0001f && !game->solver_buffers.evs_blended) {
                auto t_blend0 = debug_now(debug_stats);
                std::memcpy(game->solver_buffers.raw_nn_evs, game->solver_buffers.evs,
                            game->solver_buffers.request_count * sizeof(double));

                double heuristic_evs[kMaxAfterstateRequests];
                const HeuristicVersion hv = game->heuristic_version;
                bool is_odds_bot = false;
                auto t_heur0 = debug_now(debug_stats);

                if (static_cast<int>(hv) >= static_cast<int>(HeuristicVersion::V4)) {
                    const ResearchConfig& rcfg = get_research_config_for(hv);
                    is_odds_bot = rcfg.output_win_odds;
                    heuristic_evaluate_research<Traits>(game->state.board, game->ctx,
                                                       game->solver_buffers.requests,
                                                       game->solver_buffers.request_count,
                                                       heuristic_evs, tables, rcfg);
                } else if (hv == HeuristicVersion::V3) {
                    heuristic_evaluate_v3<Traits>(game->state.board, game->ctx,
                                                  game->solver_buffers.requests,
                                                  game->solver_buffers.request_count,
                                                  heuristic_evs, tables);
                } else if (hv == HeuristicVersion::V2) {
                    heuristic_evaluate_v2<Traits>(game->state.board, game->ctx,
                                                  game->solver_buffers.requests,
                                                  game->solver_buffers.request_count,
                                                  heuristic_evs, tables);
                } else {
                    heuristic_evaluate<Traits>(game->state.board, game->ctx,
                                               game->solver_buffers.requests,
                                               game->solver_buffers.request_count,
                                               heuristic_evs);
                }
                auto t_heur1 = debug_now(debug_stats);
                if (debug_stats) {
                    const int hv_idx = static_cast<int>(hv);
                    debug_stats->heuristic_eval_us.fetch_add(elapsed_us(t_heur0, t_heur1),
                                                             std::memory_order_relaxed);
                    debug_stats->heuristic_eval_calls.fetch_add(1, std::memory_order_relaxed);
                    debug_stats->heuristic_requests.fetch_add(game->solver_buffers.request_count,
                                                             std::memory_order_relaxed);
                    if (hv_idx >= 0 && hv_idx < 18) {
                        debug_stats->heuristic_v_counts[hv_idx].fetch_add(1,
                            std::memory_order_relaxed);
                    }
                }

                const bool is_margin_style = (hv != HeuristicVersion::V1 && !is_odds_bot);
                int n = game->solver_buffers.request_count;
                for (int i = 0; i < n; i++) {
                    if (is_odds_bot) {
                        double p = std::max(0.0, std::min(1.0, heuristic_evs[i]));
                        if (config.use_duel_margin_maximization) {
                            heuristic_evs[i] = p * 2.0 - 1.0;
                        } else {
                            heuristic_evs[i] = p;
                        }
                    } else if (is_margin_style) {
                        heuristic_evs[i] = std::tanh(
                            heuristic_evs[i] / config.duel_margin_maximization_scale);
                        if (!config.use_duel_margin_maximization) {
                            heuristic_evs[i] = (heuristic_evs[i] + 1.0) / 2.0;
                        }
                    } else if (config.use_duel_margin_maximization) {
                        heuristic_evs[i] = std::tanh(
                            heuristic_evs[i] / config.duel_margin_maximization_scale);
                    } else {
                        heuristic_evs[i] = std::max(0.0, std::min(1.0, heuristic_evs[i] / 1800.0));
                    }
                    game->solver_buffers.evs[i] =
                        (1.0 - config.heuristic_weight) * game->solver_buffers.raw_nn_evs[i] +
                        config.heuristic_weight * heuristic_evs[i];
                }
                game->solver_buffers.evs_blended = true;
                auto t_blend1 = debug_now(debug_stats);
                if (debug_stats) {
                    debug_stats->blend_us.fetch_add(elapsed_us(t_blend0, t_blend1),
                                                    std::memory_order_relaxed);
                }
            }

            SolverConfig active_cfg = config;
            active_cfg.compute_pre_roll_ev =
                (is_first_resolve && config.heuristic_weight <= 0.0001f);

            auto t_resolve0 = debug_now(debug_stats);
            SolverResult result = solver_resolve<Traits>(game->state, game->ctx, tables,
                                                        game->solver_buffers, active_cfg,
                                                        game->rng);
            auto t_resolve1 = debug_now(debug_stats);
            if (debug_stats) {
                debug_stats->solver_resolve_us.fetch_add(elapsed_us(t_resolve0, t_resolve1),
                                                         std::memory_order_relaxed);
            }

            if (result.is_exploratory) {
                game->current_turn_is_exploratory = true;
            }

            if (is_first_resolve && config.heuristic_weight <= 0.0001f) {
                game->current_turn_start_ev = result.pre_roll_ev;
            }

            if (config.debug_mode && game->is_debug_game && game->trajectory_length < 5) {
                std::ofstream dbg(game->debug_log_path, std::ios::app);
                dbg << "\n[Turn " << game->trajectory_length << " | P" << (int)game->state.board.current_player
                    << " | Rolls Left: " << (int)game->state.rolls_left << "]\n";

                dbg << "Dice: [ ";
                for (int d = 0; d < 5; ++d) dbg << (int)game->state.dice[d] << " ";
                dbg << "]\nTop 5 Evaluated Placements:\n";

                std::vector<std::pair<double, int>> ranked_reqs;
                for(int i = 0; i < game->solver_buffers.request_count; i++) {
                    auto req = game->solver_buffers.requests[i];
                    int calc = calculate_score<Traits>(req.placement.row, game->state.dice,
                                                      game->state.board.current_player,
                                                      req.placement.column, game->state.board, game->ctx);
                    if (req.score == calc) {
                        ranked_reqs.push_back({game->solver_buffers.evs[i], i});
                    }
                }
                std::sort(ranked_reqs.rbegin(), ranked_reqs.rend());

                for(int i = 0; i < std::min(5, (int)ranked_reqs.size()); i++) {
                    auto req = game->solver_buffers.requests[ranked_reqs[i].second];
                    dbg << "  Col: " << (int)req.placement.column << " Row: " << (int)req.placement.row
                        << " Score: " << (int)req.score
                        << " | NN EV: " << std::fixed << std::setprecision(4) << ranked_reqs[i].first << "\n";
                }

                if (game->state.rolls_left >= 1 && game->state.rolls_left <= 2) {
                    std::vector<std::pair<double, int>> ranked_masks;
                    for (int m = 0; m <= 32; ++m) {
                        ranked_masks.push_back({game->solver_buffers.mask_evs[m], m});
                    }
                    std::sort(ranked_masks.rbegin(), ranked_masks.rend());

                    dbg << "Top 3 Hold Masks:\n";
                    for (int m = 0; m < std::min(3, (int)ranked_masks.size()); ++m) {
                        int mask_id = ranked_masks[m].second;
                        double ev = ranked_masks[m].first;
                        if (mask_id == 32) {
                            dbg << "  Mask: STOP | EV: " << std::fixed << std::setprecision(4) << ev << "\n";
                        } else {
                            dbg << "  Mask: " << std::setw(2) << mask_id << "   | EV: " << std::fixed << std::setprecision(4) << ev << "\n";
                        }
                    }
                }

                if (result.should_place) {
                    dbg << ">>> ACTION: PLACED Score " << (int)result.score
                        << " at Col " << (int)result.placement.column << " Row " << (int)result.placement.row << "\n";
                } else {
                    dbg << ">>> ACTION: REROLL (Mask: " << (int)result.hold_mask << ")\n";
                }
            }

            if (result.should_place) {
                int    pbrs_p   = game->state.board.current_player;
                int    pbrs_col = result.placement.column;
                bool   upper_before = (game->ctx.upper_sum[pbrs_p][pbrs_col] >= 60);

                assert(game->trajectory_length < GI::kMaxTrajectorySteps);
                TrajectoryStepT<Traits>& step = game->trajectory[game->trajectory_length];

                step.value  = game->current_turn_start_ev;
                step.player = static_cast<int8_t>(game->state.board.current_player);
                step.is_exploratory = game->current_turn_is_exploratory;

                assert(result.chosen_request_idx >= 0);
                assert(result.chosen_request_idx < game->solver_buffers.request_count);
                auto t_chosen_tensor0 = debug_now(debug_stats);
                generate_tensor_batch<Traits>(
                    game->state.board, game->ctx, game->state.board.current_player,
                    &game->solver_buffers.requests[result.chosen_request_idx], 1,
                    tables,
                    step.tensor);
                auto t_chosen_tensor1 = debug_now(debug_stats);
                if (debug_stats) {
                    debug_stats->chosen_tensor_us.fetch_add(
                        elapsed_us(t_chosen_tensor0, t_chosen_tensor1),
                        std::memory_order_relaxed);
                }
                ++game->trajectory_length;

                auto t_action0 = debug_now(debug_stats);
                perform_placement<Traits>(game->state, game->ctx,
                                          result.placement.column, result.placement.row, game->rng);
                auto t_action1 = debug_now(debug_stats);
                if (debug_stats) {
                    debug_stats->perform_action_us.fetch_add(elapsed_us(t_action0, t_action1),
                                                             std::memory_order_relaxed);
                    debug_stats->worker_placements.fetch_add(1, std::memory_order_relaxed);
                }

                double pbrs = 0.0;
                if (config.use_pbrs) {
                    bool upper_after = (game->ctx.upper_sum[pbrs_p][pbrs_col] >= 60);
                    if (!upper_before && upper_after) {
                        pbrs += config.pbrs_upper_reward;
                    }
                    bool col_full = true;
                    for (int r = 0; r < kNumRows; ++r) {
                        if (game->state.board.cells[pbrs_p][pbrs_col][r] == -1) {
                            col_full = false;
                            break;
                        }
                    }
                    if (col_full && upper_after && !game->ctx.lower_has_scratch[pbrs_p][pbrs_col]) {
                        pbrs += config.pbrs_clean_reward;
                    }
                }
                step.pbrs_reward = pbrs;

                if (is_terminal<Traits>(game->state.board)) {
                    int duel = get_game_result<Traits>(game->state, game->ctx);
                    game->final_duel_margin = duel;
                    game->result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
                    game->phase  = GamePhase::kCompleted;
                    if (debug_stats) {
                        debug_stats->worker_completed_games.fetch_add(1, std::memory_order_relaxed);
                    }
                    completed.push(game);
                } else {
                    game->phase = GamePhase::kNeedRequests;
                    available.push(game);
                }

            } else {
                auto t_action0 = debug_now(debug_stats);
                perform_reroll<Traits>(game->state, result.hold_mask, game->rng);
                auto t_action1 = debug_now(debug_stats);
                if (debug_stats) {
                    debug_stats->perform_action_us.fetch_add(elapsed_us(t_action0, t_action1),
                                                             std::memory_order_relaxed);
                    debug_stats->worker_rerolls.fetch_add(1, std::memory_order_relaxed);
                }
                game->phase = GamePhase::kNeedResolve;
                available.push(game);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template void worker_thread<Yams1v1>(GameQueueT<Yams1v1>&, BatchManagerT<Yams1v1>&,
                                     BatchManagerT<Yams1v1>*, GameQueueT<Yams1v1>&,
                                     const PrecomputedTables&, const SolverConfig&,
                                     std::atomic<bool>&, SelfPlayDebugStats*);
template void worker_thread<Yams2v2>(GameQueueT<Yams2v2>&, BatchManagerT<Yams2v2>&,
                                     BatchManagerT<Yams2v2>*, GameQueueT<Yams2v2>&,
                                     const PrecomputedTables&, const SolverConfig&,
                                     std::atomic<bool>&, SelfPlayDebugStats*);
