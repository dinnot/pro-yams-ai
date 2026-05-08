#include "self_play/worker.h"

#include <cassert>
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

void worker_thread(GameQueue& available, BatchManager& batch_manager,
                   BatchManager* opponent_batch_manager,
                   GameQueue& completed,
                   const PrecomputedTables& tables, const SolverConfig& config,
                   std::atomic<bool>& shutdown) {
    (void)shutdown;  // checked via nullptr sentinel from available queue

    while (true) {
        GameInstance* game = available.pop();
        if (game == nullptr) break;  // shutdown sentinel

        if (game->phase == GamePhase::kNeedRequests) {
            // -----------------------------------------------------------
            // Step 1: generate afterstate requests.
            // -----------------------------------------------------------
            game->solver_buffers.dp_computed = false;
            game->solver_buffers.evs_blended = false;

            solver_get_requests(game->state, game->ctx, tables,
                                game->solver_buffers);

            if (game->solver_buffers.request_count == 0) {
                // No placements available. With corrected Turbo logic this
                // should be very rare. Guard against deadlock at rolls_left==0.
                if (game->state.rolls_left <= 0) {
                    // Stuck: no legal placements and no rerolls left.
                    // Force-place in any legal_all cell (Turbo) as a scratch.
                    const auto& all = game->ctx.legal_all[game->state.board.current_player];
                    if (all.count > 0) {
                        perform_placement(game->state, game->ctx,
                                          all.placements[0].column,
                                          all.placements[0].row, game->rng);
                        if (is_terminal(game->state.board)) {
                            int duel = get_game_result(game->state, game->ctx);
                            game->final_duel_margin = duel;
                            game->result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
                            game->phase = GamePhase::kCompleted;
                            completed.push(game);
                        } else {
                            available.push(game);
                        }
                    } else {
                        // Truly stuck — should never happen.
                        game->phase = GamePhase::kCompleted;
                        game->result = 0.5;
                        completed.push(game);
                    }
                } else {
                    perform_reroll(game->state, 0, game->rng);
                    available.push(game);
                }
                continue;
            }

            assert(game->solver_buffers.request_count <= kMaxAfterstateRequests);

            // -----------------------------------------------------------
            // Step 2: generate tensors directly into the shared batch buffer.
            // Past-opponent games route the opponent seat's requests to the
            // secondary batch manager (whose coordinator runs the older model).
            // -----------------------------------------------------------
            BatchManager* target_bm = &batch_manager;
            if (opponent_batch_manager != nullptr &&
                game->use_past_opponent &&
                game->state.board.current_player == game->past_opponent_player) {
                target_bm = opponent_batch_manager;
            }

            InferenceBatch* reserved_batch = nullptr;
            int batch_offset = 0;
            float* dest_ptr = target_bm->reserve(
                game->solver_buffers.request_count, reserved_batch, batch_offset);

            if (dest_ptr == nullptr) {
                // Reserve only returns null on BatchManager shutdown.
                break;
            }

            generate_tensor_batch(
                game->state.board, game->ctx,
                game->state.board.current_player,
                game->solver_buffers.requests,
                game->solver_buffers.request_count,
                tables,
                dest_ptr);

            game->phase = GamePhase::kWaitingInference;
            target_bm->commit(reserved_batch, game, batch_offset,
                              game->solver_buffers.request_count);

        } else if (game->phase == GamePhase::kNeedResolve) {
            // Capture if this is the first resolve of the turn BEFORE we run solver_resolve
            bool is_first_resolve = !game->solver_buffers.dp_computed;

            if (is_first_resolve) {
                game->current_turn_is_exploratory = false;
            }

            // ------------------------------------------------------------------
            // FIX: Run solver with PURE NN EVs to get the correct start-of-turn
            // EV for TD learning. This prevents the heuristic from poisoning targets!
            // ------------------------------------------------------------------
            if (is_first_resolve && config.heuristic_weight > 0.0) {
                SolverResult pure_res = solver_resolve_greedy(game->state, game->ctx, tables, game->solver_buffers, true);
                game->current_turn_start_ev = pure_res.pre_roll_ev;
                game->solver_buffers.dp_computed = false; // Reset so the blended resolve can run
            }

            if (config.heuristic_weight > 0.0 && !game->solver_buffers.evs_blended) {
                // Save raw NN EVs for debugging
                std::memcpy(game->solver_buffers.raw_nn_evs, game->solver_buffers.evs,
                            game->solver_buffers.request_count * sizeof(double));

                double heuristic_evs[kMaxAfterstateRequests];
                if (config.heuristic_version == 2) {
                    heuristic_evaluate_v2(game->state.board, game->ctx,
                                          game->solver_buffers.requests,
                                          game->solver_buffers.request_count,
                                          heuristic_evs, tables);
                } else {
                    heuristic_evaluate(game->state.board, game->ctx,
                                       game->solver_buffers.requests,
                                       game->solver_buffers.request_count,
                                       heuristic_evs);
                }

                // Normalize heuristic EVs to match the NN's output range, then
                // blend. V2 emits raw expected duel margin (can be negative,
                // ~±5000); V1 emits non-negative score×coeff (~0..1800).
                int n = game->solver_buffers.request_count;
                for (int i = 0; i < n; i++) {
                    if (config.heuristic_version == 2) {
                        // V2 → squash actual duel margin via tanh, matching
                        // duel_margin_maximization NN targets in [-1, 1].
                        heuristic_evs[i] = std::tanh(
                            heuristic_evs[i] / config.duel_margin_maximization_scale);
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
                game->solver_buffers.evs_blended = true; // Mark as blended!
            }

            SolverConfig active_cfg = config;
            // Only compute V2 if we didn't already get it from the pure pass
            if (is_first_resolve && config.heuristic_weight <= 0.0) {
                active_cfg.compute_pre_roll_ev = true;
            } else {
                active_cfg.compute_pre_roll_ev = false;
            }

            SolverResult result = solver_resolve(game->state, game->ctx, tables,
                                                  game->solver_buffers, active_cfg,
                                                  game->rng);

            if (result.is_exploratory) {
                game->current_turn_is_exploratory = true;
            }

            // FIX: Only capture from the main result if the heuristic is turned OFF.
            if (is_first_resolve && config.heuristic_weight <= 0.0) {
                game->current_turn_start_ev = result.pre_roll_ev;
            }

            // Log Game 0, limit to first 5 turns of the game to avoid spam
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
                    int calc = calculate_score(req.placement.row, game->state.dice, game->state.board.current_player, req.placement.column, game->state.board, game->ctx);
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
                            // To print nicely formatted, could use std::setw but let's keep it simple.
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
                // Snapshot pre-placement state for PBRS detection.
                int    pbrs_p   = game->state.board.current_player;
                int    pbrs_col = result.placement.column;
                bool   upper_before = (game->ctx.upper_sum[pbrs_p][pbrs_col] >= 60);

                // Record trajectory step before applying placement.
                assert(game->trajectory_length < GameInstance::kMaxTrajectorySteps);
                TrajectoryStep& step = game->trajectory[game->trajectory_length];

                // Use the Q-value from the START of the turn, avoiding dice degradation!
                step.value  = game->current_turn_start_ev;
                step.player = static_cast<int8_t>(game->state.board.current_player);
                step.is_exploratory = game->current_turn_is_exploratory;

                // The shared batch buffer the worker wrote into during
                // kNeedRequests has been recycled by now, so regenerate the
                // single chosen tensor (~1.2 µs) instead of memcpying.
                assert(result.chosen_request_idx >= 0);
                assert(result.chosen_request_idx < game->solver_buffers.request_count);
                generate_tensor_batch(
                    game->state.board, game->ctx, game->state.board.current_player,
                    &game->solver_buffers.requests[result.chosen_request_idx], 1,
                    tables,
                    step.tensor);
                ++game->trajectory_length;

                // Apply placement (switches player, rolls dice for next turn).
                perform_placement(game->state, game->ctx,
                                  result.placement.column, result.placement.row, game->rng);

                // Compute PBRS bonus for milestones achieved exactly on this turn.
                double pbrs = 0.0;
                if (config.use_pbrs) {
                    bool upper_after = (game->ctx.upper_sum[pbrs_p][pbrs_col] >= 60);
                    // 1. Upper Section Bonus unlocked
                    if (!upper_before && upper_after) {
                        pbrs += config.pbrs_upper_reward;
                    }
                    // 2. Clean Column unlocked
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

                if (is_terminal(game->state.board)) {
                    int duel = get_game_result(game->state, game->ctx);
                    game->final_duel_margin = duel;
                    game->result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
                    game->phase  = GamePhase::kCompleted;
                    completed.push(game);
                } else {
                    game->phase = GamePhase::kNeedRequests;
                    available.push(game);
                }

            } else {
                // Hold and reroll — same player's turn continues.
                // Because tensors only depend on board state (which hasn't changed),
                // we skip tensor generation/inference entirely for the rest of the turn!
                perform_reroll(game->state, result.hold_mask, game->rng);
                game->phase = GamePhase::kNeedResolve;
                available.push(game);
            }
        }
        // Unknown phase: ignore (shouldn't happen in well-formed usage).
    }
}
