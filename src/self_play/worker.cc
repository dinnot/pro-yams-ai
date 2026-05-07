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

void worker_thread(GameQueue& available, BatchManager& batch_manager, GameQueue& completed,
                   const PrecomputedTables& tables, const SolverConfig& config,
                   std::atomic<bool>& shutdown) {
    (void)shutdown;  // checked via nullptr sentinel from available queue
    
    double total_pop_wait_ms = 0.0;
    double total_compute_ms = 0.0;

    while (true) {
        auto t_pop_start = std::chrono::steady_clock::now();
        GameInstance* game = available.pop();
        auto t_pop_end = std::chrono::steady_clock::now();
        double pop_ms = std::chrono::duration<double, std::milli>(t_pop_end - t_pop_start).count();
        total_pop_wait_ms += pop_ms;
        
        if (game == nullptr) break;  // shutdown sentinel

        auto t_compute_start = std::chrono::steady_clock::now();

        if (game->phase == GamePhase::kNeedRequests) {
            // -----------------------------------------------------------
            // Step 1: generate afterstate requests.
            // -----------------------------------------------------------
            game->solver_buffers.dp_computed = false;
            game->solver_buffers.evs_blended = false;
            
            auto t_solver_start = std::chrono::steady_clock::now();
            solver_get_requests(game->state, game->ctx, tables,
                                game->solver_buffers);
            auto t_solver_end = std::chrono::steady_clock::now();
            double solver_ms = std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();

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
            // Step 2: generate tensors for all requests.
            // -----------------------------------------------------------
            auto t_tensor_start = std::chrono::steady_clock::now();
            
            InferenceBatch* reserved_batch = nullptr;
            int batch_offset = 0;
            float* dest_ptr = batch_manager.reserve(game->solver_buffers.request_count, reserved_batch, batch_offset);
            
            if (dest_ptr == nullptr) {
                // Should only happen on shutdown
                break;
            }

            generate_tensor_batch(
                game->state.board, game->ctx,
                game->state.board.current_player,
                game->solver_buffers.requests,
                game->solver_buffers.request_count,
                tables,
                dest_ptr);
                
            // Store the pointer and offset temporarily in solver_buffers so resolve can access it
            // for the trajectory! Wait, we can't easily add pointers to solver_buffers without modifying it.
            // But we need to save the `dest_ptr` or at least the generated tensors so we can copy
            // the chosen one to the trajectory later!
            // Wait, if we use Zero-Copy, `trajectory` recording in `kNeedResolve` needs to copy FROM the batch!
            // But by the time we are in `kNeedResolve`, the batch has been sent to GPU and recycled!
            // The memory might have been overwritten!
            // Oh no!
            // We MUST copy the chosen tensor BEFORE the batch is recycled, OR we must save the generated
            // tensors locally if we want to record them!
            // But wait, the Worker only knows WHICH tensor is chosen AFTER `solver_resolve`!
            // And `solver_resolve` requires the GPU EV predictions!
            // So the GPU inference MUST happen BEFORE we know which tensor to save!
            // Can we just regenerate the single chosen tensor during `kNeedResolve`?!
            // YES! `generate_tensor` takes microseconds for a single tensor!
            
            auto t_tensor_end = std::chrono::steady_clock::now();
            double tensor_ms = std::chrono::duration<double, std::milli>(t_tensor_end - t_tensor_start).count();

            game->phase = GamePhase::kWaitingInference;
            
            auto t_push_start = std::chrono::steady_clock::now();
            batch_manager.commit(reserved_batch, game, batch_offset, game->solver_buffers.request_count);
            auto t_push_end = std::chrono::steady_clock::now();
            double push_ms = std::chrono::duration<double, std::milli>(t_push_end - t_push_start).count();
            
            if (config.debug_mode) {
                static std::atomic<int> log_counter{0};
                if (++log_counter % 10000 == 0) {
                    std::ofstream f(config.debug_log_path + "_worker.log", std::ios::app);
                    if (f.is_open()) {
                        f << "PopWait: " << pop_ms << " ms | Solver: " << solver_ms 
                          << " ms | Tensor: " << tensor_ms << " ms | PushWait: " << push_ms << " ms\n";
                    }
                }
            }

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
                heuristic_evaluate(game->state.board, game->ctx,
                                   game->solver_buffers.requests,
                                   game->solver_buffers.request_count,
                                   heuristic_evs);

                // Normalize heuristic EVs to [0, 1] using a fixed global maximum (1800.0).
                // This preserves absolute scale, which is critical for TD learning. 
                // Dynamic normalization (min/max of current list) would map every "best"
                // move to 1.0, poisoning the value trajectory and destroying the signal.
                int n = game->solver_buffers.request_count;
                for (int i = 0; i < n; i++) {
                    if (config.use_duel_margin_maximization) {
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

                // Copy the chosen afterstate tensor.
                // Because we use Zero-Copy batching, the original tensor memory in the BatchManager
                // might have already been recycled and overwritten.
                // We regenerate the chosen tensor here. It's extremely fast for a single tensor.
                assert(result.chosen_request_idx >= 0);
                assert(result.chosen_request_idx < game->solver_buffers.request_count);
                generate_tensor_batch(
                    game->state.board, game->ctx, game->state.board.current_player,
                    &game->solver_buffers.requests[result.chosen_request_idx], 1,
                    tables,
                    step.tensor
                );
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
        
        auto t_compute_end = std::chrono::steady_clock::now();
        total_compute_ms += std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
    }
    
    std::ofstream f(config.debug_log_path + "_worker_totals.log", std::ios::app);
    if (f.is_open()) {
        f << "Worker Shutdown | Total Pop Wait: " << total_pop_wait_ms 
          << " ms | Total Compute: " << total_compute_ms << " ms\n";
    }
}
