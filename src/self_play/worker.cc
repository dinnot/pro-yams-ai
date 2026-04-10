#include "self_play/worker.h"

#include <cassert>
#include <cstring>

#include "self_play/game_instance.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"
#include "engine/tensor.h"

void worker_thread(GameQueue& available, GameQueue& pending, GameQueue& completed,
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
                                          all.placements[0].row, 0, game->rng);
                        if (is_terminal(game->state.board)) {
                            int duel = get_game_result(game->state, game->ctx);
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

            assert(game->solver_buffers.request_count <= GameInstance::kMaxAfterstates);

            // -----------------------------------------------------------
            // Step 2: generate tensors for all requests.
            // -----------------------------------------------------------
            generate_tensor_batch(
                game->state.board, game->ctx,
                game->state.board.current_player,
                game->solver_buffers.requests,
                game->solver_buffers.request_count,
                tables,
                game->tensor_buffer);

            game->phase = GamePhase::kWaitingInference;
            pending.push(game);

        } else if (game->phase == GamePhase::kNeedResolve) {
            // -----------------------------------------------------------
            // Step 3: solver resolution with pre-populated EVs.
            // -----------------------------------------------------------
            SolverResult result = solver_resolve(game->state, game->ctx, tables,
                                                  game->solver_buffers, config,
                                                  game->rng);

            // Capture the pure optimal EV at the very first decision point of the turn.
            if (game->state.rolls_left == 2) {
                game->current_turn_start_ev = result.expected_value;
            }

            if (result.should_place) {
                // Record trajectory step before applying placement.
                assert(game->trajectory_length < GameInstance::kMaxTrajectorySteps);
                TrajectoryStep& step = game->trajectory[game->trajectory_length];
                
                // Use the Q-value from the START of the turn, avoiding dice degradation!
                step.value  = game->current_turn_start_ev;
                step.player = static_cast<int8_t>(game->state.board.current_player);

                // Copy the chosen afterstate tensor from tensor_buffer.
                assert(result.chosen_request_idx >= 0);
                assert(result.chosen_request_idx < game->solver_buffers.request_count);
                std::memcpy(step.tensor,
                            game->tensor_buffer
                                + static_cast<size_t>(result.chosen_request_idx) * kTensorSize,
                            kTensorSize * sizeof(float));
                ++game->trajectory_length;

                // Apply placement (switches player, rolls dice for next turn).
                perform_placement(game->state, game->ctx,
                                  result.placement.column, result.placement.row,
                                  result.score, game->rng);

                if (is_terminal(game->state.board)) {
                    int duel = get_game_result(game->state, game->ctx);
                    game->result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
                    game->phase  = GamePhase::kCompleted;
                    completed.push(game);
                } else {
                    game->phase = GamePhase::kNeedRequests;
                    available.push(game);
                }

            } else {
                // Hold and reroll — same player's turn continues.
                perform_reroll(game->state, result.hold_mask, game->rng);
                game->phase = GamePhase::kNeedRequests;
                available.push(game);
            }
        }
        // Unknown phase: ignore (shouldn't happen in well-formed usage).
    }
}
