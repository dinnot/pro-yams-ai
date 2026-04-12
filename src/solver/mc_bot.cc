#include "solver/mc_bot.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "engine/context_rebuild.h"
#include "engine/duel.h"
#include "engine/placement.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "eval/evaluator.h"
#include "heuristic/heuristic_bot.h"

// ---------------------------------------------------------------------------
// simulate_rollout
// ---------------------------------------------------------------------------

double simulate_rollout(const BoardState& board,
                        int mc_player,
                        ProYamsNet& model, torch::Device device,
                        const PrecomputedTables& tables,
                        RNG& rng) {
    // Set up a fresh game state from the afterstate.
    GameState sim_state;
    sim_state.board = board;

    // Roll dice for the next player's turn.
    start_turn(sim_state, rng);

    // Rebuild context from the cloned board.
    GameContext sim_ctx;
    rebuild_context_from_board(sim_state.board, sim_ctx);

    SolverBuffers buffers{};
    SolverConfig greedy{0.0, 0.0, false};

    std::vector<float> tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);

    // Play to completion.
    while (!is_terminal(sim_state.board)) {
        nn_play_turn(model, device, sim_state, sim_ctx, tables,
                     buffers, tensor_buffer, greedy, rng);
    }

    return get_game_result_for_player(sim_state, sim_ctx, mc_player);
}

// ---------------------------------------------------------------------------
// simulate_rollout_from_state
// ---------------------------------------------------------------------------

// Like simulate_rollout, but the game state already has dice rolled and
// rolls_left set (mid-turn). Does NOT call start_turn — used for hold
// candidates whose reroll has already been applied.
static double simulate_rollout_from_state(GameState sim_state,
                                           GameContext sim_ctx,
                                           int mc_player,
                                           ProYamsNet& model, torch::Device device,
                                           const PrecomputedTables& tables,
                                           RNG& rng) {
    SolverBuffers buffers{};
    SolverConfig greedy{0.0, 0.0, false};
    std::vector<float> tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);

    while (!is_terminal(sim_state.board)) {
        nn_play_turn(model, device, sim_state, sim_ctx, tables,
                     buffers, tensor_buffer, greedy, rng);
    }

    return get_game_result_for_player(sim_state, sim_ctx, mc_player);
}

// ---------------------------------------------------------------------------
// select_top_candidates
// ---------------------------------------------------------------------------

int select_top_candidates(const GameState& state, const GameContext& ctx,
                          const PrecomputedTables& tables,
                          const SolverBuffers& buffers,
                          int max_k, CandidateAction* out) {
    int count = 0;
    int current_id = get_dice_state_id(state.dice, tables);
    int rolls_left = state.rolls_left;

    // --- Placement candidates ---
    // Collect all achievable placements for the current dice state, ranked by EV.
    struct PlacementEV {
        int request_idx;
        double ev;
    };
    std::vector<PlacementEV> place_candidates;
    place_candidates.reserve(buffers.request_count);

    for (int i = 0; i < buffers.request_count; ++i) {
        const AfterstateRequest& req = buffers.requests[i];
        bool achievable = (req.score == 0) ||
            (tables.score_tables.dice_score[current_id][req.placement.row] == req.score);
        if (achievable) {
            place_candidates.push_back({i, buffers.evs[i]});
        }
    }

    // Sort by EV descending.
    std::sort(place_candidates.begin(), place_candidates.end(),
              [](const PlacementEV& a, const PlacementEV& b) {
                  return a.ev > b.ev;
              });

    // Add top placement candidates.
    int place_slots = std::min(max_k, static_cast<int>(place_candidates.size()));
    // If we have rerolls, reserve some slots for hold masks.
    if (rolls_left > 0 && place_slots > 2) place_slots = 2;

    for (int i = 0; i < place_slots && count < max_k; ++i) {
        const auto& pc = place_candidates[i];
        const auto& req = buffers.requests[pc.request_idx];
        out[count].is_placement = true;
        out[count].placement = req.placement;
        out[count].score = req.score;
        out[count].hold_mask = 0;
        out[count].nn_ev = pc.ev;
        out[count].win_count = 0;
        out[count].rollout_count = 0;
        count++;
    }

    // --- Hold mask candidates (if rerolls remain) ---
    if (rolls_left > 0 && can_reroll(state, ctx)) {
        struct MaskEV {
            uint8_t mask;
            double ev;
        };
        MaskEV mask_evs[kNumHoldMasks];

        // Use V0 or V1 depending on rolls_left.
        for (int mask = 0; mask < kNumHoldMasks; ++mask) {
            int held_id = tables.moves[current_id][mask];
            int tr_count = 0;
            const Transition* tr = get_transitions(held_id, tr_count, tables);
            double ev = 0.0;
            if (rolls_left == 1) {
                // Transitions go into V0.
                for (int t = 0; t < tr_count; ++t)
                    ev += tr[t].probability * buffers.v0[tr[t].target_state_id];
            } else {
                // rolls_left == 2: transitions go into V1.
                for (int t = 0; t < tr_count; ++t)
                    ev += tr[t].probability * buffers.v1[tr[t].target_state_id];
            }
            mask_evs[mask] = {static_cast<uint8_t>(mask), ev};
        }

        // Sort by EV descending.
        std::sort(mask_evs, mask_evs + kNumHoldMasks,
                  [](const MaskEV& a, const MaskEV& b) {
                      return a.ev > b.ev;
                  });

        // Add top hold mask candidates (that are better than worst placement).
        for (int i = 0; i < kNumHoldMasks && count < max_k; ++i) {
            out[count].is_placement = false;
            out[count].placement = {};
            out[count].score = 0;
            out[count].hold_mask = mask_evs[i].mask;
            out[count].nn_ev = mask_evs[i].ev;
            out[count].win_count = 0;
            out[count].rollout_count = 0;
            count++;
        }
    }

    return count;
}

// ---------------------------------------------------------------------------
// can_terminate_early — Wilson score 95% CI
// ---------------------------------------------------------------------------

static double wilson_lower(double wins, int n) {
    // Wilson score confidence interval lower bound (z = 1.96 for 95%).
    if (n == 0) return 0.0;
    constexpr double z = 1.96;
    double p_hat = wins / n;
    double denom = 1.0 + z * z / n;
    double center = p_hat + z * z / (2.0 * n);
    double spread = z * std::sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n);
    return (center - spread) / denom;
}

static double wilson_upper(double wins, int n) {
    if (n == 0) return 1.0;
    constexpr double z = 1.96;
    double p_hat = wins / n;
    double denom = 1.0 + z * z / n;
    double center = p_hat + z * z / (2.0 * n);
    double spread = z * std::sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n);
    return (center + spread) / denom;
}

bool can_terminate_early(const CandidateAction* candidates, int count,
                         int min_rollouts) {
    if (count <= 1) return true;

    // Find the best candidate by empirical win rate.
    int best_idx = 0;
    double best_wr = candidates[0].empirical_wr();
    for (int i = 1; i < count; ++i) {
        double wr = candidates[i].empirical_wr();
        if (wr > best_wr) {
            best_wr = wr;
            best_idx = i;
        }
    }

    // Need at least min_rollouts for all candidates.
    for (int i = 0; i < count; ++i) {
        if (candidates[i].rollout_count < min_rollouts) return false;
    }

    double best_lower = wilson_lower(candidates[best_idx].win_count,
                                      candidates[best_idx].rollout_count);

    // Check if the best's lower bound exceeds all other upper bounds.
    for (int i = 0; i < count; ++i) {
        if (i == best_idx) continue;
        double other_upper = wilson_upper(candidates[i].win_count,
                                           candidates[i].rollout_count);
        if (best_lower <= other_upper) return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// mc_play_turn
// ---------------------------------------------------------------------------

void mc_play_turn(GameState& state, GameContext& ctx,
                  ProYamsNet& model, torch::Device device,
                  const PrecomputedTables& tables,
                  const MCConfig& mc_config, RNG& rng) {
    MCTimeAllocation time_alloc{mc_config.time_budget_ms, 0.0};
    SolverBuffers buffers{};
    SolverConfig greedy{0.0, 0.0, false};

    std::vector<float> tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);

    while (true) {
        using Clock = std::chrono::high_resolution_clock;
        auto decision_start = Clock::now();
        double decision_budget = time_alloc.budget_for_current_decision(
            state.rolls_left);

        // Step 1: Get afterstate requests and NN evaluations.
        solver_get_requests(state, ctx, tables, buffers);

        // Handle zero-request edge case (forced Turbo reroll).
        if (buffers.request_count == 0) {
            if (!can_reroll(state, ctx)) {
                // Deadlock: no legal moves and cannot reroll.
                // This shouldn't happen with correct rules, but prevent HANG.
                return;
            }
            perform_reroll(state, 0, rng);
            auto now = Clock::now();
            time_alloc.elapsed_ms += std::chrono::duration<double, std::milli>(
                now - decision_start).count();
            continue;
        }

        // Generate tensors and run NN inference.
        generate_tensor_batch(state.board, ctx,
                              static_cast<int>(state.board.current_player),
                              buffers.requests, buffers.request_count,
                              tables, tensor_buffer.data());
        {
            torch::NoGradGuard no_grad;
            auto input = torch::from_blob(
                tensor_buffer.data(),
                {buffers.request_count, kTensorSize},
                torch::kFloat32).to(device);
            auto output = model.forward(input).to(torch::kCPU).contiguous();
            const float* ptr = output.data_ptr<float>();
            for (int i = 0; i < buffers.request_count; ++i)
                buffers.evs[i] = static_cast<double>(ptr[i]);
        }

        // Build V0/V1 via the solver (needed for candidate selection).
        // We call solver_resolve to populate v0/v1 tables, then extract candidates.
        solver_resolve(state, ctx, tables, buffers, greedy, rng);

        // Step 2: Select top-K candidates.
        CandidateAction candidates[16];  // max_candidates capped at 16
        int num_candidates = select_top_candidates(
            state, ctx, tables, buffers,
            std::min(mc_config.max_candidates, 16), candidates);

        if (num_candidates == 0) {
            // Fallback: use solver result directly.
            SolverResult result = solver_resolve_greedy(state, ctx, tables, buffers);
            if (result.should_place) {
                perform_placement(state, ctx,
                                  result.placement.column, result.placement.row, rng);
                return;
            }
            if (!can_reroll(state, ctx)) {
                // Safety break to prevent infinite loop
                return;
            }
            perform_reroll(state, result.hold_mask, rng);
            auto now = Clock::now();
            time_alloc.elapsed_ms += std::chrono::duration<double, std::milli>(
                now - decision_start).count();
            continue;
        }

        // If only one candidate, skip rollouts.
        if (num_candidates == 1) {
            auto& best = candidates[0];
            if (best.is_placement) {
                perform_placement(state, ctx,
                                  best.placement.column, best.placement.row, rng);
                return;
            }
            if (!can_reroll(state, ctx)) {
                // Should have been prevented by select_top_candidates, but safety first.
                return;
            }
            perform_reroll(state, best.hold_mask, rng);
            auto now = Clock::now();
            time_alloc.elapsed_ms += std::chrono::duration<double, std::milli>(
                now - decision_start).count();
            continue;
        }

        // Step 3: Rollout evaluation.
        int mc_player = static_cast<int>(state.board.current_player);
        double elapsed_decision = 0.0;

        while (elapsed_decision < decision_budget) {
            for (int ci = 0; ci < num_candidates; ++ci) {
                auto& cand = candidates[ci];
                for (int b = 0; b < mc_config.rollout_batch_size; ++b) {
                    RNG rollout_rng(rng.next());
                    double outcome;

                    if (cand.is_placement) {
                        // Apply the placement to get the afterstate, then
                        // simulate from the resulting board (start_turn will
                        // roll dice for the next player inside simulate_rollout).
                        BoardState rollout_board = state.board;
                        GameContext tmp_ctx;
                        rebuild_context_from_board(rollout_board, tmp_ctx);
                        apply_placement(mc_player,
                                        cand.placement.column,
                                        cand.placement.row,
                                        cand.score,
                                        rollout_board, tmp_ctx);
                        rollout_board.current_player =
                            1 - rollout_board.current_player;
                        outcome = simulate_rollout(
                            rollout_board, mc_player,
                            model, device, tables, rollout_rng);
                    } else {
                        // Hold and reroll: apply the reroll to get the new
                        // dice state, then continue the turn from that state
                        // without calling start_turn (dice are already set).
                        GameState tmp_state;
                        tmp_state.board = state.board;
                        std::memcpy(tmp_state.dice, state.dice,
                                    sizeof(state.dice));
                        tmp_state.rolls_left = state.rolls_left;
                        RNG hold_rng(rollout_rng.next());
                        perform_reroll(tmp_state, cand.hold_mask, hold_rng);
                        GameContext tmp_ctx;
                        rebuild_context_from_board(tmp_state.board, tmp_ctx);
                        outcome = simulate_rollout_from_state(
                            tmp_state, tmp_ctx, mc_player,
                            model, device, tables, rollout_rng);
                    }

                    cand.win_count += outcome;
                    cand.rollout_count++;
                }
            }

            auto now = Clock::now();
            elapsed_decision = std::chrono::duration<double, std::milli>(
                now - decision_start).count();

            // Check early termination.
            if (can_terminate_early(candidates, num_candidates,
                                    mc_config.min_rollouts_per_candidate))
                break;
        }

        // Step 4: Pick the best candidate by empirical win rate.
        int best_idx = 0;
        double best_wr = candidates[0].empirical_wr();
        for (int i = 1; i < num_candidates; ++i) {
            double wr = candidates[i].empirical_wr();
            if (wr > best_wr) {
                best_wr = wr;
                best_idx = i;
            }
        }

        // Update time tracking.
        auto now = Clock::now();
        time_alloc.elapsed_ms += std::chrono::duration<double, std::milli>(
            now - decision_start).count();

        // Step 5: Apply the best action.
        const auto& best = candidates[best_idx];
        if (best.is_placement) {
            perform_placement(state, ctx,
                              best.placement.column, best.placement.row, rng);
            return;
        }

        // Hold and reroll — continue to next decision.
        assert(can_reroll(state, ctx));
        perform_reroll(state, best.hold_mask, rng);
    }
}
