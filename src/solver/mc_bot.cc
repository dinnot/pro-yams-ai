#include "solver/mc_bot.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

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
                        const GameContext& ctx,
                        int mc_player,
                        ProYamsNet& model, torch::Device device,
                        const PrecomputedTables& tables,
                        SolverBuffers& shared_buffers,
                        std::vector<float>& shared_tensor_buffer,
                        RNG& rng) {
    // Set up a fresh game state from the afterstate. The caller supplies a
    // matching GameContext, so we avoid an O(156) rebuild_context_from_board
    // scan on every rollout.
    GameState sim_state;
    sim_state.board = board;

    // Roll dice for the next player's turn.
    start_turn(sim_state, rng);

    GameContext sim_ctx = ctx;

    SolverConfig greedy{0.0, 0.0, false};

    // Reset solver cache state for the hoisted buffers.
    shared_buffers.dp_computed = false;
    shared_buffers.evs_blended = false;

    while (!is_terminal(sim_state.board)) {
        nn_play_turn(model, device, sim_state, sim_ctx, tables,
                     shared_buffers, shared_tensor_buffer, greedy, rng);
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
                                           SolverBuffers& shared_buffers,
                                           std::vector<float>& shared_tensor_buffer,
                                           RNG& rng) {
    SolverConfig greedy{0.0, 0.0, false};

    shared_buffers.dp_computed = false;
    shared_buffers.evs_blended = false;

    while (!is_terminal(sim_state.board)) {
        nn_play_turn(model, device, sim_state, sim_ctx, tables,
                     shared_buffers, shared_tensor_buffer, greedy, rng);
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
    int current_id = get_dice_state_id(state.dice, tables);
    int rolls_left = state.rolls_left;
    const int p = state.board.current_player;

    // Strict achievable check — mirrors solver_resolve's check_achievable
    // exactly so that candidate placements respect the Golden Rule, SS/LS
    // constraints, and the Turbo-on-last-roll exclusion. Without this, the MC
    // bot explores illegal timelines (notably voluntary scratches), making
    // its rollouts disconnected from reality.
    auto check_achievable = [&](int sid, int req_idx) -> bool {
        const AfterstateRequest& req = buffers.requests[req_idx];
        int col = req.placement.column;
        int row = req.placement.row;

        // Turbo requires at least one reroll remaining.
        if (rolls_left == 0 && col == kColTurbo) return false;

        int raw = tables.score_tables.dice_score[sid][row];
        bool is_valid = false;

        if (raw > 0 && raw >= ctx.golden_max[col][row]) {
            is_valid = true;
            if (row == kRowSS) {
                if (ctx.ls_scratched[p][col]) is_valid = false;
                else {
                    int8_t ls_val = state.board.cells[p][col][kRowLS];
                    if (ls_val != kCellEmpty && ls_val != 0 && raw >= ls_val)
                        is_valid = false;
                }
            } else if (row == kRowLS) {
                if (ctx.ss_scratched[p][col]) is_valid = false;
                else {
                    int max_ss = ctx.golden_max[col][kRowSS];
                    if (max_ss > 0 && raw <= max_ss) is_valid = false;
                }
            }
        }
        return is_valid ? (req.score == raw) : (req.score == 0);
    };

    // Collect all candidates (placements + holds) into a single list so we
    // can globally sort them by NN EV. Previously placements were written
    // first and holds appended unsorted, which meant that if the time budget
    // was exhausted before any rollouts, the bot would default to the first
    // placement rather than the globally best NN action.
    std::vector<CandidateAction> all_cands;
    all_cands.reserve(buffers.request_count + kNumHoldMasks);

    for (int i = 0; i < buffers.request_count; ++i) {
        if (!check_achievable(current_id, i)) continue;
        const AfterstateRequest& req = buffers.requests[i];
        CandidateAction c{};
        c.is_placement = true;
        c.placement = req.placement;
        c.score = req.score;
        c.hold_mask = 0;
        c.nn_ev = buffers.evs[i];
        c.win_count = 0;
        c.rollout_count = 0;
        all_cands.push_back(c);
    }

    if (rolls_left > 0 && can_reroll(state, ctx)) {
        for (int mask = 0; mask < kNumHoldMasks; ++mask) {
            int held_id = tables.moves[current_id][mask];
            int tr_count = 0;
            const Transition* tr = get_transitions(held_id, tr_count, tables);
            double ev = 0.0;
            if (rolls_left == 1) {
                for (int t = 0; t < tr_count; ++t)
                    ev += tr[t].probability * buffers.v0[tr[t].target_state_id];
            } else {
                for (int t = 0; t < tr_count; ++t)
                    ev += tr[t].probability * buffers.v1[tr[t].target_state_id];
            }
            CandidateAction c{};
            c.is_placement = false;
            c.placement = {};
            c.score = 0;
            c.hold_mask = static_cast<uint8_t>(mask);
            c.nn_ev = ev;
            c.win_count = 0;
            c.rollout_count = 0;
            all_cands.push_back(c);
        }
    }

    std::sort(all_cands.begin(), all_cands.end(),
              [](const CandidateAction& a, const CandidateAction& b) {
                  return a.nn_ev > b.nn_ev;
              });

    int count = std::min(max_k, static_cast<int>(all_cands.size()));
    for (int i = 0; i < count; ++i) out[i] = all_cands[i];
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

    // Find the best candidate by Bayesian-smoothed win rate. Pure empirical
    // comparison is noisy when rollouts are few (e.g. 1/1 beats 4/5).
    int best_idx = 0;
    double best_wr = candidates[0].bayesian_wr();
    for (int i = 1; i < count; ++i) {
        double wr = candidates[i].bayesian_wr();
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

    // Hoisted rollout buffers — reused across every simulated game during
    // this turn. Previously simulate_rollout allocated a fresh ~1.6MB tensor
    // buffer per rollout, thrashing the heap and destroying cache locality.
    SolverBuffers rollout_buffers{};
    std::vector<float> rollout_tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);

    while (true) {
        // --- ADD THESE CACHE RESETS ---
        buffers.dp_computed = false;
        buffers.evs_blended = false;
        
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

        bool time_exhausted = false;
        while (elapsed_decision < decision_budget && !time_exhausted) {
            for (int ci = 0; ci < num_candidates && !time_exhausted; ++ci) {
                auto& cand = candidates[ci];
                for (int b = 0; b < mc_config.rollout_batch_size; ++b) {
                    RNG rollout_rng(rng.next());
                    double outcome;

                    if (cand.is_placement) {
                        // Apply the placement to get the afterstate, then
                        // simulate from the resulting board. Copy the live
                        // GameContext rather than calling the O(156)
                        // rebuild_context_from_board scan; apply_placement
                        // updates the context incrementally.
                        BoardState rollout_board = state.board;
                        GameContext tmp_ctx = ctx;
                        apply_placement(mc_player,
                                        cand.placement.column,
                                        cand.placement.row,
                                        cand.score,
                                        rollout_board, tmp_ctx);
                        rollout_board.current_player =
                            1 - rollout_board.current_player;
                        outcome = simulate_rollout(
                            rollout_board, tmp_ctx, mc_player,
                            model, device, tables,
                            rollout_buffers, rollout_tensor_buffer,
                            rollout_rng);
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
                        GameContext tmp_ctx = ctx;
                        outcome = simulate_rollout_from_state(
                            tmp_state, tmp_ctx, mc_player,
                            model, device, tables,
                            rollout_buffers, rollout_tensor_buffer,
                            rollout_rng);
                    }

                    cand.win_count += outcome;
                    cand.rollout_count++;
                }

                // Check the time budget between candidates so we don't
                // overshoot after a slow rollout batch.
                auto now_inner = Clock::now();
                elapsed_decision = std::chrono::duration<double, std::milli>(
                    now_inner - decision_start).count();
                if (elapsed_decision >= decision_budget) {
                    time_exhausted = true;
                    break;
                }
            }

            // Check early termination.
            if (can_terminate_early(candidates, num_candidates,
                                    mc_config.min_rollouts_per_candidate))
                break;
        }

        // Step 4: Pick the best candidate by Bayesian-smoothed win rate.
        // The NN EV acts as a prior so early-terminated rollouts (or zero
        // rollouts due to an exhausted time budget) gracefully fall back to
        // the NN's recommendation instead of statistical noise.
        int best_idx = 0;
        double best_wr = candidates[0].bayesian_wr();
        for (int i = 1; i < num_candidates; ++i) {
            double wr = candidates[i].bayesian_wr();
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
