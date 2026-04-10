#include "solver/solver.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "engine/game_flow.h"
#include "engine/scoring.h"

// ---------------------------------------------------------------------------
// solver_get_requests
// ---------------------------------------------------------------------------

void solver_get_requests(const GameState& state, const GameContext& ctx,
                         const PrecomputedTables& tables, SolverBuffers& buffers) {
    int p = state.board.current_player;
    buffers.request_count = 0;

    const LegalPlacementCache& legal = get_legal_placements(state, ctx);

    for (int i = 0; i < legal.count; ++i) {
        int col = legal.placements[i].column;
        int row = legal.placements[i].row;

        // Always include the scratch option.
        assert(buffers.request_count < kMaxAfterstateRequests);
        buffers.requests[buffers.request_count++] = {{(int8_t)col, (int8_t)row}, 0};

        // Determine Golden Rule minimum for this cell.
        int min_threshold = ctx.golden_max[col][row];

        // Handle SS/LS constraints.
        bool forced_scratch = false;

        if (row == kRowSS) {
            // If LS is already scratched, SS must also scratch.
            if (ctx.ls_scratched[p][col]) forced_scratch = true;
        } else if (row == kRowLS) {
            // If SS is already scratched, LS must also scratch.
            if (ctx.ss_scratched[p][col]) forced_scratch = true;
        }

        if (forced_scratch) {
            // Only scratch is valid — already added above.
            continue;
        }

        // Get filtered scores >= min_threshold.
        int count = 0;
        const int8_t* scores = get_filtered_scores(row, min_threshold, count, tables);

        for (int j = 0; j < count; ++j) {
            int score = scores[j];

            // SS/LS ordering constraint (when the other row is already filled).
            if (row == kRowSS) {
                // SS must be < LS if LS is already filled.
                int8_t ls_val = state.board.cells[p][col][kRowLS];
                if (ls_val != kCellEmpty && ls_val > 0 && score >= ls_val)
                    continue;
            } else if (row == kRowLS) {
                // LS must be strictly greater than highest SS recorded by anyone.
                int max_ss = ctx.golden_max[col][kRowSS];
                if (max_ss > 0 && score <= max_ss)
                    continue;
            }

            assert(buffers.request_count < kMaxAfterstateRequests);
            buffers.requests[buffers.request_count++] = {{(int8_t)col, (int8_t)row}, (int8_t)score};
        }
    }
}

// ---------------------------------------------------------------------------
// softmax_sample
// ---------------------------------------------------------------------------

int softmax_sample(const double* values, int count, double temperature, RNG& rng) {
    assert(count > 0);
    if (count == 1) return 0;

    constexpr double kClampMin = 1e-6;
    constexpr double kClampMax = 1.0 - 1e-6;

    // Convert win probabilities to logits.
    double max_logit = -1e18;
    double logits[kMaxAfterstateRequests];  // generous, but caller ensures count <= this
    for (int i = 0; i < count; ++i) {
        double v = std::max(kClampMin, std::min(kClampMax, values[i]));
        logits[i] = std::log(v / (1.0 - v));
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Softmax with temperature (shift by max for numerical stability).
    double weights[kMaxAfterstateRequests];
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        weights[i] = std::exp((logits[i] - max_logit) / temperature);
        sum += weights[i];
    }

    // Sample proportionally.
    double r = rng.uniform_double() * sum;
    double cumulative = 0.0;
    for (int i = 0; i < count - 1; ++i) {
        cumulative += weights[i];
        if (r < cumulative) return i;
    }
    return count - 1;
}

// ---------------------------------------------------------------------------
// solver_resolve
// ---------------------------------------------------------------------------

SolverResult solver_resolve(const GameState& state, const GameContext& ctx,
                            const PrecomputedTables& tables, SolverBuffers& buffers,
                            const SolverConfig& config, RNG& rng) {
    const int p = state.board.current_player;
    const int rolls_left = state.rolls_left;

    // -----------------------------------------------------------------------
    // Build V0: for each dice state, find the best placement given buffers.evs.
    // -----------------------------------------------------------------------
    for (int sid = 0; sid < kNumDiceStates; ++sid) {
        double best_ev_stop = -std::numeric_limits<double>::infinity();
        double best_ev_no_turbo = -std::numeric_limits<double>::infinity();
        int16_t best_req_stop = -1;
        int16_t best_req_no_turbo = -1;

        for (int i = 0; i < buffers.request_count; ++i) {
            const AfterstateRequest& req = buffers.requests[i];
            bool achievable;

            if (req.score == 0) {
                achievable = true;
            } else {
                achievable = (tables.score_tables.dice_score[sid][req.placement.row] == req.score);
            }

            if (achievable) {
                if (buffers.evs[i] > best_ev_stop) {
                    best_ev_stop = buffers.evs[i];
                    best_req_stop = static_cast<int16_t>(i);
                }
                if (req.placement.column != kColTurbo) {
                    if (buffers.evs[i] > best_ev_no_turbo) {
                        best_ev_no_turbo = buffers.evs[i];
                        best_req_no_turbo = static_cast<int16_t>(i);
                    }
                }
            }
        }
        buffers.v0[sid] = best_ev_no_turbo;
        buffers.best_request_idx[sid] = best_req_no_turbo;
        buffers.stop_value[sid] = best_ev_stop;
        buffers.stop_request_idx[sid] = best_req_stop;
    }

    // Short-circuit: if no rerolls remain, return the best placement for current dice.
    if (rolls_left == 0) {
        int current_id = get_dice_state_id(state.dice, tables);
        SolverResult result;
        result.should_place = true;
        result.hold_mask = 0;
        result.expected_value = buffers.stop_value[current_id];

        // Apply placement temperature if exploration enabled.
        if (config.exploration_enabled && config.placement_temperature > 0.0) {
            double place_evs[kMaxAfterstateRequests];
            int   place_idx[kMaxAfterstateRequests];
            int   place_count = 0;
            for (int i = 0; i < buffers.request_count; ++i) {
                const AfterstateRequest& req = buffers.requests[i];
                bool achievable = (req.score == 0) ||
                    (tables.score_tables.dice_score[current_id][req.placement.row] == req.score);
                if (achievable) {
                    place_evs[place_count] = buffers.evs[i];
                    place_idx[place_count] = i;
                    ++place_count;
                }
            }
            if (place_count > 1) {
                int sel = softmax_sample(place_evs, place_count,
                                         config.placement_temperature, rng);
                int req_i = place_idx[sel];
                result.placement = buffers.requests[req_i].placement;
                result.score = buffers.requests[req_i].score;
                // Keep expected_value as greedy max (not sampled action's value)
                // so TD bootstrap targets stay off-policy-correct with replay.
                result.chosen_request_idx = static_cast<int16_t>(req_i);
                return result;
            }
        }

        int16_t req_idx = buffers.stop_request_idx[current_id];
        if (req_idx >= 0) {
            result.placement = buffers.requests[req_idx].placement;
            result.score = buffers.requests[req_idx].score;
            result.chosen_request_idx = req_idx;
        } else {
            result.placement = buffers.requests[0].placement;
            result.score = 0;
            result.chosen_request_idx = 0;
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Build V1: for each dice state, best hold mask vs stopping.
    // -----------------------------------------------------------------------
    for (int sid = 0; sid < kNumDiceStates; ++sid) {
        double best_ev = buffers.stop_value[sid];  // stop value (includes Turbo)
        int16_t best_mask = -1;                    // -1 = stop

        for (int mask = 0; mask < kNumHoldMasks; ++mask) {
            int held_id = tables.moves[sid][mask];
            int count = 0;
            const Transition* tr = get_transitions(held_id, count, tables);

            double ev = 0.0;
            for (int t = 0; t < count; ++t)
                ev += tr[t].probability * buffers.v0[tr[t].target_state_id];

            if (ev > best_ev) {
                best_ev = ev;
                best_mask = static_cast<int16_t>(mask);
            }
        }
        buffers.v1[sid] = best_ev;
        buffers.best_mask_v1[sid] = best_mask;
    }

    int current_id = get_dice_state_id(state.dice, tables);

    // -----------------------------------------------------------------------
    // 1 or 2 rerolls left: decide hold mask vs stop-and-place.
    // When rolls_left == 1, transitions land in V0; when 2, in V1.
    // -----------------------------------------------------------------------
    double best_ev = buffers.stop_value[current_id];  // stop value (includes Turbo)
    bool best_is_place = true;
    uint8_t best_mask = 0;

    // If only Turbo cells remain at rolls_left == 1, must place now —
    // rerolling would drop rolls_left to 0, making Turbo unavailable (deadlock).
    bool force_place = (rolls_left == 1 && ctx.non_turbo_cells_remaining[p] == 0);

    if (!force_place) {
        bool use_exploration = config.exploration_enabled && config.hold_temperature > 0.0;
        const double* target_v = (rolls_left == 1) ? buffers.v0 : buffers.v1;

        if (use_exploration) {
            // Collect EV for all hold masks.
            double greedy_max_ev = best_ev; // starts as stop_value
            for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                int held_id = tables.moves[current_id][mask];
                int count = 0;
                const Transition* tr = get_transitions(held_id, count, tables);
                double ev = 0.0;
                for (int t = 0; t < count; ++t)
                    ev += tr[t].probability * target_v[tr[t].target_state_id];
                buffers.mask_evs[mask] = ev;
                if (ev > greedy_max_ev) greedy_max_ev = ev;
            }
            // Include stop as the 33rd option in the softmax.
            buffers.mask_evs[kNumHoldMasks] = best_ev;

            int selected = softmax_sample(buffers.mask_evs, kNumHoldMasks + 1,
                                          config.hold_temperature, rng);
            if (selected == kNumHoldMasks) {
                // Stop and place (best_is_place remains true).
                best_ev = greedy_max_ev; 
            } else {
                best_mask = static_cast<uint8_t>(selected);
                best_is_place = false;
                best_ev = greedy_max_ev; // Use the max EV, not the sampled EV!
            }
        } else {
            // Greedy: find best mask.
            // Always fill mask_evs so callers can read hold evaluations for debug.
            for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                int held_id = tables.moves[current_id][mask];
                int count = 0;
                const Transition* tr = get_transitions(held_id, count, tables);
                double ev = 0.0;
                for (int t = 0; t < count; ++t)
                    ev += tr[t].probability * target_v[tr[t].target_state_id];
                buffers.mask_evs[mask] = ev;
                if (ev > best_ev) {
                    best_ev = ev;
                    best_mask = static_cast<uint8_t>(mask);
                    best_is_place = false;
                }
            }
            // Index kNumHoldMasks = stop-and-place option (mirrors the softmax path).
            buffers.mask_evs[kNumHoldMasks] = buffers.stop_value[current_id];
        }
    }

    SolverResult result;
    if (best_is_place) {
        int16_t req_idx = buffers.stop_request_idx[current_id];
        result.should_place = true;
        result.hold_mask = 0;
        result.placement = (req_idx >= 0) ? buffers.requests[req_idx].placement
                                          : buffers.requests[0].placement;
        result.score = (req_idx >= 0) ? buffers.requests[req_idx].score : 0;
        result.expected_value = buffers.stop_value[current_id];
        result.chosen_request_idx = (req_idx >= 0) ? req_idx : static_cast<int16_t>(0);
    } else {
        result.should_place = false;
        result.hold_mask = best_mask;
        result.placement = {};
        result.score = 0;
        result.expected_value = best_ev;
        result.chosen_request_idx = -1;
    }

    // Apply placement exploration if we decided to place.
    if (result.should_place && config.exploration_enabled &&
        config.placement_temperature > 0.0) {
        // Collect V0 values for all requests achievable by the current dice state.
        // Rebuild a list of (request_idx, ev) pairs.
        // For simplicity, collect EVs of achievable placements and sample.
        double place_evs[kMaxAfterstateRequests];
        int   place_idx[kMaxAfterstateRequests];
        int   place_count = 0;
        for (int i = 0; i < buffers.request_count; ++i) {
            const AfterstateRequest& req = buffers.requests[i];
            bool achievable = (req.score == 0) ||
                (tables.score_tables.dice_score[current_id][req.placement.row] == req.score);
            if (achievable) {
                place_evs[place_count] = buffers.evs[i];
                place_idx[place_count] = i;
                ++place_count;
            }
        }
        if (place_count > 1) {
            int sel = softmax_sample(place_evs, place_count,
                                     config.placement_temperature, rng);
            int req_i = place_idx[sel];
            result.placement = buffers.requests[req_i].placement;
            result.score = buffers.requests[req_i].score;
            // Keep expected_value as greedy max (not sampled action's value).
            result.chosen_request_idx = static_cast<int16_t>(req_i);
        }
    }

    return result;
}
