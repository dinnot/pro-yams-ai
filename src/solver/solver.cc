#include "solver/solver.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "engine/game_flow.h"
#include "engine/game_traits.h"
#include "engine/scoring.h"

// ---------------------------------------------------------------------------
// solver_get_requests
// ---------------------------------------------------------------------------

template <typename Traits>
void solver_get_requests(const GameStateT<Traits>& state,
                         const GameContextT<Traits>& ctx,
                         const PrecomputedTables& tables, SolverBuffers& buffers) {
    int p = state.board.current_player;
    buffers.request_count = 0;
    buffers.num_legal_cells = 0;

    const LegalPlacementCache& legal = get_legal_placements<Traits>(state, ctx);

    for (int i = 0; i < legal.count; ++i) {
        int col = legal.placements[i].column;
        int row = legal.placements[i].row;

        int cell_idx = buffers.num_legal_cells++;
        buffers.cell_cols[cell_idx] = static_cast<int8_t>(col);
        buffers.cell_rows[cell_idx] = static_cast<int8_t>(row);
        buffers.scratch_map[cell_idx] = -1;
        for (int s = 0; s <= 100; ++s) buffers.req_map[cell_idx][s] = -1;

        int min_threshold = ctx.golden_max[col][row];
        bool forced_scratch = false;

        if (row == kRowSS) {
            if (ctx.ls_scratched[p][col]) forced_scratch = true;
        } else if (row == kRowLS) {
            if (ctx.ss_scratched[p][col]) forced_scratch = true;
        }

        if (forced_scratch) {
            assert(buffers.request_count < kMaxAfterstateRequests);
            buffers.scratch_map[cell_idx] = buffers.request_count;
            buffers.requests[buffers.request_count++] = {{(int8_t)col, (int8_t)row}, 0};
            continue;
        }

        int count = 0;
        const int8_t* scores = get_filtered_scores(row, min_threshold, count, tables);

        for (int j = count - 1; j >= 0; --j) {
            int score = scores[j];
            if (row == kRowSS) {
                int8_t ls_val = state.board.cells[p][col][kRowLS];
                if (ls_val != kCellEmpty && ls_val > 0 && score >= ls_val) continue;
            } else if (row == kRowLS) {
                int max_ss = ctx.golden_max[col][kRowSS];
                if (max_ss > 0 && score <= max_ss) continue;
            }

            assert(buffers.request_count < kMaxAfterstateRequests);
            buffers.req_map[cell_idx][score] = buffers.request_count;
            buffers.requests[buffers.request_count++] = {{(int8_t)col, (int8_t)row}, (int8_t)score};
        }

        assert(buffers.request_count < kMaxAfterstateRequests);
        buffers.scratch_map[cell_idx] = buffers.request_count;
        buffers.requests[buffers.request_count++] = {{(int8_t)col, (int8_t)row}, 0};
    }
}

// ---------------------------------------------------------------------------
// softmax_sample
// ---------------------------------------------------------------------------

int softmax_sample(const double* values, int count, double temperature, RNG& rng,
                   bool use_margin) {
    assert(count > 0);
    if (count == 1) return 0;

    constexpr double kClampMin = 1e-6;
    constexpr double kClampMax = 1.0 - 1e-6;

    double max_logit = -1e18;
    double logits[kMaxAfterstateRequests];  // generous, but caller ensures count <= this
    if (use_margin) {
        // Values are already margin logits in [-1, 1]; scale to give temperature
        // a reasonable dynamic range.
        for (int i = 0; i < count; ++i) {
            logits[i] = values[i] * 3.0;
            if (logits[i] > max_logit) max_logit = logits[i];
        }
    } else {
        // Convert win probabilities to logits.
        for (int i = 0; i < count; ++i) {
            double v = std::max(kClampMin, std::min(kClampMax, values[i]));
            logits[i] = std::log(v / (1.0 - v));
            if (logits[i] > max_logit) max_logit = logits[i];
        }
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

template <typename Traits>
SolverResult solver_resolve(const GameStateT<Traits>& state,
                            const GameContextT<Traits>& ctx,
                            const PrecomputedTables& tables, SolverBuffers& buffers,
                            const SolverConfig& config, RNG& rng) {
    const int p = state.board.current_player;
    const int rolls_left = state.rolls_left;

    // -----------------------------------------------------------------------
    // "Lucky Yams" first-roll bonus: the player may place the maximum legal
    // score in any legal cell. Re-rolling and dice-based placement are strictly
    // dominated, so the decision reduces to picking the best cell at its
    // wildcard max score. The afterstate for (cell, max_score) is already among
    // buffers.requests (solver_get_requests enumerates every achievable score),
    // so we just read the caller-filled EVs — no extra evaluation needed.
    // -----------------------------------------------------------------------
    if (yams_bonus_active<Traits>(state)) {
        double place_evs[kMaxAfterstateRequests];
        int    place_idx[kMaxAfterstateRequests];
        int    place_count = 0;
        double best_ev = -std::numeric_limits<double>::infinity();
        int    best_req = -1;

        for (int c = 0; c < buffers.num_legal_cells; ++c) {
            int col = buffers.cell_cols[c];
            int row = buffers.cell_rows[c];
            int bonus = calculate_yams_bonus_score<Traits>(row, p, col, state.board, ctx);
            int req_idx = (bonus > 0 && buffers.req_map[c][bonus] != -1)
                          ? buffers.req_map[c][bonus]
                          : buffers.scratch_map[c];
            if (req_idx < 0) continue;

            double ev = buffers.evs[req_idx];
            place_evs[place_count] = ev;
            place_idx[place_count] = req_idx;
            ++place_count;
            if (ev > best_ev) { best_ev = ev; best_req = req_idx; }
        }

        SolverResult result;
        result.should_place      = true;
        result.hold_mask         = 0;
        // The turn resolves to an immediate placement, so the best afterstate
        // value is the natural turn-start value estimate (used as the TD
        // bootstrap in self-play) — far better than a meaningless pre-roll EV.
        result.pre_roll_ev       = best_ev;
        result.is_exploratory    = false;
        result.expected_value    = best_ev;
        result.placement         = buffers.requests[best_req].placement;
        result.score             = buffers.requests[best_req].score;
        result.chosen_request_idx = static_cast<int16_t>(best_req);

        if (config.exploration_enabled && config.placement_temperature > 0.0 &&
            place_count > 1) {
            int sel = softmax_sample(place_evs, place_count,
                                     config.placement_temperature, rng,
                                     config.use_duel_margin_maximization);
            int req_i = place_idx[sel];
            result.placement = buffers.requests[req_i].placement;
            result.score = buffers.requests[req_i].score;
            result.chosen_request_idx = static_cast<int16_t>(req_i);

            double max_ev = -1e9;
            for (int i = 0; i < place_count; ++i)
                if (place_evs[i] > max_ev) max_ev = place_evs[i];
            if (place_evs[sel] < max_ev - 1e-6) result.is_exploratory = true;
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Build V0, V1, V2 efficiently by computing expected values per held config
    // -----------------------------------------------------------------------
    if (!buffers.dp_computed) {
        for (int sid = 0; sid < kNumDiceStates; ++sid) {
            double best_ev_stop = -std::numeric_limits<double>::infinity();
            double best_ev_no_turbo = -std::numeric_limits<double>::infinity();
            int16_t best_req_stop = -1;
            int16_t best_req_no_turbo = -1;

            for (int c = 0; c < buffers.num_legal_cells; ++c) {
                int col = buffers.cell_cols[c];
                int row = buffers.cell_rows[c];
                int raw = tables.score_tables.dice_score[sid][row];

                int req_idx = (raw > 0 && buffers.req_map[c][raw] != -1)
                              ? buffers.req_map[c][raw]
                              : buffers.scratch_map[c];

                if (req_idx != -1) {
                    double ev = buffers.evs[req_idx];
                    if (ev > best_ev_stop) {
                        best_ev_stop = ev;
                        best_req_stop = static_cast<int16_t>(req_idx);
                    }
                    if (col != kColTurbo) {
                        if (ev > best_ev_no_turbo) {
                            best_ev_no_turbo = ev;
                            best_req_no_turbo = static_cast<int16_t>(req_idx);
                        }
                    }
                }
            }
            buffers.v0[sid] = best_ev_no_turbo;
            buffers.best_request_idx[sid] = best_req_no_turbo;
            buffers.stop_value[sid] = best_ev_stop;
            buffers.stop_request_idx[sid] = best_req_stop;
        }

        // Fast-cache: EV of held configs transitioning into V0
        for (int hid = 0; hid < tables.num_held_configs; ++hid) {
            int count = tables.transition_count[hid];
            const Transition* tr = tables.all_transitions.data() + tables.transition_offset[hid];
            double ev = 0.0;
            for (int t = 0; t < count; ++t) {
                ev += tr[t].probability * buffers.v0[tr[t].target_state_id];
            }
            buffers.ev_held_v0[hid] = ev;
        }

        // Build V1
        for (int sid = 0; sid < kNumDiceStates; ++sid) {
            double best_ev = buffers.stop_value[sid];
            int16_t best_mask = -1;
            for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                double ev = buffers.ev_held_v0[tables.moves[sid][mask]];
                if (ev > best_ev) {
                    best_ev = ev;
                    best_mask = static_cast<int16_t>(mask);
                }
            }
            buffers.v1[sid] = best_ev;
            buffers.best_mask_v1[sid] = best_mask;
        }

        // Gated V2 & Pre-roll
        if (config.compute_pre_roll_ev) {
            for (int hid = 0; hid < tables.num_held_configs; ++hid) {
                int count = tables.transition_count[hid];
                const Transition* tr = tables.all_transitions.data() + tables.transition_offset[hid];
                double ev = 0.0;
                for (int t = 0; t < count; ++t) {
                    ev += tr[t].probability * buffers.v1[tr[t].target_state_id];
                }
                buffers.ev_held_v1[hid] = ev;
            }

            for (int sid = 0; sid < kNumDiceStates; ++sid) {
                double best_ev = buffers.stop_value[sid];
                for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                    double ev = buffers.ev_held_v1[tables.moves[sid][mask]];
                    if (ev > best_ev) best_ev = ev;
                }
                buffers.v2[sid] = best_ev;
            }

            int held_id_none = tables.moves[0][0];
            int count_none = tables.transition_count[held_id_none];
            const Transition* tr_none = tables.all_transitions.data() + tables.transition_offset[held_id_none];
            double pre_roll = 0.0;
            for (int t = 0; t < count_none; ++t) {
                pre_roll += tr_none[t].probability * buffers.v2[tr_none[t].target_state_id];
            }
            buffers.pre_roll_ev = pre_roll;
        } else {
            buffers.pre_roll_ev = 0.0;
        }

        buffers.dp_computed = true;
    }

    int current_id = get_dice_state_id(state.dice, tables);

    // Short-circuit: if no rerolls remain, return the best placement for current dice.
    if (rolls_left == 0) {
        SolverResult result;
        result.should_place = true;
        result.hold_mask = 0;

        result.expected_value = buffers.v0[current_id];
        result.pre_roll_ev = buffers.pre_roll_ev;
        result.is_exploratory = false;

        if (config.exploration_enabled && config.placement_temperature > 0.0) {
            double place_evs[kMaxAfterstateRequests];
            int   place_idx[kMaxAfterstateRequests];
            int   place_count = 0;

            for (int c = 0; c < buffers.num_legal_cells; ++c) {
                int col = buffers.cell_cols[c];
                if (col == kColTurbo) continue;
                int row = buffers.cell_rows[c];

                int raw = tables.score_tables.dice_score[current_id][row];
                int req_idx = (raw > 0 && buffers.req_map[c][raw] != -1)
                              ? buffers.req_map[c][raw]
                              : buffers.scratch_map[c];

                if (req_idx != -1) {
                    place_evs[place_count] = buffers.evs[req_idx];
                    place_idx[place_count] = req_idx;
                    ++place_count;
                }
            }

            if (place_count > 1) {
                int sel = softmax_sample(place_evs, place_count,
                                         config.placement_temperature, rng,
                                         config.use_duel_margin_maximization);
                int req_i = place_idx[sel];
                result.placement = buffers.requests[req_i].placement;
                result.score = buffers.requests[req_i].score;
                result.chosen_request_idx = static_cast<int16_t>(req_i);

                double max_ev = -1e9;
                for (int i = 0; i < place_count; ++i) {
                    if (place_evs[i] > max_ev) max_ev = place_evs[i];
                }
                if (place_evs[sel] < max_ev - 1e-6) {
                    result.is_exploratory = true;
                }

                return result;
            }
        }

        int16_t req_idx = buffers.best_request_idx[current_id];
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
    // 1 or 2 rerolls left: decide hold mask vs stop-and-place.
    // -----------------------------------------------------------------------
    double best_ev = buffers.stop_value[current_id];
    bool best_is_place = true;
    uint8_t best_mask = 0;

    bool force_place = (rolls_left == 1 && ctx.non_turbo_cells_remaining[p] == 0);
    bool is_exploratory = false;

    if (!force_place) {
        bool use_exploration = config.exploration_enabled && config.hold_temperature > 0.0;

        if (use_exploration) {
            double greedy_max_ev = best_ev;
            for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                int held_id = tables.moves[current_id][mask];
                double ev = 0.0;
                if (rolls_left == 1) {
                    ev = buffers.ev_held_v0[held_id];
                } else {
                    if (config.compute_pre_roll_ev) {
                        ev = buffers.ev_held_v1[held_id];
                    } else {
                        int count = tables.transition_count[held_id];
                        const Transition* tr = tables.all_transitions.data() + tables.transition_offset[held_id];
                        for (int t = 0; t < count; ++t) {
                            ev += tr[t].probability * buffers.v1[tr[t].target_state_id];
                        }
                    }
                }
                buffers.mask_evs[mask] = ev;
                if (ev > greedy_max_ev) greedy_max_ev = ev;
            }
            buffers.mask_evs[kNumHoldMasks] = best_ev;

            int selected = softmax_sample(buffers.mask_evs, kNumHoldMasks + 1,
                                          config.hold_temperature, rng,
                                          config.use_duel_margin_maximization);
            if (selected == kNumHoldMasks) {
                best_ev = greedy_max_ev;
            } else {
                best_mask = static_cast<uint8_t>(selected);
                best_is_place = false;
                best_ev = greedy_max_ev;
            }
            if (buffers.mask_evs[selected] < greedy_max_ev - 1e-6) {
                is_exploratory = true;
            }
        } else {
            for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                int held_id = tables.moves[current_id][mask];
                double ev = 0.0;
                if (rolls_left == 1) {
                    ev = buffers.ev_held_v0[held_id];
                } else {
                    if (config.compute_pre_roll_ev) {
                        ev = buffers.ev_held_v1[held_id];
                    } else {
                        int count = tables.transition_count[held_id];
                        const Transition* tr = tables.all_transitions.data() + tables.transition_offset[held_id];
                        for (int t = 0; t < count; ++t) {
                            ev += tr[t].probability * buffers.v1[tr[t].target_state_id];
                        }
                    }
                }
                buffers.mask_evs[mask] = ev;
                if (ev > best_ev) {
                    best_ev = ev;
                    best_mask = static_cast<uint8_t>(mask);
                    best_is_place = false;
                }
            }
            buffers.mask_evs[kNumHoldMasks] = buffers.stop_value[current_id];
        }
    }

    SolverResult result;
    result.pre_roll_ev = buffers.pre_roll_ev;
    result.is_exploratory = is_exploratory;

    if (best_is_place) {
        int16_t req_idx = buffers.stop_request_idx[current_id];
        result.should_place = true;
        result.hold_mask = 0;
        if (req_idx >= 0) {
            result.placement = buffers.requests[req_idx].placement;
            result.score = buffers.requests[req_idx].score;
            result.chosen_request_idx = req_idx;
        } else {
            result.placement = buffers.requests[0].placement;
            result.score = 0;
            result.chosen_request_idx = 0;
        }
        result.expected_value = best_ev;

        if (config.exploration_enabled && config.placement_temperature > 0.0) {
            double place_evs[kMaxAfterstateRequests];
            int   place_idx[kMaxAfterstateRequests];
            int   place_count = 0;

            for (int c = 0; c < buffers.num_legal_cells; ++c) {
                int row = buffers.cell_rows[c];
                int raw = tables.score_tables.dice_score[current_id][row];
                int req_idx2 = (raw > 0 && buffers.req_map[c][raw] != -1)
                              ? buffers.req_map[c][raw]
                              : buffers.scratch_map[c];

                if (req_idx2 != -1) {
                    place_evs[place_count] = buffers.evs[req_idx2];
                    place_idx[place_count] = req_idx2;
                    ++place_count;
                }
            }

            if (place_count > 1) {
                int sel = softmax_sample(place_evs, place_count,
                                         config.placement_temperature, rng,
                                         config.use_duel_margin_maximization);
                int req_i = place_idx[sel];
                result.placement = buffers.requests[req_i].placement;
                result.score = buffers.requests[req_i].score;
                result.chosen_request_idx = static_cast<int16_t>(req_i);

                double max_place_ev = -1e9;
                for (int i = 0; i < place_count; ++i) {
                    if (place_evs[i] > max_place_ev) max_place_ev = place_evs[i];
                }
                if (place_evs[sel] < max_place_ev - 1e-6) {
                    result.is_exploratory = true;
                }
            }
        }
    } else {
        result.should_place = false;
        result.hold_mask = best_mask;
        result.placement = {};
        result.score = 0;
        result.expected_value = best_ev;
        result.chosen_request_idx = -1;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template void solver_get_requests<Yams1v1>(const GameStateT<Yams1v1>&,
                                           const GameContextT<Yams1v1>&,
                                           const PrecomputedTables&, SolverBuffers&);
template void solver_get_requests<Yams2v2>(const GameStateT<Yams2v2>&,
                                           const GameContextT<Yams2v2>&,
                                           const PrecomputedTables&, SolverBuffers&);
template SolverResult solver_resolve<Yams1v1>(const GameStateT<Yams1v1>&,
                                              const GameContextT<Yams1v1>&,
                                              const PrecomputedTables&, SolverBuffers&,
                                              const SolverConfig&, RNG&);
template SolverResult solver_resolve<Yams2v2>(const GameStateT<Yams2v2>&,
                                              const GameContextT<Yams2v2>&,
                                              const PrecomputedTables&, SolverBuffers&,
                                              const SolverConfig&, RNG&);
