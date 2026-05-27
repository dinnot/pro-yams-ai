#include "solver/precomputed_tables.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Linearise a sorted 1-based 5-dice array → flat index into linear_to_id.
static int linearize(const std::array<int, 5>& d) {
    return (d[0]-1)*1296 + (d[1]-1)*216 + (d[2]-1)*36 + (d[3]-1)*6 + (d[4]-1);
}

// ---------------------------------------------------------------------------
// Step 1: Enumerate 252 sorted dice states
// ---------------------------------------------------------------------------
static void enumerate_states(PrecomputedTables& t) {
    std::memset(t.linear_to_id, -1, sizeof(t.linear_to_id));
    int id = 0;
    for (int a = 1; a <= 6; ++a)
      for (int b = a; b <= 6; ++b)
        for (int c = b; c <= 6; ++c)
          for (int d = c; d <= 6; ++d)
            for (int e = d; e <= 6; ++e) {
                t.id_to_state[id] = {a, b, c, d, e};
                t.linear_to_id[linearize({a, b, c, d, e})] = static_cast<int16_t>(id);
                ++id;
            }
}

// ---------------------------------------------------------------------------
// Step 2: Build MOVES table and collect unique held configurations
// ---------------------------------------------------------------------------
// Returns the held_map (held_vec → held_id) for use in Step 3.
static std::map<std::vector<int>, int> build_moves(PrecomputedTables& t) {
    std::map<std::vector<int>, int> held_map;
    int held_counter = 0;

    for (int state = 0; state < kNumDiceStates; ++state) {
        for (int mask = 0; mask < kNumHoldMasks; ++mask) {
            std::vector<int> held_dice;
            held_dice.reserve(5);
            for (int b = 0; b < 5; ++b) {
                if ((mask >> b) & 1) {
                    held_dice.push_back(t.id_to_state[state][b]);
                }
            }
            // Dice state is sorted, so extracting bits in order gives a sorted subset.

            auto it = held_map.find(held_dice);
            if (it == held_map.end()) {
                held_map[held_dice] = held_counter;
                t.moves[state][mask] = static_cast<int16_t>(held_counter);
                ++held_counter;
            } else {
                t.moves[state][mask] = static_cast<int16_t>(it->second);
            }
        }
    }

    t.num_held_configs = held_counter;
    return held_map;
}

// ---------------------------------------------------------------------------
// Step 3: Compute transition probability distributions
// ---------------------------------------------------------------------------
static void build_transitions(PrecomputedTables& t,
                               const std::map<std::vector<int>, int>& held_map) {
    t.transition_offset.resize(t.num_held_configs);
    t.transition_count.resize(t.num_held_configs);

    for (auto const& [held_vec, held_id] : held_map) {
        int n_reroll = 5 - static_cast<int>(held_vec.size());
        t.transition_offset[held_id] = static_cast<int32_t>(t.all_transitions.size());

        if (n_reroll == 0) {
            // Holding all 5 dice — deterministic, transition to the same state.
            std::array<int, 5> full;
            for (int i = 0; i < 5; ++i) full[i] = held_vec[i];
            int target = t.linear_to_id[linearize(full)];
            t.all_transitions.push_back({static_cast<int16_t>(target), 1.0});
            t.transition_count[held_id] = 1;
            continue;
        }

        double total = std::pow(6.0, n_reroll);
        std::map<int, int> counts;  // state_id → occurrence count

        // Enumerate all 6^n_reroll reroll outcomes recursively.
        std::vector<int> roll;
        roll.reserve(n_reroll);

        std::function<void()> enumerate = [&]() {
            if (static_cast<int>(roll.size()) == n_reroll) {
                std::vector<int> combined = held_vec;
                combined.insert(combined.end(), roll.begin(), roll.end());
                std::sort(combined.begin(), combined.end());
                std::array<int, 5> arr;
                for (int i = 0; i < 5; ++i) arr[i] = combined[i];
                int sid = t.linear_to_id[linearize(arr)];
                counts[sid]++;
                return;
            }
            for (int f = 1; f <= 6; ++f) {
                roll.push_back(f);
                enumerate();
                roll.pop_back();
            }
        };

        enumerate();

        for (auto const& [sid, cnt] : counts) {
            t.all_transitions.push_back({
                static_cast<int16_t>(sid),
                static_cast<double>(cnt) / total
            });
        }
        t.transition_count[held_id] = static_cast<int16_t>(counts.size());
    }
}

// ---------------------------------------------------------------------------
// Step 5: Probability tables (mini-solver DP per row, threshold)
// ---------------------------------------------------------------------------

// 3-layer DP (V0, V1, V2) to compute probability of achieving score >= threshold
// in a row with optimal holding.  The probability from a fresh turn is then
// E[V2] (3 rolls) or E[V1] (2 rolls) over the first-roll distribution.
static void build_probability_tables(PrecomputedTables& t) {
    // held_id = 0 corresponds to "hold nothing / reroll all 5 dice" (see build_moves).
    // Its transitions give the distribution of a fresh 5-dice roll.
    const int hold_nothing_id = t.moves[0][0];  // always 0 by construction

    for (int row = 0; row < kNumRows; ++row) {
        for (int threshold = 0; threshold <= 100; ++threshold) {
            // -----------------------------------------------------------------
            // V0: binary — does the dice state achieve score >= threshold?
            // -----------------------------------------------------------------
            double v0[kNumDiceStates];
            for (int sid = 0; sid < kNumDiceStates; ++sid) {
                v0[sid] = (t.score_tables.dice_score[sid][row] >= threshold) ? 1.0 : 0.0;
            }

            // -----------------------------------------------------------------
            // V1: best expected achievement with 1 optional reroll.
            //     max over hold masks (mask=31 = hold all = same as stopping).
            // -----------------------------------------------------------------
            double v1[kNumDiceStates];
            for (int sid = 0; sid < kNumDiceStates; ++sid) {
                double best = 0.0;
                for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                    int held_id = t.moves[sid][mask];
                    int cnt = 0;
                    const Transition* tr = t.all_transitions.data()
                                         + t.transition_offset[held_id];
                    cnt = t.transition_count[held_id];
                    double ev = 0.0;
                    for (int k = 0; k < cnt; ++k)
                        ev += tr[k].probability * v0[tr[k].target_state_id];
                    if (ev > best) best = ev;
                }
                v1[sid] = best;
            }

            // -----------------------------------------------------------------
            // V2: best expected achievement with 2 optional rerolls.
            // -----------------------------------------------------------------
            double v2[kNumDiceStates];
            for (int sid = 0; sid < kNumDiceStates; ++sid) {
                double best = 0.0;
                for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                    int held_id = t.moves[sid][mask];
                    int cnt = t.transition_count[held_id];
                    const Transition* tr = t.all_transitions.data()
                                         + t.transition_offset[held_id];
                    double ev = 0.0;
                    for (int k = 0; k < cnt; ++k)
                        ev += tr[k].probability * v1[tr[k].target_state_id];
                    if (ev > best) best = ev;
                }
                v2[sid] = best;
            }

            // -----------------------------------------------------------------
            // Aggregate over the first (mandatory) roll.
            // First roll = transitions from "hold nothing" (held_id = 0).
            // -----------------------------------------------------------------
            int cnt0 = t.transition_count[hold_nothing_id];
            const Transition* tr0 = t.all_transitions.data()
                                   + t.transition_offset[hold_nothing_id];

            double p2 = 0.0, p3 = 0.0;
            for (int k = 0; k < cnt0; ++k) {
                int sid = tr0[k].target_state_id;
                p2 += tr0[k].probability * v1[sid];
                p3 += tr0[k].probability * v2[sid];
            }

            t.prob_tables.prob_2rolls[row][threshold] = p2;
            t.prob_tables.prob_3rolls[row][threshold] = p3;

            // --- ADD PRE-COMPOUNDING CALCULATION ---
            for (int tr = 0; tr <= 78; ++tr) {
                t.prob_tables.prob_2rolls_compound[row][threshold][tr] = 
                    (tr == 0) ? 0.0f : static_cast<float>(1.0 - std::pow(1.0 - p2, static_cast<double>(tr)));
                t.prob_tables.prob_3rolls_compound[row][threshold][tr] = 
                    (tr == 0) ? 0.0f : static_cast<float>(1.0 - std::pow(1.0 - p3, static_cast<double>(tr)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void init_precomputed_tables(PrecomputedTables& tables) {
    // Step 1: 252 sorted dice states + linearisation lookup
    enumerate_states(tables);

    // Step 2: hold mask → held config mapping
    auto held_map = build_moves(tables);

    // Step 3: transition probability tables
    build_transitions(tables, held_map);

    // Step 4: score tables (delegates entirely to Task 02 implementation)
    init_solver_tables(tables.score_tables);

    // Step 5: probability tables (mini-solver DP, depends on transitions + score tables)
    build_probability_tables(tables);

    // Step 6: DP tables (V2.1). Load from cache when present; otherwise compute
    // the tables once and persist them so the next process loads instantly.
    // (Set PRO_YAMS_NO_DP=1 to skip this entirely and leave dp_tables empty —
    // DP-dependent tensor features then fall back to safe defaults of 0. Useful
    // for fast unit tests that do not exercise DP features.)
    if (const char* skip = std::getenv("PRO_YAMS_NO_DP"); !(skip && skip[0] && skip[0] != '0')) {
        std::vector<std::string> candidates;
        if (const char* env = std::getenv("DP_TABLES_CACHE"); env && env[0]) {
            candidates.emplace_back(env);
        }
        candidates.emplace_back("cache/dp_tables/dp_v1.bin");
        candidates.emplace_back("../cache/dp_tables/dp_v1.bin");
        candidates.emplace_back("/home/sorin/pro_yams_ai/cache/dp_tables/dp_v1.bin");

        std::string existing;
        for (const auto& path : candidates) {
            std::error_code ec;
            if (std::filesystem::exists(path, ec)) { existing = path; break; }
        }

        if (!existing.empty()) {
            // Loads if valid; init_dp_tables recomputes + re-saves on a stale
            // (version / size mismatch) cache.
            init_dp_tables(tables.dp_tables, tables, existing);
        } else {
            // No cache anywhere → compute now and save to the preferred path so
            // subsequent processes load it instantly. This is a one-time cost
            // (~1-2 min, ~3.4 GB on disk).
            const std::string& target = candidates.front();
            std::fprintf(stderr,
                "[pro_yams] DP cache not found; computing tables and saving to "
                "'%s' (one-time, ~1-2 min)...\n", target.c_str());
            init_dp_tables(tables.dp_tables, tables, target);
            std::fprintf(stderr, "[pro_yams] DP cache written.\n");
        }
    }
}

// ---------------------------------------------------------------------------
// Accessor implementations
// ---------------------------------------------------------------------------

int get_held_config(int dice_state_id, int hold_mask,
                    const PrecomputedTables& tables) {
    assert(dice_state_id >= 0 && dice_state_id < kNumDiceStates);
    assert(hold_mask >= 0 && hold_mask < kNumHoldMasks);
    return tables.moves[dice_state_id][hold_mask];
}

const Transition* get_transitions(int held_config_id, int& count,
                                  const PrecomputedTables& tables) {
    assert(held_config_id >= 0 && held_config_id < tables.num_held_configs);
    count = tables.transition_count[held_config_id];
    return tables.all_transitions.data() + tables.transition_offset[held_config_id];
}

int get_dice_state_id(const int8_t dice[kNumDice],
                      const PrecomputedTables& tables) {
    int8_t sorted[kNumDice] = {dice[0], dice[1], dice[2], dice[3], dice[4]};
    std::sort(sorted, sorted + kNumDice);
    int idx = (sorted[0]-1)*1296 + (sorted[1]-1)*216 + (sorted[2]-1)*36
            + (sorted[3]-1)*6 + (sorted[4]-1);
    assert(idx >= 0 && idx < kMaxLinearIndex);
    return tables.linear_to_id[idx];
}

int get_dice_score(int dice_state_id, int row,
                   const PrecomputedTables& tables) {
    assert(dice_state_id >= 0 && dice_state_id < kNumDiceStates);
    assert(row >= 0 && row < kNumRows);
    return tables.score_tables.dice_score[dice_state_id][row];
}

const int8_t* get_filtered_scores(int row, int min_threshold, int& count,
                                  const PrecomputedTables& tables) {
    assert(row >= 0 && row < kNumRows);
    assert(min_threshold >= 0 && min_threshold <= 100);
    count = tables.score_tables.filtered_count[row][min_threshold];
    return tables.score_tables.filtered_scores[row][min_threshold];
}
