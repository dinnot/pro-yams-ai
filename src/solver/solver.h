#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "engine/constants.h"
#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_state.h"
#include "engine/rng.h"
#include "solver/precomputed_tables.h"

// ---------------------------------------------------------------------------
// Afterstate request — one (placement, score) pair for the caller to evaluate.
// ---------------------------------------------------------------------------
struct AfterstateRequest {
    Placement placement;  // (column, row)
    int8_t    score;      // Score to place (0 = scratch)
};

// ---------------------------------------------------------------------------
// Solver result — the best action for the current dice/board state.
// ---------------------------------------------------------------------------
struct SolverResult {
    bool     should_place;         // true = place, false = hold and reroll
    uint8_t  hold_mask;            // Which dice to keep (valid only if !should_place)
    Placement placement;           // Where to place (valid only if should_place)
    int8_t   score;                // Score to place (valid only if should_place)
    double   expected_value;       // EV of the best action
    int16_t  chosen_request_idx;   // Index into requests[] for chosen placement (-1 if hold)
    double   pre_roll_ev;          // EV of the state before the first roll
    bool     is_exploratory;       // true if an off-policy action was taken
};

// ---------------------------------------------------------------------------
// SolverConfig — temperature parameters for exploration.
// ---------------------------------------------------------------------------
struct SolverConfig {
    double placement_temperature = 0.0;  // 0.0 = greedy, >0 = softmax exploration
    double hold_temperature = 0.0;       // 0.0 = greedy, >0 = softmax exploration
    bool   exploration_enabled = false;   // Master switch (false = always greedy)
    bool   debug_mode = false;
    double heuristic_weight = 0.0;
    int    heuristic_version = 2;  // 1 = V1 (greedy), 2 = V2 (DP-driven duel margin)
    bool   use_duel_margin_maximization = false;
    double duel_margin_maximization_scale = 4000.0;
    bool   use_pbrs          = false;
    double pbrs_upper_reward = 0.1;
    double pbrs_clean_reward = 0.2;
    std::string debug_log_path = "";
    bool   compute_pre_roll_ev = false;  // Opt-in V2 computation
};

// ---------------------------------------------------------------------------
// SolverBuffers — pre-allocated working storage, one instance per game thread.
// Reused across all solver calls to avoid per-call heap allocation.
// ---------------------------------------------------------------------------
constexpr int kMaxAfterstateRequests = 512;

struct SolverBuffers {
    AfterstateRequest requests[kMaxAfterstateRequests];
    int               request_count;

    double evs[kMaxAfterstateRequests];
    
    double raw_nn_evs[kMaxAfterstateRequests]; // Saves raw NN outputs for logging
    bool   evs_blended = false;                // Prevents double-blending heuristic
    bool   dp_computed = false;                // Layer 0 and Layer 1 DP cached flag

    constexpr static int kMaxHeldConfigs = 792; // Safe theoretical upper bound (actual is 462)

    // Internal DP tables
    double   v0[kNumDiceStates];               // Layer 0: no rerolls left
    double   v1[kNumDiceStates];               // Layer 1: one reroll left
    double   v2[kNumDiceStates];               // Layer 2: two rerolls left
    double   pre_roll_ev;                      // Pre-roll EV for the turn

    double   ev_held_v0[kMaxHeldConfigs];      // Fast-cache: EV of held configs -> V0
    double   ev_held_v1[kMaxHeldConfigs];      // Fast-cache: EV of held configs -> V1

    int16_t  req_map[78][101];                 // Maps [cell_idx][raw_score] -> req_idx. -1 if invalid
    int16_t  scratch_map[78];                  // Maps [cell_idx] -> scratch req_idx
    int8_t   cell_cols[78];
    int8_t   cell_rows[78];
    int8_t   num_legal_cells;

    int16_t  best_request_idx[kNumDiceStates]; // Best placement index at V0
    int16_t  best_mask_v1[kNumDiceStates];     // Best hold mask at V1 (-1=stop)

    double   stop_value[kNumDiceStates];       // Best EV including Turbo (for stop decisions)
    int16_t  stop_request_idx[kNumDiceStates]; // Best request idx including Turbo

    // Scratch space for softmax sampling (32 hold masks + 1 stop option)
    double   mask_evs[kNumHoldMasks + 1];
};

// ---------------------------------------------------------------------------
// Solver API (templated on game variant).
//
// The solver works from a single player's perspective — `state.board.current_player`
// indexes board.cells / ctx.legal_*. No multi-player loops; templating just
// generalises the data-structure types.
// ---------------------------------------------------------------------------

/// Step 1: Populate buffers.requests with every (placement, score) afterstate
/// that needs evaluation. Sets buffers.request_count.
template <typename Traits>
void solver_get_requests(const GameStateT<Traits>& state,
                         const GameContextT<Traits>& ctx,
                         const PrecomputedTables& tables, SolverBuffers& buffers);

/// Step 2: Given buffers.evs filled by the caller, compute the best action
/// via expectimax DP over reroll layers.
template <typename Traits>
SolverResult solver_resolve(const GameStateT<Traits>& state,
                            const GameContextT<Traits>& ctx,
                            const PrecomputedTables& tables, SolverBuffers& buffers,
                            const SolverConfig& config, RNG& rng);

/// Convenience: greedy resolve (no exploration).
template <typename Traits>
inline SolverResult solver_resolve_greedy(const GameStateT<Traits>& state,
                                          const GameContextT<Traits>& ctx,
                                          const PrecomputedTables& tables,
                                          SolverBuffers& buffers,
                                          bool compute_pre_roll_ev = false) {
    RNG dummy(0);
    SolverConfig cfg;
    cfg.compute_pre_roll_ev = compute_pre_roll_ev;
    return solver_resolve<Traits>(state, ctx, tables, buffers, cfg, dummy);
}

// ---------------------------------------------------------------------------
// Softmax sampling with logit transformation.
// ---------------------------------------------------------------------------

/// Sample an index proportionally from values using softmax.
/// If use_margin is false, values are win probabilities in (0, 1) and are
/// converted to logits via log(v/(1-v)). If true, values are margin logits in
/// [-1, 1] and are scaled directly by 3.0.
int softmax_sample(const double* values, int count, double temperature, RNG& rng,
                   bool use_margin = false);
