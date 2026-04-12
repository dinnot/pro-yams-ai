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
};

// ---------------------------------------------------------------------------
// SolverConfig — temperature parameters for exploration.
// ---------------------------------------------------------------------------
struct SolverConfig {
    double placement_temperature;  // 0.0 = greedy, >0 = softmax exploration
    double hold_temperature;       // 0.0 = greedy, >0 = softmax exploration
    bool   exploration_enabled;    // Master switch (false = always greedy)
    bool   debug_mode = false;
    double heuristic_weight = 0.0;
    std::string debug_log_path;
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

    // Internal DP tables
    double   v0[kNumDiceStates];               // Layer 0: no rerolls left
    double   v1[kNumDiceStates];               // Layer 1: one reroll left

    int16_t  best_request_idx[kNumDiceStates]; // Best placement index at V0
    int16_t  best_mask_v1[kNumDiceStates];     // Best hold mask at V1 (-1=stop)

    double   stop_value[kNumDiceStates];       // Best EV including Turbo (for stop decisions)
    int16_t  stop_request_idx[kNumDiceStates]; // Best request idx including Turbo

    // Scratch space for softmax sampling (32 hold masks + 1 stop option)
    double   mask_evs[kNumHoldMasks + 1];
};

// ---------------------------------------------------------------------------
// Solver API
// ---------------------------------------------------------------------------

/// Step 1: Populate buffers.requests with every (placement, score) afterstate
/// that needs evaluation. Sets buffers.request_count.
void solver_get_requests(const GameState& state, const GameContext& ctx,
                         const PrecomputedTables& tables, SolverBuffers& buffers);

/// Step 2: Given buffers.evs filled by the caller, compute the best action
/// via expectimax DP over reroll layers.
SolverResult solver_resolve(const GameState& state, const GameContext& ctx,
                            const PrecomputedTables& tables, SolverBuffers& buffers,
                            const SolverConfig& config, RNG& rng);

/// Convenience: greedy resolve (no exploration).
inline SolverResult solver_resolve_greedy(const GameState& state, const GameContext& ctx,
                                          const PrecomputedTables& tables,
                                          SolverBuffers& buffers) {
    // Provide a dummy RNG — it won't be called in greedy mode.
    RNG dummy(0);
    SolverConfig cfg{0.0, 0.0, false};
    return solver_resolve(state, ctx, tables, buffers, cfg, dummy);
}

// ---------------------------------------------------------------------------
// Softmax sampling with logit transformation.
// ---------------------------------------------------------------------------

/// Sample an index proportionally from win-probability values using softmax.
/// Converts values to logits before applying temperature.
/// values must be in (0, 1) — clamped internally if needed.
int softmax_sample(const double* values, int count, double temperature, RNG& rng);
