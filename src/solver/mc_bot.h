#pragma once

#include <vector>

#include <torch/torch.h>

#include "engine/game_context.h"
#include "engine/game_state.h"
#include "engine/rng.h"
#include "model/pro_yams_net.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// MCConfig — parameters for MC rollout-enhanced play.
// ---------------------------------------------------------------------------
struct MCConfig {
    double time_budget_ms          = 2000.0;  // Total time per turn
    int    max_candidates          = 5;       // Top-K actions to evaluate
    int    min_rollouts_per_candidate = 10;   // Minimum before comparing
    int    rollout_batch_size      = 4;       // Rollouts per check-time cycle
};

// ---------------------------------------------------------------------------
// CandidateAction — one candidate action being evaluated by rollouts.
// ---------------------------------------------------------------------------
struct CandidateAction {
    bool      is_placement  = false; // true = place, false = hold+reroll
    Placement placement     = {};    // Only valid if is_placement
    int8_t    score         = 0;     // Only valid if is_placement
    uint8_t   hold_mask     = 0;     // Only valid if !is_placement
    double    nn_ev         = 0.0;   // NN-based expected value
    double    win_count     = 0.0;   // Accumulated wins from rollouts
    int       rollout_count = 0;     // Number of rollouts completed

    double empirical_wr() const {
        return rollout_count > 0 ? win_count / rollout_count : 0.0;
    }

    // Bayesian-smoothed win rate: blends the NN EV (as a prior worth
    // `prior_weight` pseudo-rollouts) with the empirical rollouts. When
    // rollouts are few, this stays anchored to the NN EV; as rollouts grow,
    // empirical evidence takes over.
    double bayesian_wr(double prior_weight = 10.0) const {
        return (nn_ev * prior_weight + win_count) /
               (prior_weight + rollout_count);
    }
};

// ---------------------------------------------------------------------------
// MC time allocation — budget management across roll phases within a turn.
// ---------------------------------------------------------------------------
struct MCTimeAllocation {
    double total_budget_ms;
    double elapsed_ms;

    /// Budget for the current decision point given how many rolls remain.
    /// Earlier decisions get more time (they're more impactful).
    double budget_for_current_decision(int rolls_left) const {
        double remaining = total_budget_ms - elapsed_ms;
        if (rolls_left == 0) return remaining;
        // Weight: current decision gets proportionally more.
        constexpr double weights[] = {1.0, 0.6, 0.4};  // rolls_left = 0, 1, 2
        double total_weight = 0;
        for (int r = 0; r <= rolls_left; ++r) total_weight += weights[r];
        return remaining * weights[rolls_left] / total_weight;
    }
};

// ---------------------------------------------------------------------------
// simulate_rollout — play one complete game from an afterstate to completion.
//
// Both players use the NN+solver bot with greedy play (no exploration).
//
// @param board       The afterstate board (after the candidate action)
// @param mc_player   Which player the MC bot controls (for result interpretation)
// @param model       NN model for inference
// @param device      GPU/CPU device
// @param tables      Precomputed tables
// @param rng         Random engine (each rollout should use a unique seed)
// @return 1.0 if mc_player wins, 0.0 if they lose, 0.5 if draw
// ---------------------------------------------------------------------------
double simulate_rollout(const BoardState& board,
                        const GameContext& ctx,
                        int mc_player,
                        ProYamsNet& model, torch::Device device,
                        const PrecomputedTables& tables,
                        SolverBuffers& shared_buffers,
                        std::vector<float>& shared_tensor_buffer,
                        RNG& rng);

// ---------------------------------------------------------------------------
// select_top_candidates — extract the top-K candidate actions from a solved
// state. Includes both placement and hold+reroll options.
//
// @param state       Current game state
// @param ctx         Game context
// @param tables      Precomputed tables
// @param buffers     Solver buffers (must have evs[] filled and V0/V1 computed)
// @param max_k       Maximum candidates to return
// @param out         Output array (must hold max_k elements)
// @return Number of candidates written to out
// ---------------------------------------------------------------------------
int select_top_candidates(const GameState& state, const GameContext& ctx,
                          const PrecomputedTables& tables,
                          const SolverBuffers& buffers,
                          int max_k, CandidateAction* out);

// ---------------------------------------------------------------------------
// can_terminate_early — check if the leading candidate is statistically
// significantly better than all others (Wilson score 95% CI).
// ---------------------------------------------------------------------------
bool can_terminate_early(const CandidateAction* candidates, int count,
                         int min_rollouts);

// ---------------------------------------------------------------------------
// mc_play_turn — play one complete turn using MC rollout-enhanced decisions.
//
// @param state     Game state (modified: hold/placement applied)
// @param ctx       Game context (modified)
// @param model     NN model
// @param device    GPU device
// @param tables    Precomputed tables
// @param mc_config MC configuration
// @param rng       Random engine
// ---------------------------------------------------------------------------
void mc_play_turn(GameState& state, GameContext& ctx,
                  ProYamsNet& model, torch::Device device,
                  const PrecomputedTables& tables,
                  const MCConfig& mc_config, RNG& rng);
