#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "engine/game_context.h"
#include "engine/game_state.h"
#include "engine/rng.h"
#include "model/pro_yams_net.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// EvalResult — aggregated results from one evaluation run.
// ---------------------------------------------------------------------------
struct EvalResult {
    int    total_games     = 0;
    int    nn_wins         = 0;
    int    heuristic_wins  = 0;
    int    draws           = 0;
    int    nn_wins_as_p0   = 0;   // NN wins when playing as player 0
    int    nn_wins_as_p1   = 0;   // NN wins when playing as player 1
    int    games_as_p0     = 0;   // Total games where NN was player 0
    int    games_as_p1     = 0;   // Total games where NN was player 1
    double avg_duel_margin = 0.0; // Average duel point margin across all games

    double nn_win_rate() const {
        return total_games > 0
            ? static_cast<double>(nn_wins) / total_games : 0.0;
    }
    double nn_win_rate_as_p0() const {
        return games_as_p0 > 0
            ? static_cast<double>(nn_wins_as_p0) / games_as_p0 : 0.0;
    }
    double nn_win_rate_as_p1() const {
        return games_as_p1 > 0
            ? static_cast<double>(nn_wins_as_p1) / games_as_p1 : 0.0;
    }
};

// ---------------------------------------------------------------------------
// get_game_result_for_player — convert duel outcome to a specific player's
// perspective win probability: 1.0 win, 0.0 loss, 0.5 draw.
// ---------------------------------------------------------------------------
double get_game_result_for_player(const GameState& state,
                                   const GameContext& ctx,
                                   int player);

// ---------------------------------------------------------------------------
// nn_play_turn — play one complete turn using synchronous NN inference.
//
// Tensor buffer is heap-allocated by the caller (kMaxAfterstates * kTensorSize
// floats) to avoid stack overflow from large stack allocation.
// ---------------------------------------------------------------------------
void nn_play_turn(ProYamsNet& model, torch::Device device,
                   GameState& state, GameContext& ctx,
                   const PrecomputedTables& tables,
                   SolverBuffers& buffers,
                   std::vector<float>& tensor_buffer,
                   const SolverConfig& config, RNG& rng);

// ---------------------------------------------------------------------------
// play_eval_game — play one game between NN and heuristic bot.
//
// @param nn_player      Which player the NN controls (0 or 1)
// @param out_duel_margin Output: raw duel point margin from NN's perspective
// @return 1.0 if NN wins, 0.0 if heuristic wins, 0.5 if draw
// ---------------------------------------------------------------------------
double play_eval_game(ProYamsNet& model, torch::Device device,
                       const PrecomputedTables& tables,
                       int nn_player, RNG& rng,
                       int& out_duel_margin);

// ---------------------------------------------------------------------------
// run_evaluation — play num_games between NN and heuristic bot.
//
// Games alternate which player the NN controls. Runs synchronously on the
// calling thread (no async queue). The model is set to eval mode internally.
//
// @param base_seed Seed for reproducibility (step number works well)
// ---------------------------------------------------------------------------
EvalResult run_evaluation(ProYamsNet& model, torch::Device device,
                           const PrecomputedTables& tables,
                           int num_games, uint64_t base_seed);
