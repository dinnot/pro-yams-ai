#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "engine/game_context.h"
#include "engine/game_state.h"
#include "engine/game_traits.h"
#include "engine/rng.h"
#include "model/pro_yams_net.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// EvalResult — aggregated results from one evaluation run.
//
// In 1v1, "nn_wins_as_p0" / "games_as_p0" mean: NN plays seat P0.
// In 2v2, the same fields mean: NN plays Team 0 (seats P0 + P2). The "as_p1"
// fields mean: NN plays Team 1 (seats P1 + P3).
// ---------------------------------------------------------------------------
struct EvalResult {
    int    total_games     = 0;
    int    nn_wins         = 0;
    int    heuristic_wins  = 0;
    int    draws           = 0;
    int    nn_wins_as_p0   = 0;
    int    nn_wins_as_p1   = 0;
    int    games_as_p0     = 0;
    int    games_as_p1     = 0;
    double avg_duel_margin = 0.0;

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
// (or team's) perspective: 1.0 win, 0.0 loss, 0.5 draw.
//
// Templated on Traits so the team-aware logic comes from Traits::are_teammates:
// the player is mapped to its team (via `are_teammates(player, 0)`), and the
// returned value is the win probability for that team.
// ---------------------------------------------------------------------------
template <typename Traits>
double get_game_result_for_player(const GameStateT<Traits>& state,
                                   const GameContextT<Traits>& ctx,
                                   int player);

// ---------------------------------------------------------------------------
// nn_play_turn<Traits> — play one complete turn using synchronous NN inference.
//
// Tensor buffer is heap-allocated by the caller
// (kMaxAfterstateRequests * Traits::kTensorSize floats).
// ---------------------------------------------------------------------------
template <typename Traits>
void nn_play_turn(ProYamsNet& model, torch::Device device,
                   GameStateT<Traits>& state, GameContextT<Traits>& ctx,
                   const PrecomputedTables& tables,
                   SolverBuffers& buffers,
                   std::vector<float>& tensor_buffer,
                   const SolverConfig& config, RNG& rng);

// ---------------------------------------------------------------------------
// play_eval_game — play one game between NN and heuristic bot.
//
// 1v1: `nn_player` is the NN's seat (0 or 1).
// 2v2: `nn_player` is the *anchor* seat of the NN's team (0 = Team 0, 1 = Team 1).
//      The NN controls both seats on that team via Traits::are_teammates.
//
// @param out_duel_margin Output: raw duel point margin from NN's perspective.
// @return 1.0 if NN/NN-team wins, 0.0 if heuristic wins, 0.5 if draw.
// ---------------------------------------------------------------------------
template <typename Traits>
double play_eval_game(ProYamsNet& model, torch::Device device,
                       const PrecomputedTables& tables,
                       int nn_player, RNG& rng,
                       int& out_duel_margin);

// ---------------------------------------------------------------------------
// run_evaluation — play num_games between NN and heuristic bot.
//
// Games alternate which player (1v1) or team (2v2) the NN controls. Runs
// synchronously on the calling thread. The model is set to eval mode.
// ---------------------------------------------------------------------------
template <typename Traits>
EvalResult run_evaluation(ProYamsNet& model, torch::Device device,
                           const PrecomputedTables& tables,
                           int num_games, uint64_t base_seed);

// 1v1 convenience overloads (existing 1v1-only call sites can keep working
// without sprinkling <Yams1v1> everywhere).
inline EvalResult run_evaluation(ProYamsNet& model, torch::Device device,
                                  const PrecomputedTables& tables,
                                  int num_games, uint64_t base_seed) {
    return run_evaluation<Yams1v1>(model, device, tables, num_games, base_seed);
}

inline double play_eval_game(ProYamsNet& model, torch::Device device,
                              const PrecomputedTables& tables,
                              int nn_player, RNG& rng, int& out_duel_margin) {
    return play_eval_game<Yams1v1>(model, device, tables,
                                    nn_player, rng, out_duel_margin);
}

inline double get_game_result_for_player(const GameStateT<Yams1v1>& state,
                                          const GameContextT<Yams1v1>& ctx,
                                          int player) {
    return get_game_result_for_player<Yams1v1>(state, ctx, player);
}
