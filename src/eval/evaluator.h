#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "engine/game_context.h"
#include "engine/game_state.h"
#include "engine/game_traits.h"
#include "engine/rng.h"
#include "heuristic/heuristic_bot.h"  // HeuristicVersion
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

// Default number of games played concurrently by the parallel eval helpers.
inline constexpr int kDefaultEvalThreads = 16;

// ---------------------------------------------------------------------------
// run_evaluation — play num_games between NN and heuristic bot.
//
// Games alternate which player (1v1) or team (2v2) the NN controls. The model
// is set to eval mode. Games are independent and run concurrently across
// `num_threads` worker threads (capped to num_games); per-game seeds are fixed
// by game index, so results are identical regardless of thread count.
// The heuristic opponent's version is randomised per game via
// random_heuristic_version().
// ---------------------------------------------------------------------------
template <typename Traits>
EvalResult run_evaluation(ProYamsNet& model, torch::Device device,
                           const PrecomputedTables& tables,
                           int num_games, uint64_t base_seed,
                           int num_threads = kDefaultEvalThreads);

// ---------------------------------------------------------------------------
// run_evaluation_vs<Traits> — NN versus one specific heuristic version.
//
// Used by the distillation loop to measure the student's win rate against
// a fixed reference heuristic (default V2). The heuristic version stays
// constant across all games, so the result is a clean, deterministic
// estimate of "NN strength vs <ref>" rather than the mixed-bag average
// produced by run_evaluation.
//
// `EvalResult.nn_*` fields refer to the NN candidate (same as run_evaluation).
// ---------------------------------------------------------------------------
template <typename Traits>
EvalResult run_evaluation_vs(ProYamsNet& model, torch::Device device,
                              const PrecomputedTables& tables,
                              HeuristicVersion heuristic_version,
                              int num_games, uint64_t base_seed,
                              int num_threads = kDefaultEvalThreads);

// ---------------------------------------------------------------------------
// run_heuristic_vs_heuristic<Traits> — two heuristic bots play against each
// other for num_games. Used by distillation to cache the teacher's win
// rate against the reference heuristic at startup (the teacher is fixed,
// so we measure it once).
//
// `EvalResult.nn_*` fields refer to the *candidate* side (i.e. the
// candidate's wins / wr_as_p0 / wr_as_p1) — same shape as the NN-eval
// path, so downstream code can treat both EvalResults the same way.
// V_n-vs-V_n collapses to ~50% by symmetry; the function runs it anyway
// to keep the code path uniform.
// ---------------------------------------------------------------------------
template <typename Traits>
EvalResult run_heuristic_vs_heuristic(const PrecomputedTables& tables,
                                       HeuristicVersion candidate_version,
                                       HeuristicVersion reference_version,
                                       int num_games, uint64_t base_seed,
                                       int num_threads = kDefaultEvalThreads);

// 1v1 convenience overloads (existing 1v1-only call sites can keep working
// without sprinkling <Yams1v1> everywhere).
inline EvalResult run_evaluation(ProYamsNet& model, torch::Device device,
                                  const PrecomputedTables& tables,
                                  int num_games, uint64_t base_seed,
                                  int num_threads = kDefaultEvalThreads) {
    return run_evaluation<Yams1v1>(model, device, tables, num_games, base_seed,
                                    num_threads);
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
