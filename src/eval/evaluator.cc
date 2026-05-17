#include "eval/evaluator.h"

#include <cassert>
#include <cstring>

#include "engine/duel.h"
#include "engine/game_flow.h"
#include "engine/game_traits.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "heuristic/heuristic_bot.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// get_game_result_for_player<Traits>
// ---------------------------------------------------------------------------

template <typename Traits>
double get_game_result_for_player(const GameStateT<Traits>& state,
                                   const GameContextT<Traits>& ctx,
                                   int player) {
    int duel = compute_duel<Traits>(state.board, ctx);
    // compute_duel returns the Team-0 margin. Flip for any player NOT on Team 0.
    if (!Traits::are_teammates(player, 0)) duel = -duel;
    if (duel > 0) return 1.0;
    if (duel < 0) return 0.0;
    return 0.5;
}

// ---------------------------------------------------------------------------
// nn_play_turn<Traits>
// ---------------------------------------------------------------------------

template <typename Traits>
void nn_play_turn(ProYamsNet& model, torch::Device device,
                   GameStateT<Traits>& state, GameContextT<Traits>& ctx,
                   const PrecomputedTables& tables,
                   SolverBuffers& buffers,
                   std::vector<float>& tensor_buffer,
                   const SolverConfig& config, RNG& rng) {

    buffers.dp_computed = false;
    buffers.evs_blended = false;

    solver_get_requests<Traits>(state, ctx, tables, buffers);

    if (buffers.request_count == 0) {
        while (can_reroll<Traits>(state, ctx)) {
            perform_reroll<Traits>(state, 0, rng);
        }
        auto& all = ctx.legal_all[state.board.current_player];
        if (all.count > 0) {
            perform_placement<Traits>(state, ctx,
                                      all.placements[0].column,
                                      all.placements[0].row, rng);
        }
        return;
    }

    generate_tensor_batch<Traits>(state.board, ctx,
                                   static_cast<int>(state.board.current_player),
                                   buffers.requests, buffers.request_count,
                                   tables, tensor_buffer.data());

    {
        torch::NoGradGuard no_grad;
        auto input = torch::from_blob(
            tensor_buffer.data(),
            {buffers.request_count, Traits::kTensorSize},
            torch::kFloat32
        ).to(device);
        auto output = model.forward(input).to(torch::kCPU).contiguous();
        const float* ptr = output.template data_ptr<float>();
        for (int i = 0; i < buffers.request_count; ++i) {
            buffers.evs[i] = static_cast<double>(ptr[i]);
        }
    }

    while (true) {
        SolverResult result = solver_resolve<Traits>(state, ctx, tables, buffers,
                                                    config, rng);

        if (result.should_place) {
            perform_placement<Traits>(state, ctx,
                                      result.placement.column, result.placement.row, rng);
            return;
        } else {
            if (!can_reroll<Traits>(state, ctx)) {
                int current_id = get_dice_state_id(state.dice, tables);
                int16_t req_idx = buffers.stop_request_idx[current_id];
                if (req_idx < 0) req_idx = 0;
                perform_placement<Traits>(state, ctx,
                                          buffers.requests[req_idx].placement.column,
                                          buffers.requests[req_idx].placement.row, rng);
                return;
            }
            perform_reroll<Traits>(state, result.hold_mask, rng);
        }
    }
}

// ---------------------------------------------------------------------------
// play_eval_game<Traits>
// ---------------------------------------------------------------------------

template <typename Traits>
double play_eval_game(ProYamsNet& model, torch::Device device,
                       const PrecomputedTables& tables,
                       int nn_player, RNG& rng,
                       int& out_duel_margin) {
    GameStateT<Traits>   state;
    GameContextT<Traits> ctx;
    SolverBuffers buffers{};
    SolverConfig greedy_cfg;

    std::vector<float> tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * Traits::kTensorSize);

    init_game<Traits>(state, ctx, rng);

    // Pick a random heuristic variant for this game.
    const HeuristicVersion hv = random_heuristic_version(rng);

    while (!is_terminal<Traits>(state.board)) {
        int player = static_cast<int>(state.board.current_player);
        // In 1v1 are_teammates(p, q) collapses to p == q, so `is_nn_seat` is
        // equivalent to `player == nn_player`. In 2v2 it returns true for both
        // seats on the NN's team.
        const bool is_nn_seat = Traits::are_teammates(player, nn_player);
        if (is_nn_seat) {
            nn_play_turn<Traits>(model, device, state, ctx, tables,
                                  buffers, tensor_buffer, greedy_cfg, rng);
        } else {
            heuristic_play_turn<Traits>(state, ctx, tables, buffers, rng, hv);
        }
    }

    int duel = compute_duel<Traits>(state.board, ctx);
    // Convert to NN's perspective (Team-0 margin → flip iff NN is on Team 1).
    if (!Traits::are_teammates(nn_player, 0)) duel = -duel;
    out_duel_margin = duel;

    return get_game_result_for_player<Traits>(state, ctx, nn_player);
}

// ---------------------------------------------------------------------------
// run_evaluation<Traits>
// ---------------------------------------------------------------------------

template <typename Traits>
EvalResult run_evaluation(ProYamsNet& model, torch::Device device,
                           const PrecomputedTables& tables,
                           int num_games, uint64_t base_seed) {
    EvalResult result{};
    result.total_games = num_games;

    model.eval();

    double margin_sum = 0.0;

    for (int i = 0; i < num_games; ++i) {
        RNG rng(base_seed + static_cast<uint64_t>(i));
        // nn_player alternates: i even → NN anchors seat 0 (Team 0 in 2v2);
        // i odd → NN anchors seat 1 (Team 1 in 2v2).
        int nn_player = i % 2;

        if (nn_player == 0) result.games_as_p0++;
        else                result.games_as_p1++;

        int duel_margin = 0;
        double outcome = play_eval_game<Traits>(model, device, tables,
                                                nn_player, rng, duel_margin);

        margin_sum += duel_margin;

        if (outcome == 1.0) {
            result.nn_wins++;
            if (nn_player == 0) result.nn_wins_as_p0++;
            else                result.nn_wins_as_p1++;
        } else if (outcome == 0.0) {
            result.heuristic_wins++;
        } else {
            result.draws++;
        }
    }

    result.avg_duel_margin = num_games > 0
        ? margin_sum / num_games : 0.0;

    return result;
}

// ---------------------------------------------------------------------------
// play_eval_game_vs_impl<Traits>
//
// Internal helper that plays one NN-vs-fixed-heuristic game. Shared between
// play_eval_game (random per-game version) and run_evaluation_vs (fixed).
// ---------------------------------------------------------------------------
namespace {
template <typename Traits>
double play_eval_game_with_version(ProYamsNet& model, torch::Device device,
                                    const PrecomputedTables& tables,
                                    int nn_player,
                                    HeuristicVersion hv,
                                    RNG& rng,
                                    int& out_duel_margin) {
    GameStateT<Traits>   state;
    GameContextT<Traits> ctx;
    SolverBuffers buffers{};
    SolverConfig greedy_cfg;

    std::vector<float> tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * Traits::kTensorSize);

    init_game<Traits>(state, ctx, rng);

    while (!is_terminal<Traits>(state.board)) {
        int player = static_cast<int>(state.board.current_player);
        const bool is_nn_seat = Traits::are_teammates(player, nn_player);
        if (is_nn_seat) {
            nn_play_turn<Traits>(model, device, state, ctx, tables,
                                  buffers, tensor_buffer, greedy_cfg, rng);
        } else {
            heuristic_play_turn<Traits>(state, ctx, tables, buffers, rng, hv);
        }
    }

    int duel = compute_duel<Traits>(state.board, ctx);
    if (!Traits::are_teammates(nn_player, 0)) duel = -duel;
    out_duel_margin = duel;
    return get_game_result_for_player<Traits>(state, ctx, nn_player);
}
}  // namespace

// ---------------------------------------------------------------------------
// run_evaluation_vs<Traits>
// ---------------------------------------------------------------------------

template <typename Traits>
EvalResult run_evaluation_vs(ProYamsNet& model, torch::Device device,
                              const PrecomputedTables& tables,
                              HeuristicVersion heuristic_version,
                              int num_games, uint64_t base_seed) {
    EvalResult result{};
    result.total_games = num_games;

    model.eval();

    double margin_sum = 0.0;
    for (int i = 0; i < num_games; ++i) {
        RNG rng(base_seed + static_cast<uint64_t>(i));
        int nn_player = i % 2;

        if (nn_player == 0) result.games_as_p0++;
        else                result.games_as_p1++;

        int duel_margin = 0;
        double outcome = play_eval_game_with_version<Traits>(
            model, device, tables, nn_player, heuristic_version, rng, duel_margin);

        margin_sum += duel_margin;

        if (outcome == 1.0) {
            result.nn_wins++;
            if (nn_player == 0) result.nn_wins_as_p0++;
            else                result.nn_wins_as_p1++;
        } else if (outcome == 0.0) {
            result.heuristic_wins++;
        } else {
            result.draws++;
        }
    }

    result.avg_duel_margin = num_games > 0 ? margin_sum / num_games : 0.0;
    return result;
}

// ---------------------------------------------------------------------------
// run_heuristic_vs_heuristic<Traits>
// ---------------------------------------------------------------------------

template <typename Traits>
EvalResult run_heuristic_vs_heuristic(const PrecomputedTables& tables,
                                       HeuristicVersion candidate_version,
                                       HeuristicVersion reference_version,
                                       int num_games, uint64_t base_seed) {
    EvalResult result{};
    result.total_games = num_games;

    double margin_sum = 0.0;
    for (int i = 0; i < num_games; ++i) {
        RNG rng(base_seed + static_cast<uint64_t>(i));
        int candidate_player = i % 2;
        if (candidate_player == 0) result.games_as_p0++;
        else                       result.games_as_p1++;

        GameStateT<Traits>   state;
        GameContextT<Traits> ctx;
        SolverBuffers buffers{};
        init_game<Traits>(state, ctx, rng);

        while (!is_terminal<Traits>(state.board)) {
            int p = static_cast<int>(state.board.current_player);
            const bool is_candidate_seat =
                Traits::are_teammates(p, candidate_player);
            HeuristicVersion hv = is_candidate_seat
                ? candidate_version : reference_version;
            heuristic_play_turn<Traits>(state, ctx, tables, buffers, rng, hv);
        }

        int duel = compute_duel<Traits>(state.board, ctx);
        if (!Traits::are_teammates(candidate_player, 0)) duel = -duel;
        margin_sum += duel;

        if (duel > 0) {
            result.nn_wins++;
            if (candidate_player == 0) result.nn_wins_as_p0++;
            else                       result.nn_wins_as_p1++;
        } else if (duel < 0) {
            result.heuristic_wins++;
        } else {
            result.draws++;
        }
    }

    result.avg_duel_margin = num_games > 0 ? margin_sum / num_games : 0.0;
    return result;
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------

template double get_game_result_for_player<Yams1v1>(const GameStateT<Yams1v1>&,
                                                    const GameContextT<Yams1v1>&,
                                                    int);
template double get_game_result_for_player<Yams2v2>(const GameStateT<Yams2v2>&,
                                                    const GameContextT<Yams2v2>&,
                                                    int);
template void nn_play_turn<Yams1v1>(ProYamsNet&, torch::Device,
                                    GameStateT<Yams1v1>&, GameContextT<Yams1v1>&,
                                    const PrecomputedTables&, SolverBuffers&,
                                    std::vector<float>&, const SolverConfig&, RNG&);
template void nn_play_turn<Yams2v2>(ProYamsNet&, torch::Device,
                                    GameStateT<Yams2v2>&, GameContextT<Yams2v2>&,
                                    const PrecomputedTables&, SolverBuffers&,
                                    std::vector<float>&, const SolverConfig&, RNG&);
template double play_eval_game<Yams1v1>(ProYamsNet&, torch::Device,
                                        const PrecomputedTables&, int, RNG&, int&);
template double play_eval_game<Yams2v2>(ProYamsNet&, torch::Device,
                                        const PrecomputedTables&, int, RNG&, int&);
template EvalResult run_evaluation<Yams1v1>(ProYamsNet&, torch::Device,
                                            const PrecomputedTables&, int, uint64_t);
template EvalResult run_evaluation<Yams2v2>(ProYamsNet&, torch::Device,
                                            const PrecomputedTables&, int, uint64_t);
template EvalResult run_evaluation_vs<Yams1v1>(ProYamsNet&, torch::Device,
                                               const PrecomputedTables&,
                                               HeuristicVersion, int, uint64_t);
template EvalResult run_evaluation_vs<Yams2v2>(ProYamsNet&, torch::Device,
                                               const PrecomputedTables&,
                                               HeuristicVersion, int, uint64_t);
template EvalResult run_heuristic_vs_heuristic<Yams1v1>(const PrecomputedTables&,
                                                        HeuristicVersion,
                                                        HeuristicVersion,
                                                        int, uint64_t);
template EvalResult run_heuristic_vs_heuristic<Yams2v2>(const PrecomputedTables&,
                                                        HeuristicVersion,
                                                        HeuristicVersion,
                                                        int, uint64_t);
