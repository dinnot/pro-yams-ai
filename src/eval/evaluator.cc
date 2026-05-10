#include "eval/evaluator.h"

#include <cassert>
#include <cstring>

#include "engine/duel.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "heuristic/heuristic_bot.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// get_game_result_for_player
// ---------------------------------------------------------------------------

double get_game_result_for_player(const GameState& state,
                                   const GameContext& ctx,
                                   int player) {
    int duel = compute_duel(state.board, ctx);
    if (player == 1) duel = -duel;
    if (duel > 0) return 1.0;
    if (duel < 0) return 0.0;
    return 0.5;
}

// ---------------------------------------------------------------------------
// nn_play_turn — synchronous NN inference for one complete turn.
// ---------------------------------------------------------------------------

void nn_play_turn(ProYamsNet& model, torch::Device device,
                   GameState& state, GameContext& ctx,
                   const PrecomputedTables& tables,
                   SolverBuffers& buffers,
                   std::vector<float>& tensor_buffer,
                   const SolverConfig& config, RNG& rng) {
                   
    buffers.dp_computed = false;
    buffers.evs_blended = false;
    
    // Step 1: get afterstate requests ONCE per turn
    solver_get_requests(state, ctx, tables, buffers);

    if (buffers.request_count == 0) {
        while (can_reroll(state, ctx)) {
            perform_reroll(state, 0, rng);
        }
        auto& all = ctx.legal_all[state.board.current_player];
        if (all.count > 0) {
            perform_placement(state, ctx, all.placements[0].column, all.placements[0].row, rng);
        }
        return; 
    }

    // Step 2: generate tensors and infer ONCE per turn
    generate_tensor_batch(state.board, ctx,
                          static_cast<int>(state.board.current_player),
                          buffers.requests, buffers.request_count,
                          tables, tensor_buffer.data());

    {
        torch::NoGradGuard no_grad;
        auto input = torch::from_blob(
            tensor_buffer.data(),
            {buffers.request_count, kTensorSize},
            torch::kFloat32
        ).to(device);
        auto output = model.forward(input).to(torch::kCPU).contiguous();
        const float* ptr = output.data_ptr<float>();
        for (int i = 0; i < buffers.request_count; ++i) {
            buffers.evs[i] = static_cast<double>(ptr[i]);
        }
    }

    // Step 3: resolve in a loop
    while (true) {
        SolverResult result = solver_resolve(state, ctx, tables, buffers,
                                              config, rng);

        if (result.should_place) {
            perform_placement(state, ctx,
                              result.placement.column, result.placement.row, rng);
            return;
        } else {
            if (!can_reroll(state, ctx)) {
                // Safety break to prevent infinite loop.
                int current_id = get_dice_state_id(state.dice, tables);
                int16_t req_idx = buffers.stop_request_idx[current_id];
                if (req_idx < 0) req_idx = 0;
                perform_placement(state, ctx, buffers.requests[req_idx].placement.column,
                                  buffers.requests[req_idx].placement.row, rng);
                return;
            }
            perform_reroll(state, result.hold_mask, rng);
        }
    }
}

// ---------------------------------------------------------------------------
// play_eval_game
// ---------------------------------------------------------------------------

double play_eval_game(ProYamsNet& model, torch::Device device,
                       const PrecomputedTables& tables,
                       int nn_player, RNG& rng,
                       int& out_duel_margin) {
    GameState    state;
    GameContext  ctx;
    SolverBuffers buffers{};
    SolverConfig greedy_cfg;

    // Tensor buffer is heap-allocated to avoid stack overflow.
    std::vector<float> tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);

    init_game(state, ctx, rng);

    // Pick a random heuristic variant for this game so the NN is evaluated
    // against a diverse mix of opponents.
    const HeuristicVersion hv = random_heuristic_version(rng);

    while (!is_terminal(state.board)) {
        int player = static_cast<int>(state.board.current_player);
        if (player == nn_player) {
            nn_play_turn(model, device, state, ctx, tables,
                         buffers, tensor_buffer, greedy_cfg, rng);
        } else {
            heuristic_play_turn(state, ctx, tables, buffers, rng, hv);
        }
    }

    int duel = compute_duel(state.board, ctx);
    // Convert to NN's perspective
    if (nn_player == 1) duel = -duel;
    out_duel_margin = duel;

    return get_game_result_for_player(state, ctx, nn_player);
}

// ---------------------------------------------------------------------------
// run_evaluation
// ---------------------------------------------------------------------------

EvalResult run_evaluation(ProYamsNet& model, torch::Device device,
                           const PrecomputedTables& tables,
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
        double outcome = play_eval_game(model, device, tables,
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
