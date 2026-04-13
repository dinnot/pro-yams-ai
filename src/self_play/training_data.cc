#include "self_play/training_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

int extract_training_samples(const GameInstance& game,
                              TDMode td_mode, double td_lambda,
                              TrainingSample* samples, int max_samples) {
    int n = std::min(game.trajectory_length, max_samples);

    for (int i = 0; i < n; ++i) {
        const TrajectoryStep& step = game.trajectory[i];
        int8_t my_player = step.player;

        // Copy the afterstate tensor.
        std::memcpy(samples[i].state, step.tensor, kTensorSize * sizeof(float));

        double target = 0.0;

        switch (td_mode) {
        case TDMode::kMC:
            // Actual game outcome from this player's perspective.
            target = (my_player == 0) ? game.result : 1.0 - game.result;
            break;

        case TDMode::kTD0:
            if (i + 1 < game.trajectory_length) {
                // Bootstrap: next afterstate value is from the opponent's
                // perspective (players alternate), so flip it.
                target = 1.0 - game.trajectory[i + 1].value;
            } else {
                // Last step — use actual game outcome.
                target = (my_player == 0) ? game.result : 1.0 - game.result;
            }
            break;

        case TDMode::kTDLambda: {
            // Weighted blend of all future bootstrapped values + terminal.
            // weight(j) = (1 - λ) * λ^(j-i-1)  for j = i+1 .. T-1
            // weight(terminal) = λ^(T-i-1)
            double accumulated = 0.0;
            double weight_sum  = 0.0;
            double lam_power   = 1.0;  // λ^(j-i-1) starting at j=i+1

            for (int j = i + 1; j < game.trajectory_length; ++j) {
                const TrajectoryStep& future = game.trajectory[j];
                double bootstrap;
                if (future.player != my_player) {
                    // Opponent's value — flip perspective.
                    bootstrap = 1.0 - future.value;
                } else {
                    bootstrap = future.value;
                }
                double w = (1.0 - td_lambda) * lam_power;
                accumulated += w * bootstrap;
                weight_sum  += w;
                lam_power   *= td_lambda;
            }

            // Terminal value gets the remaining weight (λ^(T-i-1)).
            double terminal = (my_player == 0) ? game.result : 1.0 - game.result;
            accumulated += lam_power * terminal;
            weight_sum  += lam_power;

            target = (weight_sum > 0.0) ? accumulated / weight_sum : terminal;
            break;
        }
        }

        // Clamp target strictly to [0.0, 1.0] to prevent PyTorch BCE Loss 
        // from crashing due to floating-point precision drift.
        if (std::isnan(target)) target = 0.5;
        samples[i].target = std::max(0.0, std::min(1.0, target));
    }

    return n;
}
