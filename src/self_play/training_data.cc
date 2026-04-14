#include "self_play/training_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

int extract_training_samples(const GameInstance& game,
                              TDMode td_mode, double td_lambda,
                              bool use_margin, double margin_scale,
                              TrainingSample* samples, int max_samples) {
    int n = std::min(game.trajectory_length, max_samples);

    // Terminal target from the perspective of player 0.
    const double terminal_p0 = use_margin
        ? std::tanh(static_cast<double>(game.final_duel_margin) / margin_scale)
        : game.result;

    auto terminal_for = [&](int8_t my_player) -> double {
        if (use_margin) {
            double m = (my_player == 0) ? terminal_p0 : -terminal_p0;
            return m;
        }
        return (my_player == 0) ? terminal_p0 : 1.0 - terminal_p0;
    };

    auto flip = [&](double v) -> double {
        return use_margin ? -v : 1.0 - v;
    };

    for (int i = 0; i < n; ++i) {
        const TrajectoryStep& step = game.trajectory[i];
        int8_t my_player = step.player;

        std::memcpy(samples[i].state, step.tensor, kTensorSize * sizeof(float));

        double target = 0.0;

        switch (td_mode) {
        case TDMode::kMC:
            target = terminal_for(my_player);
            break;

        case TDMode::kTD0:
            if (i + 1 < game.trajectory_length) {
                target = flip(game.trajectory[i + 1].value);
            } else {
                target = terminal_for(my_player);
            }
            break;

        case TDMode::kTDLambda: {
            double accumulated = 0.0;
            double weight_sum  = 0.0;
            double lam_power   = 1.0;

            for (int j = i + 1; j < game.trajectory_length; ++j) {
                const TrajectoryStep& future = game.trajectory[j];
                double bootstrap = (future.player != my_player)
                    ? flip(future.value)
                    : future.value;
                double w = (1.0 - td_lambda) * lam_power;
                accumulated += w * bootstrap;
                weight_sum  += w;
                lam_power   *= td_lambda;
            }

            double terminal = terminal_for(my_player);
            accumulated += lam_power * terminal;
            weight_sum  += lam_power;

            target = (weight_sum > 0.0) ? accumulated / weight_sum : terminal;
            break;
        }
        }

        if (std::isnan(target)) target = use_margin ? 0.0 : 0.5;
        if (use_margin) {
            samples[i].target = std::max(-1.0, std::min(1.0, target));
        } else {
            samples[i].target = std::max(0.0, std::min(1.0, target));
        }
    }

    return n;
}
