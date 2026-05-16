#include "self_play/training_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "engine/game_traits.h"

template <typename Traits>
int extract_training_samples(const GameInstanceT<Traits>& game,
                              TDMode td_mode, double td_lambda,
                              bool use_margin, double margin_scale,
                              bool use_pbrs,
                              TrainingSampleT<Traits>* samples, int max_samples,
                              int exclude_player) {
    int traj_len = game.trajectory_length;

    const double terminal_p0 = use_margin
        ? std::tanh(static_cast<double>(game.final_duel_margin) / margin_scale)
        : game.result;

    auto terminal_for = [&](int8_t my_player) -> double {
        const bool same_team_as_p0 = Traits::are_teammates(my_player, 0);
        if (use_margin) {
            return same_team_as_p0 ? terminal_p0 : -terminal_p0;
        }
        return same_team_as_p0 ? terminal_p0 : 1.0 - terminal_p0;
    };

    auto flip = [&](double v) -> double {
        return use_margin ? -v : 1.0 - v;
    };

    // Precompute team-0-perspective TD(λ) targets in a single O(N) backward
    // pass. For each trajectory step i, g_team0[i] is the TD(λ) target as
    // seen from team 0's perspective. Per-step targets for the actual
    // my_player are obtained by flipping when my_player is on team 1.
    //
    // Recursion (no per-step reward, no discount):
    //   g[N-1] = terminal_p0
    //   For i < N-1:
    //     If step i+1 is exploratory   ⇒ g[i] = v_{i+1}           (trace cut)
    //     Else                          ⇒ g[i] = (1-λ)·v_{i+1} + λ·g[i+1]
    //   where v_j = future.value if are_teammates(player_j, 0) else flip(value).
    //
    // The weights in the original forward loop summed to 1, so the previous
    // sum/weight_sum division is unnecessary here — the recursion preserves
    // that property by construction.
    std::vector<double> g_team0;
    if (td_mode == TDMode::kTDLambda && traj_len > 0) {
        g_team0.resize(traj_len);
        g_team0[traj_len - 1] = terminal_p0;
        for (int idx = traj_len - 2; idx >= 0; --idx) {
            const TrajectoryStepT<Traits>& next = game.trajectory[idx + 1];
            const double v_next = Traits::are_teammates(next.player, 0)
                ? next.value
                : flip(next.value);
            if (next.is_exploratory) {
                g_team0[idx] = v_next;
            } else {
                g_team0[idx] = (1.0 - td_lambda) * v_next
                             + td_lambda * g_team0[idx + 1];
            }
        }
    }

    int out = 0;
    for (int i = 0; i < traj_len && out < max_samples; ++i) {
        const TrajectoryStepT<Traits>& step = game.trajectory[i];
        int8_t my_player = step.player;
        // In 2v2, exclude_player names one seat but the whole team plays the
        // past opponent — skip every trajectory step belonging to that team.
        // In 1v1, are_teammates(p, q) reduces to (p == q), preserving prior
        // behaviour exactly.
        if (exclude_player >= 0 &&
            Traits::are_teammates(my_player, exclude_player)) continue;

        std::memcpy(samples[out].state, step.tensor,
                    Traits::kTensorSize * sizeof(float));

        double target = 0.0;

        switch (td_mode) {
        case TDMode::kMC:
            target = terminal_for(my_player);
            break;

        case TDMode::kTD0:
            if (i + 1 < game.trajectory_length) {
                const TrajectoryStepT<Traits>& next = game.trajectory[i + 1];
                target = Traits::are_teammates(next.player, my_player)
                    ? next.value
                    : flip(next.value);
            } else {
                target = terminal_for(my_player);
            }
            break;

        case TDMode::kTDLambda: {
            const double g = g_team0[i];
            target = Traits::are_teammates(my_player, 0) ? g : flip(g);
            break;
        }
        }

        if (std::isnan(target)) target = use_margin ? 0.0 : 0.5;

        if (use_pbrs) {
            target += step.pbrs_reward;
        }

        if (use_margin) {
            samples[out].target = std::max(-1.0, std::min(1.0, target));
        } else {
            samples[out].target = std::max(0.0, std::min(1.0, target));
        }
        ++out;
    }

    return out;
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------

template int extract_training_samples<Yams1v1>(const GameInstanceT<Yams1v1>&,
                                               TDMode, double, bool, double, bool,
                                               TrainingSampleT<Yams1v1>*, int, int);
template int extract_training_samples<Yams2v2>(const GameInstanceT<Yams2v2>&,
                                               TDMode, double, bool, double, bool,
                                               TrainingSampleT<Yams2v2>*, int, int);
