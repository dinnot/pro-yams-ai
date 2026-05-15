#pragma once

#include <cstdint>
#include "engine/constants.h"
#include "engine/game_state.h"
#include "engine/game_context.h"
#include "engine/game_traits.h"
#include "engine/rng.h"
#include "engine/tensor.h"
#include "heuristic/heuristic_bot.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// GamePhase — state machine for an in-progress self-play game.
// ---------------------------------------------------------------------------
enum class GamePhase {
    kNeedRequests,     // Worker: generate afterstate requests + tensors
    kWaitingInference, // In pending queue, waiting for GPU inference
    kNeedResolve,      // Worker: EVs ready, run solver_resolve
    kCompleted         // Game finished, ready for data collection
};

// ---------------------------------------------------------------------------
// TrajectoryStepT — one recorded afterstate per placement in the game.
// Templated so tensor[] sizes correctly for the variant.
// ---------------------------------------------------------------------------
template <typename Traits>
struct TrajectoryStepT {
    float   tensor[Traits::kTensorSize];  // Afterstate tensor at time of placement
    double  value;                         // V(s) from solver for the chosen placement
    int8_t  player;                        // Which player made this placement
    double  pbrs_reward = 0.0;             // Potential-based reward shaping bonus
    bool    is_exploratory = false;        // True if an off-policy action was taken
};

using TrajectoryStep   = TrajectoryStepT<Yams1v1>;
using TrajectoryStep2v2 = TrajectoryStepT<Yams2v2>;

// ---------------------------------------------------------------------------
// GameInstanceT — all per-game state, buffers, and trajectory.
//
// Only ever accessed by one thread at a time — no locking on per-game data.
// ---------------------------------------------------------------------------
template <typename Traits>
struct GameInstanceT {
    /// Default constructor: seeds RNG with 0 (caller should reseed before use).
    GameInstanceT() : rng(0) {}

    // === Game state ===
    GameStateT<Traits>   state;
    GameContextT<Traits> ctx;
    RNG                  rng;
    int                  game_id;
    GamePhase            phase;

    bool        is_debug_game = false;
    std::string debug_log_path;

    // === Solver buffers (reused across turns) ===
    SolverBuffers solver_buffers;

    // === Trajectory for training data ===
    // One entry per placement over the entire game.
    static constexpr int kMaxTrajectorySteps = Traits::kTotalCells;
    TrajectoryStepT<Traits> trajectory[kMaxTrajectorySteps];
    int                     trajectory_length;

    // === Game result ===
    double result;  // 1.0 = player 0 wins, 0.0 = player 1 wins, 0.5 = draw
                    // In 2v2: 1.0 = Team 0 wins, 0.0 = Team 1 wins.
    int    final_duel_margin = 0;  // Raw P0 (1v1) / Team-0 (2v2) duel points

    double current_turn_start_ev = 0.0;
    bool   current_turn_is_exploratory = false;

    // === Past-opponent rotation ===
    bool use_past_opponent      = false;
    int  past_opponent_player   = -1;  // 0 or 1 when active (1v1); -1 otherwise

    // === Per-game heuristic variant ===
    HeuristicVersion heuristic_version = HeuristicVersion::V2;
};

using GameInstance    = GameInstanceT<Yams1v1>;
using GameInstance2v2 = GameInstanceT<Yams2v2>;
