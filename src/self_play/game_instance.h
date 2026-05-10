#pragma once

#include <cstdint>
#include "engine/constants.h"
#include "engine/game_state.h"
#include "engine/game_context.h"
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
// TrajectoryStep — one recorded afterstate per placement in the game.
// ---------------------------------------------------------------------------
struct TrajectoryStep {
    float   tensor[kTensorSize];  // Afterstate tensor at time of placement
    double  value;                 // V(s) from solver for the chosen placement
    int8_t  player;               // Which player made this placement (0 or 1)
    double  pbrs_reward = 0.0;    // Potential-based reward shaping bonus for this step
    bool    is_exploratory = false; // True if an off-policy action was taken
};

// ---------------------------------------------------------------------------
// GameInstance — all per-game state, buffers, and trajectory.
//
// Only ever accessed by one thread at a time — no locking on per-game data.
// Memory: ~1.6MB (tensor_buffer) + ~500KB (trajectory) + ~13KB (solver
// buffers) + ~200 bytes (game state) ≈ 2.1MB total per game.
// ---------------------------------------------------------------------------
struct GameInstance {
    /// Default constructor: seeds RNG with 0 (caller should reseed before use).
    GameInstance() : rng(0) {}

    // === Game state ===
    GameState   state;
    GameContext ctx;
    RNG         rng;
    int         game_id;
    GamePhase   phase;

    bool        is_debug_game = false;
    std::string debug_log_path;

    // === Solver buffers (reused across turns) ===
    SolverBuffers solver_buffers;

    // === Tensor buffer for afterstate evaluation ===
    // Removed: Tensors are now generated directly into the central PyTorch batch (Zero-Copy).

    // === Trajectory for training data ===
    // One entry per placement over the entire game (max 156 placements).
    static constexpr int kMaxTrajectorySteps = kNumPlayers * kNumColumns * kNumRows;
    TrajectoryStep trajectory[kMaxTrajectorySteps];
    int            trajectory_length;

    // === Game result ===
    double result;  // 1.0 = player 0 wins, 0.0 = player 1 wins, 0.5 = draw
    int    final_duel_margin = 0;  // Raw P0 duel points at terminal state

    double current_turn_start_ev = 0.0;
    bool   current_turn_is_exploratory = false;

    // === Past-opponent rotation ===
    // When use_past_opponent is true, the player whose seat == past_opponent_player
    // has its inference requests routed to the opponent BatchManager (loaded from
    // an older checkpoint). The other seat plays the current model.
    bool use_past_opponent      = false;
    int  past_opponent_player   = -1;  // 0 or 1 when active; -1 otherwise

    // === Per-game heuristic variant ===
    // Randomly chosen at game start so training exposes the NN to all variants.
    HeuristicVersion heuristic_version = HeuristicVersion::V2;
};
