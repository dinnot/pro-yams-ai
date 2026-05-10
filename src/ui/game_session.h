#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_state.h"
#include "engine/rng.h"
#include "engine/tensor.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// PlayerType — what controls each side in a session.
// ---------------------------------------------------------------------------
enum class PlayerType {
    kHuman,
    kHeuristic,    // V1: greedy score × coefficient
    kHeuristicV2,  // V2: DP-driven expected duel margin
    kHeuristicV3,  // V3: V2 baseline + strategic rules (column aim, blocking, clean focus)
    kHeuristicV4,
    kHeuristicV5,
    kHeuristicV6,
    kHeuristicV7,
    kHeuristicV8,
    kHeuristicV9,
    kHeuristicV10,
    kHeuristicV11,
    kHeuristicV12,
    kHeuristicV13,
    kHeuristicV14,
    kHeuristicV15,
    kNNSolver,
    kMCRollout
};

// ---------------------------------------------------------------------------
// GameSession — all state for one active UI game.
// ---------------------------------------------------------------------------
struct GameSession {
    int session_id = 0;

    GameState   state;
    GameContext ctx;
    SolverBuffers buffers;
    RNG         rng{0};

    PlayerType player_types[2] = {PlayerType::kHeuristic, PlayerType::kHeuristic};

    bool   game_over = false;
    double result    = 0.0;  // set when game ends

    // Heap-allocated tensor buffer for NN inference (kMaxAfterstateRequests * kTensorSize floats)
    std::vector<float> tensor_buffer;

    // ---------------------------------------------------------------------------
    // Debug evaluation data — populated only when debug_mode == true.
    // ---------------------------------------------------------------------------

    /// One hold mask candidate and its expected value (win probability).
    struct HoldCandidateEval {
        uint8_t hold_mask;
        float   expected_value;
    };

    /// One placement candidate and the bot's evaluation of it.
    struct PlacementCandidateEval {
        Placement placement;
        int8_t    score;        // Actual score achievable now
        float     eval_value;   // Bot's raw afterstate EV (win probability)
    };

    // Turn history
    struct TurnRecord {
        int    player  = 0;
        int8_t initial_dice[kNumDice] = {};
        std::vector<uint8_t>             hold_masks;
        std::vector<std::array<int8_t, kNumDice>> dice_after_hold;
        Placement placement = {0, 0};
        int8_t    score     = 0;
        bool      is_forced_reroll = false;  // true when hold was forced (Turbo edge case)

        // Debug info (populated only when GameSession::debug_mode == true).
        // hold_evals[i] = candidates considered at reroll step i, sorted desc by expected_value.
        std::vector<std::vector<HoldCandidateEval>> hold_evals;
        // All placements the bot evaluated, sorted desc by eval_value.
        std::vector<PlacementCandidateEval> placement_evals;
        // NN evaluation of the resulting board after this turn, from placing
        // player's perspective. Populated when debug_mode == true and a NN
        // model is available (regardless of player type).
        float board_nn_value     = -1.0f;
        bool  has_board_nn_value = false;
    };
    std::vector<TurnRecord> history;

    // Current turn state (used while a human is playing)
    TurnRecord current_turn;
    bool waiting_for_human = false;

    // When true, bots record hold/placement evaluations into each TurnRecord.
    bool debug_mode = false;

    // Most-recent NN evaluation of the board state (from the last placing
    // player's perspective). Populated whenever debug_mode && nn is available.
    float current_board_nn_value = 0.0f;
    bool  has_current_board_nn   = false;
};
