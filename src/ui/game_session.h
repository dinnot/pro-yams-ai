#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_state.h"
#include "engine/game_traits.h"
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
    kHeuristicV16,
    kHeuristicV17,
    kNNSolver,
    kMCRollout
};

// ---------------------------------------------------------------------------
// GameSessionT<Traits> — all state for one active UI game.
//
// Templated on the game variant so 1v1 sessions have 2 player_types and 2
// sheets, 2v2 sessions have 4 of each.
// ---------------------------------------------------------------------------
template <typename Traits>
struct GameSessionT {
    int session_id = 0;

    GameStateT<Traits>   state;
    GameContextT<Traits> ctx;
    SolverBuffers        buffers;
    RNG                  rng{0};

    /// One player_types slot per seat. Default-initialized to V2 heuristic.
    PlayerType player_types[Traits::kNumPlayers] = {};

    bool   game_over = false;
    double result    = 0.0;

    // Heap-allocated tensor buffer for NN inference
    // (kMaxAfterstateRequests * Traits::kTensorSize floats).
    std::vector<float> tensor_buffer;

    // ---------------------------------------------------------------------------
    // Debug evaluation data — populated only when debug_mode == true.
    // ---------------------------------------------------------------------------
    struct HoldCandidateEval {
        uint8_t hold_mask;
        float   expected_value;
    };

    struct PlacementCandidateEval {
        Placement placement;
        int8_t    score;
        float     eval_value;
    };

    struct TurnRecord {
        int    player  = 0;
        int8_t initial_dice[kNumDice] = {};
        std::vector<uint8_t>                       hold_masks;
        std::vector<std::array<int8_t, kNumDice>>  dice_after_hold;
        Placement placement = {0, 0};
        int8_t    score     = 0;
        bool      is_forced_reroll = false;

        std::vector<std::vector<HoldCandidateEval>> hold_evals;
        std::vector<PlacementCandidateEval>          placement_evals;
        float board_nn_value     = -1.0f;
        bool  has_board_nn_value = false;
    };

    std::vector<TurnRecord> history;

    TurnRecord current_turn;
    bool waiting_for_human = false;

    bool debug_mode = false;

    // ---------------------------------------------------------------------------
    // Shared multiplayer (two humans on one team vs NN) — set only for the
    // "Play with a friend" matchmade 2v2 game. Two browsers drive this one
    // session, each owning a single human seat.
    //   shared_multiplayer : enables heartbeat-based disconnect → NN takeover.
    //   seat_last_seen_ms  : last heartbeat per seat (epoch ms); 0 = never seen.
    //   seat_nn_takeover   : a human seat that timed out and was flipped to NN
    //                        (so the surviving player can be told their teammate
    //                        left). Indexed by seat.
    // ---------------------------------------------------------------------------
    bool    shared_multiplayer = false;
    int64_t seat_last_seen_ms[Traits::kNumPlayers] = {};
    bool    seat_nn_takeover[Traits::kNumPlayers]  = {};
    // A human seat that has gone silent past the grace period and is a candidate
    // for NN takeover. Surfaced to the surviving teammate, who decides whether to
    // switch it to the AI — the flip is never automatic. Clears if the player
    // resumes (heartbeats again).
    bool    seat_disconnected[Traits::kNumPlayers] = {};

    // Tentative ("preview") hold mask the player on the move has selected but
    // not yet committed via a reroll — pushed live by the active client so a
    // shared-game teammate can watch the hold pattern form. Only meaningful
    // while waiting_for_human and live_hold_player == current_player; reset to
    // an inert player (-1) on each commit so a stale mask is never shown.
    int     live_hold_player = -1;
    uint8_t live_hold_mask   = 0;

    float current_board_nn_value = 0.0f;
    bool  has_current_board_nn   = false;
};

// Backward-compat aliases.
using GameSession    = GameSessionT<Yams1v1>;
using GameSession2v2 = GameSessionT<Yams2v2>;
