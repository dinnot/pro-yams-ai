#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "model/pro_yams_net.h"
#include "solver/precomputed_tables.h"
#include "ui/game_session.h"

// ---------------------------------------------------------------------------
// SessionEntry — wraps a GameSession with a per-session mutex so that
// operations on one session never block another session.
// ---------------------------------------------------------------------------
struct SessionEntry {
    std::unique_ptr<GameSession> session;
    std::mutex mutex;  // protects this session's state
};

// ---------------------------------------------------------------------------
// SessionManager — thread-safe manager for active game sessions.
//
// Locking strategy:
//   map_mutex_ — protects the sessions_ map (insert/find/erase). Held very
//     briefly, never during bot computation.
//   SessionEntry::mutex — protects individual session state. Held during
//     game-state mutations (bot turns, placements, holds). A slow MC bot turn
//     on session A does not block session B.
// ---------------------------------------------------------------------------
class SessionManager {
public:
    /// `nn_model` may be nullptr when no checkpoint is loaded (NN unavailable).
    SessionManager(const PrecomputedTables& tables,
                   ProYamsNet* nn_model,
                   torch::Device device);

    /// Create a new game session. Returns its ID.
    /// debug_mode: when true, bots record hold/placement evaluations per turn.
    int create_session(PlayerType p0_type, PlayerType p1_type, uint64_t seed,
                       bool debug_mode = false);

    /// Get a copy of the current session state (safe to call from any thread).
    /// Returns false if the session does not exist.
    bool get_session_copy(int session_id, GameSession& out) const;

    /// Advance one bot turn. No-op if it's a human turn or game is over.
    /// Returns the TurnRecord for what happened (empty record if no advance).
    GameSession::TurnRecord advance_turn(int session_id);

    /// Play all remaining bot turns until the game ends or a human turn is
    /// reached. Thread-safe.
    void play_to_completion(int session_id);

    /// Human action: hold specified dice and reroll.
    /// Returns false if the session doesn't exist, game is over, or can't reroll.
    bool human_hold(int session_id, uint8_t hold_mask);

    /// Human action: place score in the given cell.
    /// Returns false if session doesn't exist, game over, not human's turn,
    /// or placement is illegal.
    bool human_place(int session_id, int column, int row);

    /// Legal placements for the current human turn, with scores.
    /// Each pair is (Placement, score). Returns empty if not a human turn.
    /// Also returns whether a reroll is possible (atomically with the options).
    std::vector<std::pair<Placement, int>> get_human_options(int session_id,
                                                              bool& out_can_reroll) const;

    /// Remove a session.
    void remove_session(int session_id);

    /// Whether the NN model is available.
    bool has_nn() const { return nn_model_ != nullptr; }

    /// Compute the 986-feature observation tensor for the session's current
    /// board state from the given player's perspective. Also runs NN
    /// inference if a model is loaded and writes the win-probability into
    /// out_nn_value (only valid when out_has_nn is set).
    /// Returns false if the session does not exist.
    bool compute_current_tensor(int session_id, int player,
                                 std::vector<float>& out_tensor,
                                 float& out_nn_value, bool& out_has_nn) const;

private:
    // Get a shared_ptr to a session entry. Returns nullptr if not found.
    // Briefly locks map_mutex_.
    std::shared_ptr<SessionEntry> get_entry(int session_id) const;

    // Internal helpers (must be called with the session's mutex held).
    void play_heuristic_turn(GameSession& session);
    void play_nn_turn(GameSession& session);
    void play_mc_turn(GameSession& session);

    // Play one bot turn for the given session. Returns true if game is now over.
    bool play_one_bot_turn(GameSession& session);

    // If nn_model_ is available, evaluate the current board state from
    // record.player's perspective and store the result in record.
    // Must be called after perform_placement has updated session state.
    void eval_and_store_board_nn(GameSession& session, GameSession::TurnRecord& record);

    // Shared state (read-only after construction — no mutex needed)
    const PrecomputedTables& tables_;
    ProYamsNet*              nn_model_;
    torch::Device            device_;

    // Session map, protected by map_mutex_.
    std::map<int, std::shared_ptr<SessionEntry>> sessions_;
    int next_session_id_ = 1;
    mutable std::mutex map_mutex_;
};
