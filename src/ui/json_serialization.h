#pragma once

#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_traits.h"
#include "model/model_config.h"
#include "ui/game_session.h"

// ---------------------------------------------------------------------------
// Column and row name tables (used by serialization and the REST API).
// ---------------------------------------------------------------------------
inline const char* column_name(int col) {
    static const char* names[] = {"down", "free", "up", "mid", "turbo", "updown"};
    return (col >= 0 && col < 6) ? names[col] : "unknown";
}

inline const char* row_name(int row) {
    static const char* names[] = {
        "1s", "2s", "3s", "4s", "5s", "6s",
        "SS", "LS", "FH", "K", "STR", "U8", "Y"};
    return (row >= 0 && row < 13) ? names[row] : "unknown";
}

// ---------------------------------------------------------------------------
// Serialize the full game state of a session to JSON.
//
// The output includes a top-level "game_variant" field ("1v1" or "2v2") and
// a "num_players" count so the frontend can switch layouts. Player boards
// are emitted as player0..playerN-1, where N = Traits::kNumPlayers.
// ---------------------------------------------------------------------------
template <typename Traits>
nlohmann::json game_state_to_json(const GameSessionT<Traits>& session);

// ---------------------------------------------------------------------------
// Serialize placement options to JSON. Variant-agnostic.
// ---------------------------------------------------------------------------
nlohmann::json options_to_json(
    const std::vector<std::pair<Placement, int>>& options,
    bool can_reroll);
