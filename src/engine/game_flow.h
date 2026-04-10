#pragma once

#include <cstdint>
#include "engine/game_state.h"
#include "engine/game_context.h"
#include "engine/rng.h"

// ---------------------------------------------------------------------------
// Game Flow — turn management for Pro Yams
//
// A turn proceeds as follows:
//   1. start_turn()     — rolls all 5 dice, sets rolls_left = 2
//   2. Optionally:        perform_reroll() up to 2 times
//                         OR place early in a non-Turbo cell
//   3. perform_placement() — places score, switches player, starts next turn
//
// Turbo column rule: can place in kColTurbo only when rolls_left > 0
// (i.e., the player is limited to 2 rolls max for Turbo placements).
// ---------------------------------------------------------------------------

/// Initialize a full game: board + context + rolls the first player's dice.
void init_game(GameState& gs, GameContext& ctx, RNG& rng);

/// Roll all 5 dice at the start of a turn. Sets rolls_left = 2.
void start_turn(GameState& gs, RNG& rng);

/// Reroll dice not held. hold_mask is a 5-bit bitmask (bit i = keep dice[i]).
/// Decrements rolls_left. No-op if rolls_left == 0.
void perform_reroll(GameState& gs, uint8_t hold_mask, RNG& rng);

/// Returns true if the player can reroll.
/// False when rolls_left == 0, or when rolls_left == 1 and only Turbo cells remain.
bool can_reroll(const GameState& gs, const GameContext& ctx);

/// Returns the set of legal placements for the current dice state:
///   - rolls_left >  0: legal_all      (early placement; Turbo available)
///   - rolls_left == 0: legal_no_turbo (must place; Turbo max roll limit exceeded)
const LegalPlacementCache& get_legal_placements(const GameState& gs, const GameContext& ctx);

/// Apply placement for the current player.
/// Uses the caller-supplied score (0 = scratch), calls apply_placement,
/// switches player, and (if game not over) calls start_turn for the next player.
int perform_placement(GameState& gs, GameContext& ctx, int column, int row, int score, RNG& rng);

/// Compute the final duel result. Only valid when is_terminal(gs.board).
int get_game_result(const GameState& gs, const GameContext& ctx);
