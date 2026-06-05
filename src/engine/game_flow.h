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
//   3. perform_placement() — places score, advances to next player, starts next turn
//
// Turn order is clockwise modulo Traits::kNumPlayers.
//   - 1v1: 0, 1, 0, 1, ...
//   - 2v2: 0, 1, 2, 3, 0, ... (A → B → C → D)
//
// Turbo column rule: can place in kColTurbo only when rolls_left > 0
// (i.e., the player is limited to 2 rolls max for Turbo placements).
// ---------------------------------------------------------------------------

/// Initialize a full game: board + context + rolls the first player's dice.
template <typename Traits>
void init_game(GameStateT<Traits>& gs, GameContextT<Traits>& ctx, RNG& rng);

/// Roll all 5 dice at the start of a turn. Sets rolls_left = 2.
template <typename Traits>
void start_turn(GameStateT<Traits>& gs, RNG& rng);

/// Reroll dice not held. hold_mask is a 5-bit bitmask (bit i = keep dice[i]).
/// Decrements rolls_left. No-op if rolls_left == 0.
template <typename Traits>
void perform_reroll(GameStateT<Traits>& gs, uint8_t hold_mask, RNG& rng);

/// Returns true if the "Lucky Yams" bonus is active for the current dice state:
/// the rule is enabled, this is the player's first roll (rolls_left == 2), and
/// all five dice show the same face. While active, the player may place the
/// maximum legal score in any cell (see calculate_yams_bonus_score). Re-rolling
/// forfeits the bonus (rolls_left drops below 2).
template <typename Traits>
bool yams_bonus_active(const GameStateT<Traits>& gs);

/// Returns true if the player can reroll.
/// False when rolls_left == 0, or when rolls_left == 1 and only Turbo cells remain.
template <typename Traits>
bool can_reroll(const GameStateT<Traits>& gs, const GameContextT<Traits>& ctx);

/// Returns the set of legal placements for the current dice state:
///   - rolls_left >  0: legal_all      (early placement; Turbo available)
///   - rolls_left == 0: legal_no_turbo (must place; Turbo max roll limit exceeded)
template <typename Traits>
const LegalPlacementCache& get_legal_placements(const GameStateT<Traits>& gs,
                                                const GameContextT<Traits>& ctx);

/// Apply placement for the current player.
/// Calculates the score from gs.dice, calls apply_placement, advances player,
/// and (if game not over) calls start_turn for the next player.
/// Returns the score placed.
template <typename Traits>
int perform_placement(GameStateT<Traits>& gs, GameContextT<Traits>& ctx,
                      int column, int row, RNG& rng);

/// Compute the final duel result. Only valid when is_terminal(gs.board).
/// Currently 1v1-only (Yams1v1). 2v2 instantiation lands in Task 3 once
/// compute_duel is templatized.
template <typename Traits>
int get_game_result(const GameStateT<Traits>& gs, const GameContextT<Traits>& ctx);
