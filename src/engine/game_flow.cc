#include "engine/game_flow.h"

#include "engine/board_init.h"
#include "engine/duel.h"
#include "engine/game_traits.h"
#include "engine/placement.h"
#include "engine/scoring.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

template <typename Traits>
static void roll_all_dice(GameStateT<Traits>& gs, RNG& rng) {
    for (int i = 0; i < kNumDice; ++i)
        gs.dice[i] = static_cast<int8_t>(rng.uniform_int(1, kNumDieSides));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

template <typename Traits>
void init_game(GameStateT<Traits>& gs, GameContextT<Traits>& ctx, RNG& rng) {
    init_board<Traits>(gs.board, rng);
    init_context<Traits>(ctx, gs.board);
    start_turn<Traits>(gs, rng);
}

template <typename Traits>
void start_turn(GameStateT<Traits>& gs, RNG& rng) {
    gs.rolls_left = 2;
    roll_all_dice<Traits>(gs, rng);
    sort_dice(gs.dice);
}

template <typename Traits>
void perform_reroll(GameStateT<Traits>& gs, uint8_t hold_mask, RNG& rng) {
    if (gs.rolls_left == 0) return;
    for (int i = 0; i < kNumDice; ++i) {
        if (!((hold_mask >> i) & 1))
            gs.dice[i] = static_cast<int8_t>(rng.uniform_int(1, kNumDieSides));
    }
    sort_dice(gs.dice);
    --gs.rolls_left;
}

template <typename Traits>
bool can_reroll(const GameStateT<Traits>& gs, const GameContextT<Traits>& ctx) {
    if (gs.rolls_left <= 0) return false;
    if (gs.rolls_left == 1 &&
        ctx.non_turbo_cells_remaining[gs.board.current_player] == 0) {
        return false;
    }
    return true;
}

template <typename Traits>
const LegalPlacementCache& get_legal_placements(const GameStateT<Traits>& gs,
                                                const GameContextT<Traits>& ctx) {
    int p = gs.board.current_player;
    if (gs.rolls_left > 0 || ctx.legal_no_turbo[p].count == 0)
        return ctx.legal_all[p];
    return ctx.legal_no_turbo[p];
}

template <typename Traits>
int perform_placement(GameStateT<Traits>& gs, GameContextT<Traits>& ctx,
                      int column, int row, RNG& rng) {
    int p = gs.board.current_player;

    // The Engine acts as the absolute authority again!
    int score = calculate_score<Traits>(row, gs.dice, p, column, gs.board, ctx);

    apply_placement<Traits>(p, column, row, score, gs.board, ctx);
    if (!is_terminal<Traits>(gs.board)) {
        // Clockwise rotation: A → B → C → D → A ... (1v1: 0 ↔ 1).
        gs.board.current_player = static_cast<int8_t>((p + 1) % Traits::kNumPlayers);
        start_turn<Traits>(gs, rng);
    }
    return score;
}

template <typename Traits>
int get_game_result(const GameStateT<Traits>& gs, const GameContextT<Traits>& ctx) {
    return compute_duel(gs.board, ctx);
}

// ---------------------------------------------------------------------------
// Explicit instantiations
//
// Most functions are instantiated for both Yams1v1 and Yams2v2. The exception
// is get_game_result, which calls compute_duel — currently 1v1-only. The 2v2
// instantiation lands in Task 3 once duel.cc is templatized.
// ---------------------------------------------------------------------------

template void init_game<Yams1v1>(GameStateT<Yams1v1>&, GameContextT<Yams1v1>&, RNG&);
template void init_game<Yams2v2>(GameStateT<Yams2v2>&, GameContextT<Yams2v2>&, RNG&);
template void start_turn<Yams1v1>(GameStateT<Yams1v1>&, RNG&);
template void start_turn<Yams2v2>(GameStateT<Yams2v2>&, RNG&);
template void perform_reroll<Yams1v1>(GameStateT<Yams1v1>&, uint8_t, RNG&);
template void perform_reroll<Yams2v2>(GameStateT<Yams2v2>&, uint8_t, RNG&);
template bool can_reroll<Yams1v1>(const GameStateT<Yams1v1>&, const GameContextT<Yams1v1>&);
template bool can_reroll<Yams2v2>(const GameStateT<Yams2v2>&, const GameContextT<Yams2v2>&);
template const LegalPlacementCache& get_legal_placements<Yams1v1>(
    const GameStateT<Yams1v1>&, const GameContextT<Yams1v1>&);
template const LegalPlacementCache& get_legal_placements<Yams2v2>(
    const GameStateT<Yams2v2>&, const GameContextT<Yams2v2>&);
template int perform_placement<Yams1v1>(GameStateT<Yams1v1>&, GameContextT<Yams1v1>&, int, int, RNG&);
template int perform_placement<Yams2v2>(GameStateT<Yams2v2>&, GameContextT<Yams2v2>&, int, int, RNG&);

template int get_game_result<Yams1v1>(const GameStateT<Yams1v1>&, const GameContextT<Yams1v1>&);
template int get_game_result<Yams2v2>(const GameStateT<Yams2v2>&, const GameContextT<Yams2v2>&);
