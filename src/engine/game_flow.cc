#include "engine/game_flow.h"

#include "engine/board_init.h"
#include "engine/scoring.h"
#include "engine/placement.h"
#include "engine/duel.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void roll_all_dice(GameState& gs, RNG& rng) {
    for (int i = 0; i < kNumDice; ++i)
        gs.dice[i] = static_cast<int8_t>(rng.uniform_int(1, kNumDieSides));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void init_game(GameState& gs, GameContext& ctx, RNG& rng) {
    init_board(gs.board, rng);
    init_context(ctx, gs.board);
    start_turn(gs, rng);
}

void start_turn(GameState& gs, RNG& rng) {
    gs.rolls_left = 2;
    roll_all_dice(gs, rng);
    sort_dice(gs.dice);
}

void perform_reroll(GameState& gs, uint8_t hold_mask, RNG& rng) {
    if (gs.rolls_left == 0) return;
    for (int i = 0; i < kNumDice; ++i) {
        if (!((hold_mask >> i) & 1))
            gs.dice[i] = static_cast<int8_t>(rng.uniform_int(1, kNumDieSides));
    }
    sort_dice(gs.dice);
    --gs.rolls_left;
}

bool can_reroll(const GameState& gs, const GameContext& ctx) {
    if (gs.rolls_left <= 0) return false;
    if (gs.rolls_left == 1 &&
        ctx.non_turbo_cells_remaining[gs.board.current_player] == 0) {
        return false;
    }
    return true;
}

const LegalPlacementCache& get_legal_placements(const GameState& gs, const GameContext& ctx) {
    int p = gs.board.current_player;
    if (gs.rolls_left > 0)
        return ctx.legal_all[p];
    return ctx.legal_no_turbo[p];
}

int perform_placement(GameState& gs, GameContext& ctx, int column, int row, RNG& rng) {
    int p = gs.board.current_player;
    
    // The Engine acts as the absolute authority again!
    int score = calculate_score(row, gs.dice, p, column, gs.board, ctx);
    
    apply_placement(p, column, row, score, gs.board, ctx);
    if (!is_terminal(gs.board)) {
        gs.board.current_player = static_cast<int8_t>(1 - p);
        start_turn(gs, rng);
    }
    return score;
}

int get_game_result(const GameState& gs, const GameContext& ctx) {
    return compute_duel(gs.board, ctx);
}
