#include "engine/duel.h"

#include <cassert>

int upper_section_bonus(int sum) {
    if (sum >= 100) return 500;
    if (sum >= 90)  return 200;
    if (sum >= 80)  return 100;
    if (sum >= 70)  return 50;
    if (sum >= 60)  return 30;
    return 0;
}

int crush_multiplier(int my_raw, int opp_raw) {
    if (opp_raw == 0 && my_raw > 0) return 5;
    if (opp_raw > 0 && my_raw >= 5 * opp_raw) return 5;
    if (opp_raw > 0 && my_raw >= 4 * opp_raw) return 4;
    if (opp_raw > 0 && my_raw >= 3 * opp_raw) return 3;
    if (opp_raw > 0 && my_raw >= 2 * opp_raw) return 2;
    return 1;
}

int compute_duel(const BoardState& board, const GameContext& ctx) {
    int total_duel_points = 0;

    for (int col = 0; col < kNumColumns; ++col) {
        // Step 1: Raw scores per player
        int raw_score[kNumPlayers] = {};
        for (int p = 0; p < kNumPlayers; ++p) {
            int cell_sum = 0;
            for (int row = 0; row < kNumRows; ++row) {
                int8_t v = board.cells[p][col][row];
                if (v > 0) cell_sum += v;
            }
            raw_score[p] = cell_sum + upper_section_bonus(ctx.upper_sum[p][col]);
        }

        // Step 2: Crush multipliers (both directions)
        int crush0 = crush_multiplier(raw_score[0], raw_score[1]);  // player 0 crushing player 1
        int crush1 = crush_multiplier(raw_score[1], raw_score[0]);  // player 1 crushing player 0
        int active_crush = (crush0 > crush1) ? crush0 : crush1;

        // Step 3: Clean column bonus
        // Clean = upper_sum >= 60 AND no lower section scratch
        bool is_clean[kNumPlayers];
        for (int p = 0; p < kNumPlayers; ++p) {
            is_clean[p] = (ctx.upper_sum[p][col] >= 60) && (!ctx.lower_has_scratch[p][col]);
        }
        // Bonus value depends on whether any crush is active
        int clean_bonus = (active_crush > 1) ? 100 : 200;

        int adjusted[kNumPlayers];
        for (int p = 0; p < kNumPlayers; ++p) {
            adjusted[p] = raw_score[p] + (is_clean[p] ? clean_bonus : 0);
        }

        // Step 4: Duel points for this column
        int difference = adjusted[0] - adjusted[1];
        int coeff = board.coefficients[col];
        total_duel_points += difference * active_crush * coeff;
    }

    return total_duel_points;
}
