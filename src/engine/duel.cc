#include "engine/duel.h"
#include "engine/game_traits.h"

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

// ---------------------------------------------------------------------------
// compute_duel — per-pairing crush multiplier and clean-column bonus value.
//
// The math for one pairing (t0_player vs t1_player) in one column:
//   1. raw0 = sum of cells + upper_section_bonus(upper_sum)  // per player
//   2. crush_t0 = crush_multiplier(raw0, raw1)               // 1..5 directional
//      crush_t1 = crush_multiplier(raw1, raw0)
//      active_crush = max(crush_t0, crush_t1)                // only one can be >1
//   3. bonus_value = (active_crush > 1) ? 100 : 200          // applies to both sides
//   4. adj0 = raw0 + (is_clean[t0p] ? bonus_value : 0)
//      adj1 = raw1 + (is_clean[t1p] ? bonus_value : 0)
//   5. pairing_points = (adj0 - adj1) * active_crush * coefficient[col]
//
// Total Team 0 margin = sum of pairing_points over every (t0p, t1p) in
// Traits::kTeam0 × Traits::kTeam1, over every column.
//
// In 1v1 this collapses to a single 1×1 pairing (0, 1) per column — bit-for-bit
// identical to the previous non-templated implementation.
// ---------------------------------------------------------------------------
template <typename Traits>
std::array<int, kNumColumns> compute_duel_columns(
    const BoardStateT<Traits>& board,
    const GameContextT<Traits>& ctx) {
    std::array<int, kNumColumns> column_points{};

    for (int col = 0; col < kNumColumns; ++col) {
        // Step 1: Raw scores per player.
        int raw_score[Traits::kNumPlayers] = {};
        for (int p = 0; p < Traits::kNumPlayers; ++p) {
            int cell_sum = 0;
            for (int row = 0; row < kNumRows; ++row) {
                int8_t v = board.cells[p][col][row];
                if (v > 0) cell_sum += v;
            }
            raw_score[p] = cell_sum + upper_section_bonus(ctx.upper_sum[p][col]);
        }

        // Step 2: Clean-column eligibility per player.
        bool is_clean[Traits::kNumPlayers];
        for (int p = 0; p < Traits::kNumPlayers; ++p) {
            is_clean[p] = (ctx.upper_sum[p][col] >= 60) &&
                          (!ctx.lower_has_scratch[p][col]);
        }

        const int coeff = board.coefficients[col];

        // Step 3: Per-pairing crush and clean-bonus value, summed into Team 0.
        int col_points = 0;
        for (int i = 0; i < Traits::kPlayersPerTeam; ++i) {
            for (int j = 0; j < Traits::kPlayersPerTeam; ++j) {
                const int t0p = Traits::kTeam0[i];
                const int t1p = Traits::kTeam1[j];

                const int raw0 = raw_score[t0p];
                const int raw1 = raw_score[t1p];

                const int crush_t0 = crush_multiplier(raw0, raw1);
                const int crush_t1 = crush_multiplier(raw1, raw0);
                const int active_crush = (crush_t0 > crush_t1) ? crush_t0 : crush_t1;

                // Clean-column bonus value is per-pairing: same value for both
                // sides within this pairing, but the same player can see a
                // different bonus value in their other pairing.
                const int bonus_value = (active_crush > 1) ? 100 : 200;

                const int adj0 = raw0 + (is_clean[t0p] ? bonus_value : 0);
                const int adj1 = raw1 + (is_clean[t1p] ? bonus_value : 0);
                const int diff = adj0 - adj1;

                col_points += diff * active_crush * coeff;
            }
        }
        column_points[col] = col_points;
    }

    return column_points;
}

template <typename Traits>
int compute_duel(const BoardStateT<Traits>& board,
                 const GameContextT<Traits>& ctx) {
    const std::array<int, kNumColumns> cols = compute_duel_columns(board, ctx);
    int total_duel_points = 0;
    for (int c = 0; c < kNumColumns; ++c) total_duel_points += cols[c];
    return total_duel_points;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template int compute_duel<Yams1v1>(const BoardStateT<Yams1v1>&,
                                   const GameContextT<Yams1v1>&);
template int compute_duel<Yams2v2>(const BoardStateT<Yams2v2>&,
                                   const GameContextT<Yams2v2>&);

template std::array<int, kNumColumns> compute_duel_columns<Yams1v1>(
    const BoardStateT<Yams1v1>&, const GameContextT<Yams1v1>&);
template std::array<int, kNumColumns> compute_duel_columns<Yams2v2>(
    const BoardStateT<Yams2v2>&, const GameContextT<Yams2v2>&);
