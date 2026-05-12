#pragma once

#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_state.h"
#include "engine/rng.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// Heuristic versions.
//   V1 — score × column coefficient (greedy, no DP).
//   V2 — DP-driven expected global duel margin (uses dp_tables).
//   V3 — V2 baseline plus four hard-coded strategic rules:
//        R1: aim for >= 150 raw points per column (dropped on lost columns)
//        R2: be ahead on Up/Down; block the opponent on high-coeff Up/Down
//        R3: amplify clean-bonus value on high-coeff columns
//        R4: focus on columns with high probability of clean bonus
// ---------------------------------------------------------------------------
enum class HeuristicVersion {
    V1  = 1,
    V2  = 2,
    V3  = 3,
    // V4..V15 — research-derived variants, each a fixed ResearchConfig.
    // Win rates measured head-to-head vs V2 over N≥1500 paired games
    // (see scripts/heuristic_sweep notes).
    V4  = 4,   // smooth + early-high-coeff(1.0)         — ~56% vs V2
    V5  = 5,   // smooth + early-high-coeff(1.5)         — ~55%
    V6  = 6,   // smooth + coeff²(0.20)                  — ~57%
    V7  = 7,   // smooth + coeff²(0.10) + turbo-avoid    — ~57.4%
    V8  = 8,   // smooth + coeff²(0.20) + turbo-avoid    — ~57.1%
    V9  = 9,   // smooth + coeff²(0.50) + turbo-avoid    — ~57%
    V10 = 10,  // smooth + ehc(0.5) + csq(0.12)          — ~56.5%
    V11 = 11,  // smooth + ehc(1.0) + csq(0.05)          — ~56.7%
    V12 = 12,  // smooth + ehc(0.5) + csq(0.20)          — ~55.5%
    V13 = 13,  // smooth + csq(0.15) + turbo + R3(.10)   — ~57.6% (BEST)
    V14 = 14,  // V13 + upper-bonus protection penalty   — TBD
    V15 = 15,  // V13 + aggressive upper-bonus penalty   — TBD
    V16 = 16,  // V13 (smooth) + Win Odds Evaluation
    V17 = 17,  // V14 + Win Odds Evaluation
};

// ---------------------------------------------------------------------------
// V1 evaluator — score × column coefficient.
//
// Captures the insight that higher scores in higher-coefficient columns are
// better, without any strategic reasoning.
// ---------------------------------------------------------------------------
void heuristic_evaluate(const BoardState& board, const GameContext& ctx,
                        const AfterstateRequest* requests, int request_count,
                        double* evs);

// ---------------------------------------------------------------------------
// V2 evaluator — DP-driven expected global duel margin.
//
// For each candidate placement, simulate it on a clone, then sum across all
// 6 columns the expected (E_me - E_opp) * coefficient * crush_multiplier,
// including a clean-column bonus contribution. Returns absolute expected
// duel points (can be negative). Heavy negative penalty applied when the
// candidate is a voluntary scratch (score == 0).
// ---------------------------------------------------------------------------
void heuristic_evaluate_v2(const BoardState& base_board, const GameContext& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables);

// ---------------------------------------------------------------------------
// V3 evaluator — V2 baseline plus rule-based strategic adjustments.
//
// Starts from heuristic_evaluate_v2's expected_global_margin and adds four
// additive bonus/penalty terms (rules R1..R4 above). Rule weights are
// deliberately modest so V2's expected-value signal still dominates and the
// rules act as nudges/tiebreakers.
// ---------------------------------------------------------------------------

// Tunable weights for V3's rule layer. Defaults are set in heuristic_bot.cc;
// callers may override via set_v3_weights() (e.g. tuning sweeps).
struct V3Weights {
    double w_r1_per_point;     // Rule 1 — penalty per missing point toward 150
    double w_r2_block_high;    // Rule 2 — block bonus on high-coeff Up/Down upper section
    double w_r3_high_coeff;    // Rule 3 — extra clean weight on high-coeff columns
    double w_r4_near_clean;    // Rule 4 — boost above clean-probability threshold
    double r4_threshold;       // P_clean above this triggers Rule 4
};

const V3Weights& v3_weights();
void set_v3_weights(const V3Weights& w);

void heuristic_evaluate_v3(const BoardState& base_board, const GameContext& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables);

// ---------------------------------------------------------------------------
// Research evaluator — a configurable V2-style evaluator with knobs for
// structural variations (T allocation, crush rounding, clean nonlinearity,
// scratch penalty, variance penalty, opponent-aware terms). Used by the
// benchmark sweep to discover configurations that beat V2; winning configs
// are promoted to numbered HeuristicVersion entries.
// ---------------------------------------------------------------------------

enum class TStrategy : int {
    V2_DEFAULT     = 0,  // T = max(empty, empty + (78 - filled) / 6)
    EMPTY_ONLY     = 1,  // T = empty
    EMPTY_PLUS_1   = 2,  // T = empty + 1
    EMPTY_PLUS_2   = 3,  // T = empty + 2
    EMPTY_DOUBLE   = 4,  // T = empty * 2
    PROPORTIONAL   = 5,  // T = floor((78-filled) * empty / total_empty), min empty
};

enum class CrushMode : int {
    ROUND_INT      = 0,  // V2: round float E to int, then crush_multiplier
    FLOOR_INT      = 1,  // floor E to int (more conservative for small E)
    FLOAT_SMOOTH   = 2,  // continuous crush (linear interpolation)
};

struct ResearchConfig {
    TStrategy t_me  = TStrategy::V2_DEFAULT;
    TStrategy t_opp = TStrategy::V2_DEFAULT;
    CrushMode crush = CrushMode::ROUND_INT;

    // Use fast Gaussian CDF to map margins to win probability [0, 1].
    bool   output_win_odds = false;

    // Clean-bonus exponent: bonus_val * pow(P_clean, clean_power)
    double clean_power = 1.0;

    // Voluntary scratch penalty: when score == 0 and the row's golden_max is
    // also 0 (no one has placed there yet, so the scratch is voluntary in a
    // weak sense), subtract scratch_penalty * row_max * coeff.
    double scratch_penalty = 0.0;

    // Variance penalty: subtract variance_penalty * coeff for placements in
    // very high-variance rows (Yams, U8) when alternatives exist.
    double variance_penalty = 0.0;

    // Opponent-aware bonus: when E_opp > E_me by a margin, push harder by
    // multiplying our column contribution by (1 + opp_aware_factor).
    double opp_aware_factor = 0.0;

    // Bonus for placements that consume a high-coefficient cell early.
    // Effectively rewards finishing high-coeff columns sooner.
    double early_high_coeff_bonus = 0.0;

    // Variant of early_high_coeff_bonus that fades as the game progresses
    // (multiplied by (78 - filled_me) / 78).
    double early_progressive_bonus = 0.0;

    // Bonus per cell still empty in the candidate's column, scaled by coeff^2.
    // Amplifies focus on the very highest-coefficient columns.
    double coeff_sq_bonus = 0.0;

    // Bonus for placements in columns with more cells already filled
    // (rewards finishing columns rather than starting new ones).
    double completion_bonus = 0.0;

    // Bonus per dominant column where we have more positive cells than opp.
    double dominance_bonus = 0.0;

    // Push toward upper-section sum >= 100 (worth +500 bonus). Reward
    // proportional to (current_upper_sum / 100) * coeff for the candidate's
    // column when the column has empty upper rows.
    double upper100_bonus = 0.0;

    // Penalty for voluntary scratches in high-coeff (>=14) columns only.
    double hc_scratch_penalty = 0.0;

    // Bonus for consuming dice toward filling large-sum upper cells.
    double upper_priority_bonus = 0.0;

    // Coeff^2 bonus restricted to columns with coeff >= 14.
    double coeff_sq_high_only = 0.0;

    // Penalty for placements in the Turbo column (col 4) early in the game,
    // discouraging Turbo use until necessary.
    double turbo_avoidance = 0.0;

    // Bonus for placements that complete a column entirely (last empty cell).
    double last_cell_bonus = 0.0;

    // Penalty for upper-section placements that decrease the chances of
    // reaching the upper bonus (60+) on columns with coefficient >= 12,
    // excluding the Down column. The penalty is proportional to how far
    // below par (3 × face_value) the placed score falls.
    double upper_bonus_penalty = 0.0;

    // Multiplier on the "smooth" crush-multiplier formula (1.0 = default).
    // Larger values steepen the curve; smaller values flatten it.
    double crush_smoothness = 1.0;

    // 1-ply opponent lookahead: for each candidate, subtract the opponent's
    // best response evaluation from V2's perspective.
    bool   lookahead_1ply = false;
    double lookahead_weight = 1.0;

    // Optional V3-style rule layer on top.
    bool   use_v3_rules = false;
    V3Weights v3 = {0, 0, 0, 0, 0.5};
};

void heuristic_evaluate_research(const BoardState& base_board, const GameContext& base_ctx,
                                 const AfterstateRequest* requests, int request_count,
                                 double* evs, const PrecomputedTables& tables,
                                 const ResearchConfig& cfg);

// Play one turn using the research evaluator with the given config.
void heuristic_play_turn_research(GameState& state, GameContext& ctx,
                                  const PrecomputedTables& tables,
                                  SolverBuffers& buffers, RNG& rng,
                                  const ResearchConfig& cfg);

// Returns the ResearchConfig for a numbered V4..V15 version.
// Throws std::invalid_argument for V1/V2/V3 (which use their own evaluators).
const ResearchConfig& get_research_config_for(HeuristicVersion v);

// ---------------------------------------------------------------------------
// Play one complete turn using the heuristic bot.
//
// Calls solver_get_requests → heuristic_evaluate(_v2) → solver_resolve in a
// loop until a placement is made. Makes hold/reroll decisions, then applies
// the chosen placement and advances to the next player's turn.
//
// On entry:  dice are already rolled (start_turn has been called).
// On exit:   the placement has been applied; next player's dice are rolled.
// ---------------------------------------------------------------------------
void heuristic_play_turn(GameState& state, GameContext& ctx,
                         const PrecomputedTables& tables,
                         SolverBuffers& buffers, RNG& rng,
                         HeuristicVersion version = HeuristicVersion::V2);

// ---------------------------------------------------------------------------
// Play a complete game using the heuristic bot for both players.
//
// Returns the raw duel score from player 0's perspective (see compute_duel).
// ---------------------------------------------------------------------------
int play_heuristic_game(RNG& rng, const PrecomputedTables& tables,
                        HeuristicVersion version = HeuristicVersion::V2);

// ---------------------------------------------------------------------------
// Pick a random HeuristicVersion uniformly from all available versions.
// ---------------------------------------------------------------------------
HeuristicVersion random_heuristic_version(RNG& rng);
