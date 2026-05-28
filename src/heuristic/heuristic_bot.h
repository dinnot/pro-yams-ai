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
//   V2 — DP-driven expected team duel margin (uses dp_tables).
//   V3 — V2 baseline plus four hard-coded strategic rules:
//        R1: aim for >= 150 raw points per column (dropped when ALL opps are
//            so scratched they can't catch up — see ResearchConfig comments).
//        R2: place high upper-section scores on high-coeff Up/Down columns
//            to raise golden_max (blocks ALL opponents simultaneously).
//        R3: amplify clean-bonus value on high-coeff columns.
//        R4: focus on columns with high probability of clean bonus.
// ---------------------------------------------------------------------------
enum class HeuristicVersion {
    V1  = 1,
    V2  = 2,
    V3  = 3,
    // V4..V17 — research-derived variants, each a fixed ResearchConfig.
    V4  = 4,
    V5  = 5,
    V6  = 6,
    V7  = 7,
    V8  = 8,
    V9  = 9,
    V10 = 10,
    V11 = 11,
    V12 = 12,
    V13 = 13,
    V14 = 14,
    V15 = 15,
    V16 = 16,
    V17 = 17,
};

// ---------------------------------------------------------------------------
// V1 evaluator — score × column coefficient. Variant-agnostic.
// ---------------------------------------------------------------------------
template <typename Traits>
void heuristic_evaluate(const BoardStateT<Traits>& board,
                        const GameContextT<Traits>& ctx,
                        const AfterstateRequest* requests, int request_count,
                        double* evs);

// ---------------------------------------------------------------------------
// V2 evaluator — DP-driven expected team duel margin.
//
// 1v1: single (P0 vs P1) pairing per column.
// 2v2: four cross-team pairings per column ((P0,P1),(P0,P3),(P2,P1),(P2,P3)),
//      crush multiplier and clean-bonus value computed independently per
//      pairing. The returned EV is the margin from the active player's TEAM
//      perspective (positive = my team is ahead).
// ---------------------------------------------------------------------------
template <typename Traits>
void heuristic_evaluate_v2(const BoardStateT<Traits>& base_board,
                           const GameContextT<Traits>& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables);

// ---------------------------------------------------------------------------
// V3 evaluator — V2 baseline plus rule-based strategic adjustments.
//
// Mutable V3 weights — defaults chosen so V2's signal still dominates.
// ---------------------------------------------------------------------------
struct V3Weights {
    double w_r1_per_point;     // Rule 1 — penalty per missing point toward 150
    double w_r2_block_high;    // Rule 2 — block bonus on high-coeff Up/Down upper section
    double w_r3_high_coeff;    // Rule 3 — extra clean weight on high-coeff columns
    double w_r4_near_clean;    // Rule 4 — boost above clean-probability threshold
    double r4_threshold;       // P_clean above this triggers Rule 4
};

const V3Weights& v3_weights();
void set_v3_weights(const V3Weights& w);

template <typename Traits>
void heuristic_evaluate_v3(const BoardStateT<Traits>& base_board,
                           const GameContextT<Traits>& base_ctx,
                           const AfterstateRequest* requests, int request_count,
                           double* evs, const PrecomputedTables& tables);

// ---------------------------------------------------------------------------
// Research evaluator — configurable V2-style with structural knobs.
// ---------------------------------------------------------------------------

enum class TStrategy : int {
    V2_DEFAULT     = 0,
    EMPTY_ONLY     = 1,
    EMPTY_PLUS_1   = 2,
    EMPTY_PLUS_2   = 3,
    EMPTY_DOUBLE   = 4,
    PROPORTIONAL   = 5,
};

enum class CrushMode : int {
    ROUND_INT      = 0,
    FLOOR_INT      = 1,
    FLOAT_SMOOTH   = 2,
};

struct ResearchConfig {
    TStrategy t_me  = TStrategy::V2_DEFAULT;
    TStrategy t_opp = TStrategy::V2_DEFAULT;
    CrushMode crush = CrushMode::ROUND_INT;

    bool   output_win_odds = false;
    double clean_power = 1.0;
    double scratch_penalty = 0.0;
    double variance_penalty = 0.0;
    double opp_aware_factor = 0.0;       // 2v2: "behind the WORST threat"
    double early_high_coeff_bonus = 0.0;
    double early_progressive_bonus = 0.0;
    double coeff_sq_bonus = 0.0;
    double completion_bonus = 0.0;
    double dominance_bonus = 0.0;         // 2v2: "more cells than AVERAGE opp"
    double upper100_bonus = 0.0;
    double hc_scratch_penalty = 0.0;
    double upper_priority_bonus = 0.0;
    double coeff_sq_high_only = 0.0;
    double turbo_avoidance = 0.0;
    double last_cell_bonus = 0.0;
    double upper_bonus_penalty = 0.0;
    double crush_smoothness = 1.0;

    // 1-ply opponent lookahead — declared but not implemented.
    bool   lookahead_1ply = false;
    double lookahead_weight = 1.0;

    bool   use_v3_rules = false;
    V3Weights v3 = {0, 0, 0, 0, 0.5};
};

template <typename Traits>
void heuristic_evaluate_research(const BoardStateT<Traits>& base_board,
                                 const GameContextT<Traits>& base_ctx,
                                 const AfterstateRequest* requests, int request_count,
                                 double* evs, const PrecomputedTables& tables,
                                 const ResearchConfig& cfg);

// Returns the ResearchConfig for a numbered V4..V17 version. Variant-agnostic.
// Throws std::invalid_argument for V1/V2/V3 (which use their own evaluators).
const ResearchConfig& get_research_config_for(HeuristicVersion v);

// ---------------------------------------------------------------------------
// Play-turn functions — templated on the game variant.
// ---------------------------------------------------------------------------

/// Play one turn using the research evaluator with the given config.
template <typename Traits>
void heuristic_play_turn_research(GameStateT<Traits>& state,
                                  GameContextT<Traits>& ctx,
                                  const PrecomputedTables& tables,
                                  SolverBuffers& buffers, RNG& rng,
                                  const ResearchConfig& cfg);

/// Play one complete turn using the heuristic bot.
template <typename Traits>
void heuristic_play_turn(GameStateT<Traits>& state,
                         GameContextT<Traits>& ctx,
                         const PrecomputedTables& tables,
                         SolverBuffers& buffers, RNG& rng,
                         HeuristicVersion version = HeuristicVersion::V2);

/// Play a complete game using the heuristic bot for all players. Returns the
/// duel margin from the Team-0 / P0 perspective (compute_duel<Traits>(...)).
template <typename Traits>
int play_heuristic_game(RNG& rng, const PrecomputedTables& tables,
                        HeuristicVersion version = HeuristicVersion::V2);

/// Pick a random HeuristicVersion uniformly from V1..V17.
HeuristicVersion random_heuristic_version(RNG& rng);
