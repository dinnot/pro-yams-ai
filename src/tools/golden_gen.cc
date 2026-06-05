// ===========================================================================
// golden_gen — Pro Yams golden test corpus generator.
//
// Produces JSONL fixtures that are the authoritative correctness oracle for a
// separate, from-scratch TypeScript re-implementation of this rules engine.
// Every value emitted is *what this engine actually computes* at each decision
// (legal set + per-option score + canReroll) and at game end (full duel
// breakdown). Nothing is hand-derived: the generator drives the real engine
// (engine/game_flow, scoring, duel) and records its outputs verbatim.
//
// Strategy (see the project task spec):
//   * Simulate many full games per variant under the production rule config
//     (Lucky Yams ON), with a MIX of policies:
//       - random-legal      (breadth: surfaces forced scratches, weird boards)
//       - heuristic         (realism: strong-agent-like distributions)
//       - turbo-hoarding    (targeted: drives the Turbo 2-roll / reroll-illegal
//                            endgame by filling non-Turbo cells first)
//       - lucky-yams-forced (targeted: guarantees first-roll five-of-a-kind
//                            Lucky-Yams maxima across every row)
//     The policy only picks which legal action advances the game; the recorded
//     oracle is always the engine's own computed legal set/scores/duel.
//   * Coverage-guided selection: pass 1 simulates every candidate and keeps a
//     lightweight coverage signature + its (kind, seed, param) so the game is
//     deterministically regenerable. We greedily keep games that add new
//     coverage, then add a random breadth sample. Pass 2 re-simulates only the
//     selected games and writes their JSONL.
//   * Self-verify: re-read the emitted JSONL and replay it through the engine
//     (RNG-independently — driven entirely by the recorded dice), asserting the
//     recorded legal/scores/canReroll/result all match. Fails loudly on any
//     mismatch (that would be a serialization bug).
//
// Determinism: every game is a pure function of (kind, seed, param), so the
// corpus regenerates bit-for-bit from the master seed.
// ===========================================================================

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/board_init.h"
#include "engine/constants.h"
#include "engine/context_rebuild.h"
#include "engine/duel.h"
#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_rules.h"
#include "engine/game_state.h"
#include "engine/game_traits.h"
#include "engine/placement.h"
#include "engine/rng.h"
#include "engine/scoring.h"

#include "heuristic/heuristic_bot.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

using json = nlohmann::ordered_json;

// Process-wide precomputed solver tables, built once in main(). Used only by
// the heuristic action policy (for realistic action distributions); the
// recorded oracle never depends on them.
static PrecomputedTables* g_tables = nullptr;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// In-memory game record (mirrors the emitted JSON 1:1).
// ---------------------------------------------------------------------------
struct LegalEntry { int column, row, score; };

struct ActionRec {
    bool reroll = false;
    int holdMask = 0;            // valid when reroll
    int column = 0, row = 0, score = 0;  // valid when place
};

struct DecisionRec {
    int currentPlayer = 0;
    int dice[kNumDice] = {0,0,0,0,0};
    int rollsLeft = 0;
    bool canReroll = false;
    std::vector<LegalEntry> legal;
    ActionRec action;
};

struct PairingRec {
    int column, t0Player, t1Player, adjustedA, adjustedB, crush;
    long long points;
};

struct GameRec {
    std::string variant;
    int coefficients[kNumColumns] = {0,0,0,0,0,0};
    bool luckyYams = true;
    std::vector<DecisionRec> decisions;
    std::vector<long long> finalScores;          // per player
    std::array<long long, kNumColumns> duelColumns{};
    long long duelTotal = 0;
    std::vector<PairingRec> pairings;            // per column x cross-team pairing
    std::set<int> coverage;                      // tag ids exercised by this game
};

// ---------------------------------------------------------------------------
// Coverage tag registry — string <-> stable id.
// ---------------------------------------------------------------------------
struct TagRegistry {
    std::map<std::string, int> id_of;
    std::vector<std::string> name_of;
    int id(const std::string& s) {
        auto it = id_of.find(s);
        if (it != id_of.end()) return it->second;
        int v = static_cast<int>(name_of.size());
        id_of.emplace(s, v);
        name_of.push_back(s);
        return v;
    }
};
static TagRegistry g_tags;

// Policy kinds.
enum PolicyKind { kRandom = 0, kHeuristic = 1, kTurboHoard = 2, kLuckyYamsForced = 3 };

// A regenerable description of one candidate game.
struct GameSpec {
    int kind;
    uint64_t seed;
    int param;  // lucky-yams target row; unused otherwise
};

// ---------------------------------------------------------------------------
// Raw-score helper — used ONLY for coverage tagging (golden tie/miss
// detection). The emitted oracle scores always come from the engine's
// calculate_score / calculate_yams_bonus_score, never from this.
// ---------------------------------------------------------------------------
static int raw_score_for_tag(int row, const int8_t dice[kNumDice]) {
    int counts[7] = {0,0,0,0,0,0,0};
    int sum = 0;
    for (int i = 0; i < kNumDice; ++i) { counts[dice[i]]++; sum += dice[i]; }
    switch (row) {
        case kRow1s: return counts[1] * 1;
        case kRow2s: return counts[2] * 2;
        case kRow3s: return counts[3] * 3;
        case kRow4s: return counts[4] * 4;
        case kRow5s: return counts[5] * 5;
        case kRow6s: return counts[6] * 6;
        case kRowSS:
        case kRowLS: return sum;
        case kRowFH: {
            bool h3=false,h2=false,h5=false;
            for (int f=1;f<=6;++f){ if(counts[f]==3)h3=true; if(counts[f]==2)h2=true; if(counts[f]==5)h5=true; }
            return ((h3&&h2)||h5)?25:0;
        }
        case kRowK: { for(int f=1;f<=6;++f) if(counts[f]>=4) return f*4; return 0; }
        case kRowSTR: {
            bool sm=counts[1]&&counts[2]&&counts[3]&&counts[4]&&counts[5];
            bool lg=counts[2]&&counts[3]&&counts[4]&&counts[5]&&counts[6];
            if(lg)return 30; if(sm)return 25; return 0;
        }
        case kRowU8: return sum<8?sum:0;
        case kRowY: { for(int f=1;f<=6;++f) if(counts[f]==5) return 50; return 0; }
        default: return 0;
    }
}

static int upper_tier(int sum) { return upper_section_bonus(sum); }

// ---------------------------------------------------------------------------
// Duel breakdown — replicate compute_duel's per-pairing math so the TS port can
// be checked exactly, AND assert our per-column sum equals the engine's
// compute_duel_columns (guards the replication against drift).
// ---------------------------------------------------------------------------
template <typename Traits>
static std::vector<PairingRec> build_pairings(const BoardStateT<Traits>& board,
                                              const GameContextT<Traits>& ctx,
                                              GameRec& rec) {
    std::vector<PairingRec> out;
    const auto engine_cols = compute_duel_columns<Traits>(board, ctx);

    for (int col = 0; col < kNumColumns; ++col) {
        int raw_score[Traits::kNumPlayers] = {};
        bool is_clean[Traits::kNumPlayers] = {};
        for (int p = 0; p < Traits::kNumPlayers; ++p) {
            int cell_sum = 0;
            for (int row = 0; row < kNumRows; ++row) {
                int8_t v = board.cells[p][col][row];
                if (v > 0) cell_sum += v;
            }
            raw_score[p] = cell_sum + upper_section_bonus(ctx.upper_sum[p][col]);
            is_clean[p] = (ctx.upper_sum[p][col] >= 60) && (!ctx.lower_has_scratch[p][col]);
        }
        const int coeff = board.coefficients[col];
        long long col_sum = 0;
        for (int i = 0; i < Traits::kPlayersPerTeam; ++i) {
            for (int j = 0; j < Traits::kPlayersPerTeam; ++j) {
                const int t0p = Traits::kTeam0[i];
                const int t1p = Traits::kTeam1[j];
                const int raw0 = raw_score[t0p];
                const int raw1 = raw_score[t1p];
                const int c0 = crush_multiplier(raw0, raw1);
                const int c1 = crush_multiplier(raw1, raw0);
                const int active = (c0 > c1) ? c0 : c1;
                const int bonus_value = (active > 1) ? 100 : 200;
                const int adj0 = raw0 + (is_clean[t0p] ? bonus_value : 0);
                const int adj1 = raw1 + (is_clean[t1p] ? bonus_value : 0);
                const long long pts = (long long)(adj0 - adj1) * active * coeff;
                col_sum += pts;
                out.push_back({col, t0p, t1p, adj0, adj1, active, pts});

                // Duel coverage tags.
                rec.coverage.insert(g_tags.id("crush_" + std::to_string(active)));
                if (is_clean[t0p] || is_clean[t1p])
                    rec.coverage.insert(g_tags.id(active > 1 ? "clean_bonus_100" : "clean_bonus_200"));
                if (pts > 0) rec.coverage.insert(g_tags.id("duel_col_positive"));
                else if (pts < 0) rec.coverage.insert(g_tags.id("duel_col_negative"));
                else rec.coverage.insert(g_tags.id("duel_col_zero"));
            }
        }
        // Replication guard: our breakdown must sum to the engine's truth.
        if (col_sum != engine_cols[col]) {
            std::cerr << "FATAL: duel breakdown mismatch col " << col
                      << " ours=" << col_sum << " engine=" << engine_cols[col] << "\n";
            std::exit(2);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Policy: choose an action for the current decision. Returns an ActionRec whose
// fields are already legal/consistent with the current state.
// ---------------------------------------------------------------------------
// Heuristic (strong-agent) action: drive the real solver pipeline at decision
// granularity. Per the engine's own play loop, requests + EVs are computed once
// at turn start (rolls_left == 2) and reused across reroll layers; each decision
// then resolves greedily. We mirror that exactly, including the forced-place
// fallback when reroll is illegal.
template <typename Traits>
static ActionRec heuristic_action(GameStateT<Traits>& gs, GameContextT<Traits>& ctx,
                                  SolverBuffers& buffers, bool reroll_ok) {
    if (gs.rolls_left == 2) {  // turn start
        buffers.dp_computed = false;
        solver_get_requests<Traits>(gs, ctx, *g_tables, buffers);
        heuristic_evaluate_v2<Traits>(gs.board, ctx, buffers.requests,
                                      buffers.request_count, buffers.evs, *g_tables);
    }
    SolverResult r = solver_resolve_greedy<Traits>(gs, ctx, *g_tables, buffers);
    ActionRec a;
    if (r.should_place) { a.column = r.placement.column; a.row = r.placement.row; return a; }
    if (!reroll_ok) {  // forced to place — pick the solver's stop choice
        int id = get_dice_state_id(gs.dice, *g_tables);
        int16_t ri = buffers.stop_request_idx[id];
        if (ri < 0) ri = 0;
        a.column = buffers.requests[ri].placement.column;
        a.row = buffers.requests[ri].placement.row;
        return a;
    }
    a.reroll = true;
    a.holdMask = r.hold_mask & 0x1F;
    return a;
}

template <typename Traits>
static ActionRec choose_policy_action(int kind,
                                      const GameStateT<Traits>& gs,
                                      const GameContextT<Traits>& ctx,
                                      const LegalPlacementCache& legal,
                                      const std::vector<LegalEntry>& legal_scored,
                                      bool yams_active,
                                      bool reroll_ok,
                                      RNG& rng) {
    ActionRec a;

    // Reroll probability for the random-style policies.
    double p_reroll = (gs.rolls_left == 2) ? 0.55 : 0.45;
    if (yams_active) p_reroll = 0.12;  // mostly keep the Lucky Yams so we record bonus placements

    if (reroll_ok && rng.uniform_double() < p_reroll) {
        a.reroll = true;
        a.holdMask = rng.uniform_int(0, 31);
        return a;
    }

    // Placement.
    if (kind == kTurboHoard) {
        // Prefer a non-Turbo cell so Turbo fills last (drives the reroll-illegal
        // / Turbo 2-roll endgame). Only place Turbo when nothing else is legal.
        std::vector<int> non_turbo;
        for (int i = 0; i < legal.count; ++i)
            if (legal.placements[i].column != kColTurbo) non_turbo.push_back(i);
        int pick = non_turbo.empty()
                       ? rng.uniform_int(0, legal.count - 1)
                       : non_turbo[rng.uniform_int(0, (int)non_turbo.size() - 1)];
        a.column = legal.placements[pick].column;
        a.row = legal.placements[pick].row;
        return a;
    }

    // Random-legal placement.
    int pick = rng.uniform_int(0, legal.count - 1);
    a.column = legal.placements[pick].column;
    a.row = legal.placements[pick].row;
    return a;
}

// ---------------------------------------------------------------------------
// Tag a single decision (cell/turbo/golden/scratch/lucky-yams coverage).
// ---------------------------------------------------------------------------
template <typename Traits>
static void tag_decision(GameRec& rec, const GameStateT<Traits>& gs,
                         const GameContextT<Traits>& ctx,
                         const DecisionRec& d, bool yams_active) {
    if (yams_active) rec.coverage.insert(g_tags.id("lucky_yams_active"));
    if (d.rollsLeft >= 1 && !d.canReroll)
        rec.coverage.insert(g_tags.id("turbo_reroll_illegal"));

    // Forced scratch: every legal option scores 0.
    bool all_zero = !d.legal.empty();
    for (const auto& e : d.legal) if (e.score != 0) { all_zero = false; break; }
    if (all_zero) rec.coverage.insert(g_tags.id("forced_scratch"));

    // Legal-cell availability (covers Mid / UpDown wrap-adjacency reach, Turbo
    // availability, column order frontiers, ...).
    for (const auto& e : d.legal)
        rec.coverage.insert(g_tags.id("cell_legal_" + std::to_string(e.column) + "_" + std::to_string(e.row)));

    if (d.action.reroll) return;

    const int col = d.action.column, row = d.action.row, score = d.action.score;
    rec.coverage.insert(g_tags.id("cell_placed_" + std::to_string(col) + "_" + std::to_string(row)));
    if (score == 0) rec.coverage.insert(g_tags.id("placed_scratch"));

    if (col == kColTurbo)
        rec.coverage.insert(g_tags.id(d.rollsLeft == 2 ? "turbo_place_r2" : "turbo_place_r1"));

    if (yams_active) {
        rec.coverage.insert(g_tags.id("lucky_yams_place_row_" + std::to_string(row)));
        return;  // golden tagging below is for dice-derived placements only
    }

    // Golden-Rule tie / miss (relative to the bar in this cell).
    const int gmax = ctx.golden_max[col][row];
    const int raw = raw_score_for_tag(row, gs.dice);
    if (score > 0 && raw == gmax && gmax > 0)
        rec.coverage.insert(g_tags.id("golden_tie_meet"));
    if (raw > 0 && score == 0 && raw < gmax)
        rec.coverage.insert(g_tags.id("golden_miss"));

    // SS/LS interlock + mutual destruction.
    if (row == kRowSS || row == kRowLS) {
        int other = (row == kRowSS) ? kRowLS : kRowSS;
        int8_t partner = gs.board.cells[d.currentPlayer][col][other];
        if (score == 0 && partner != kCellEmpty)
            rec.coverage.insert(g_tags.id("ss_ls_forced_scratch"));
    }
}

// ---------------------------------------------------------------------------
// Simulate one full game. Pure function of (kind, seed, param).
// Records every decision + final duel breakdown + coverage tags.
// ---------------------------------------------------------------------------
template <typename Traits>
static GameRec play_game(int kind, uint64_t seed, int param) {
    GameRec rec;
    rec.variant = (Traits::kNumPlayers == 2) ? "1v1" : "2v2";
    rec.luckyYams = game_rules().yams_first_roll_bonus;

    RNG rng(seed);
    GameStateT<Traits> gs;
    GameContextT<Traits> ctx;
    // SolverBuffers is large; heap-allocate so it never threatens the stack.
    auto buffers = std::make_unique<SolverBuffers>();
    init_game<Traits>(gs, ctx, rng);
    for (int c = 0; c < kNumColumns; ++c) rec.coefficients[c] = gs.board.coefficients[c];

    bool first_decision = true;

    while (!is_terminal<Traits>(gs.board)) {
        const int p = gs.board.current_player;

        // Targeted: force a first-roll five-of-a-kind on the very first move.
        if (kind == kLuckyYamsForced && first_decision) {
            int8_t face = static_cast<int8_t>(1 + (param % kNumDieSides));
            for (int i = 0; i < kNumDice; ++i) gs.dice[i] = face;
            gs.rolls_left = 2;  // ensure first-roll
        }
        first_decision = false;

        const bool yams_active = yams_bonus_active<Traits>(gs);
        const bool reroll_ok = can_reroll<Traits>(gs, ctx);
        const LegalPlacementCache& legal = get_legal_placements<Traits>(gs, ctx);

        DecisionRec d;
        d.currentPlayer = p;
        for (int i = 0; i < kNumDice; ++i) d.dice[i] = gs.dice[i];
        d.rollsLeft = gs.rolls_left;
        d.canReroll = reroll_ok;
        for (int i = 0; i < legal.count; ++i) {
            const int col = legal.placements[i].column;
            const int row = legal.placements[i].row;
            const int score = yams_active
                                  ? calculate_yams_bonus_score<Traits>(row, p, col, gs.board, ctx)
                                  : calculate_score<Traits>(row, gs.dice, p, col, gs.board, ctx);
            d.legal.push_back({col, row, score});
        }

        // Lucky-yams-forced: on the forced first roll, deterministically place
        // (in Free, which accepts any row on an empty board) to record the
        // bonus maximum for the target row.
        ActionRec a;
        if (kind == kLuckyYamsForced && yams_active && d.action.reroll == false &&
            rec.decisions.empty()) {
            a.reroll = false;
            a.column = kColFree;
            a.row = param % kNumRows;
            if (!legal.is_legal[a.column][a.row]) {  // safety: fall back to any legal cell
                a.column = legal.placements[0].column;
                a.row = legal.placements[0].row;
            }
        } else if (kind == kHeuristic) {
            a = heuristic_action<Traits>(gs, ctx, *buffers, reroll_ok);
        } else {
            a = choose_policy_action<Traits>(kind, gs, ctx, legal, d.legal,
                                             yams_active, reroll_ok, rng);
        }

        if (a.reroll) {
            d.action = a;
            tag_decision<Traits>(rec, gs, ctx, d, yams_active);
            rec.decisions.push_back(std::move(d));
            perform_reroll<Traits>(gs, static_cast<uint8_t>(a.holdMask), rng);
        } else {
            // Compute the engine's score for the chosen cell exactly as
            // perform_placement will, and record it.
            a.score = yams_active
                          ? calculate_yams_bonus_score<Traits>(a.row, p, a.column, gs.board, ctx)
                          : calculate_score<Traits>(a.row, gs.dice, p, a.column, gs.board, ctx);
            d.action = a;
            tag_decision<Traits>(rec, gs, ctx, d, yams_active);
            rec.decisions.push_back(std::move(d));
            int applied = perform_placement<Traits>(gs, ctx, a.column, a.row, rng);
            (void)applied;  // == a.score by construction
        }
    }

    // Terminal: duel result + breakdown.
    {
        const auto cols = compute_duel_columns<Traits>(gs.board, ctx);
        for (int c = 0; c < kNumColumns; ++c) rec.duelColumns[c] = cols[c];
    }
    rec.duelTotal = compute_duel<Traits>(gs.board, ctx);
    rec.pairings = build_pairings<Traits>(gs.board, ctx, rec);

    rec.finalScores.assign(Traits::kNumPlayers, 0);
    for (int p = 0; p < Traits::kNumPlayers; ++p) {
        bool team0 = false;
        for (int i = 0; i < Traits::kPlayersPerTeam; ++i)
            if (Traits::kTeam0[i] == p) team0 = true;
        rec.finalScores[p] = team0 ? rec.duelTotal : -rec.duelTotal;
    }

    // Final-board coverage: upper-bonus tiers per player/column.
    for (int p = 0; p < Traits::kNumPlayers; ++p)
        for (int c = 0; c < kNumColumns; ++c) {
            int sum = ctx.upper_sum[p][c];
            rec.coverage.insert(g_tags.id("upper_tier_" + std::to_string(upper_tier(sum))));
            if (sum >= 55 && sum <= 65) rec.coverage.insert(g_tags.id("upper_near_60"));
            // SS/LS mutual destruction: both ended scratched.
            if (gs.board.cells[p][c][kRowSS] == 0 && gs.board.cells[p][c][kRowLS] == 0)
                rec.coverage.insert(g_tags.id("ss_ls_mutual_destruction"));
        }

    return rec;
}

static GameRec play_game_variant(const std::string& variant, const GameSpec& s) {
    if (variant == "1v1") return play_game<Yams1v1>(s.kind, s.seed, s.param);
    return play_game<Yams2v2>(s.kind, s.seed, s.param);
}

// ---------------------------------------------------------------------------
// JSON serialization.
// ---------------------------------------------------------------------------
static json to_json(const GameRec& g) {
    json j;
    j["variant"] = g.variant;
    j["coefficients"] = std::vector<int>(g.coefficients, g.coefficients + kNumColumns);
    j["luckyYams"] = g.luckyYams;

    json decs = json::array();
    for (const auto& d : g.decisions) {
        json jd;
        jd["currentPlayer"] = d.currentPlayer;
        jd["dice"] = std::vector<int>(d.dice, d.dice + kNumDice);
        jd["rollsLeft"] = d.rollsLeft;
        jd["canReroll"] = d.canReroll;
        json legal = json::array();
        for (const auto& e : d.legal)
            legal.push_back({{"column", e.column}, {"row", e.row}, {"score", e.score}});
        jd["legal"] = legal;
        json a;
        if (d.action.reroll) { a["type"] = "reroll"; a["holdMask"] = d.action.holdMask; }
        else { a["type"] = "place"; a["column"] = d.action.column; a["row"] = d.action.row; a["score"] = d.action.score; }
        jd["action"] = a;
        decs.push_back(jd);
    }
    j["decisions"] = decs;

    json res;
    res["finalScores"] = g.finalScores;
    res["duelColumns"] = std::vector<long long>(g.duelColumns.begin(), g.duelColumns.end());
    res["duelTotal"] = g.duelTotal;
    json pr = json::array();
    for (const auto& p : g.pairings)
        pr.push_back({{"column", p.column}, {"t0Player", p.t0Player}, {"t1Player", p.t1Player},
                      {"adjustedA", p.adjustedA}, {"adjustedB", p.adjustedB},
                      {"crush", p.crush}, {"points", p.points}});
    res["pairings"] = pr;
    j["result"] = res;
    return j;
}

// ===========================================================================
// Self-verification: replay an emitted JSON line through the engine, driven
// entirely by the recorded dice (no RNG), asserting every recorded value.
// ===========================================================================
template <typename Traits>
static bool verify_game(const json& j, std::string& err) {
    GameStateT<Traits> gs;
    GameContextT<Traits> ctx;

    // Reconstruct a fresh, empty board with the recorded coefficients.
    for (int p = 0; p < Traits::kNumPlayers; ++p)
        for (int c = 0; c < kNumColumns; ++c)
            for (int r = 0; r < kNumRows; ++r)
                gs.board.cells[p][c][r] = kCellEmpty;
    auto coeffs = j.at("coefficients");
    for (int c = 0; c < kNumColumns; ++c) gs.board.coefficients[c] = coeffs[c].get<int>();
    gs.board.cells_filled = 0;
    gs.board.current_player = 0;
    init_context<Traits>(ctx, gs.board);

    const auto& decs = j.at("decisions");
    for (const auto& d : decs) {
        const int p = d.at("currentPlayer").get<int>();
        gs.board.current_player = static_cast<int8_t>(p);
        gs.rolls_left = static_cast<int8_t>(d.at("rollsLeft").get<int>());
        const auto& dice = d.at("dice");
        for (int i = 0; i < kNumDice; ++i) gs.dice[i] = static_cast<int8_t>(dice[i].get<int>());

        const bool yams_active = yams_bonus_active<Traits>(gs);
        const bool reroll_ok = can_reroll<Traits>(gs, ctx);
        if (reroll_ok != d.at("canReroll").get<bool>()) { err = "canReroll mismatch"; return false; }

        // Recompute legal set + scores and compare against the record.
        const LegalPlacementCache& legal = get_legal_placements<Traits>(gs, ctx);
        const auto& jlegal = d.at("legal");
        if ((int)jlegal.size() != legal.count) { err = "legal count mismatch"; return false; }
        // Build engine map (col,row)->score; check the record matches it exactly.
        std::map<std::pair<int,int>, int> engine_scores;
        for (int i = 0; i < legal.count; ++i) {
            const int col = legal.placements[i].column;
            const int row = legal.placements[i].row;
            const int score = yams_active
                                  ? calculate_yams_bonus_score<Traits>(row, p, col, gs.board, ctx)
                                  : calculate_score<Traits>(row, gs.dice, p, col, gs.board, ctx);
            engine_scores[{col, row}] = score;
        }
        for (const auto& e : jlegal) {
            std::pair<int,int> key{e.at("column").get<int>(), e.at("row").get<int>()};
            auto it = engine_scores.find(key);
            if (it == engine_scores.end()) { err = "legal cell not in engine set"; return false; }
            if (it->second != e.at("score").get<int>()) { err = "legal score mismatch"; return false; }
        }

        const auto& act = d.at("action");
        if (act.at("type").get<std::string>() == "reroll") {
            // No board change; the next decision supplies the post-reroll dice.
            continue;
        }
        const int col = act.at("column").get<int>();
        const int row = act.at("row").get<int>();
        const int rec_score = act.at("score").get<int>();
        const int score = yams_active
                              ? calculate_yams_bonus_score<Traits>(row, p, col, gs.board, ctx)
                              : calculate_score<Traits>(row, gs.dice, p, col, gs.board, ctx);
        if (score != rec_score) { err = "placement score mismatch"; return false; }
        if (!legal.is_legal[col][row]) { err = "placed in illegal cell"; return false; }
        apply_placement<Traits>(p, col, row, score, gs.board, ctx);
    }

    if (!is_terminal<Traits>(gs.board)) { err = "game not terminal after replay"; return false; }

    // Duel result.
    const auto cols = compute_duel_columns<Traits>(gs.board, ctx);
    const long long total = compute_duel<Traits>(gs.board, ctx);
    const auto& res = j.at("result");
    if (total != res.at("duelTotal").get<long long>()) { err = "duelTotal mismatch"; return false; }
    const auto& jcols = res.at("duelColumns");
    for (int c = 0; c < kNumColumns; ++c)
        if (cols[c] != jcols[c].get<long long>()) { err = "duelColumn mismatch"; return false; }

    return true;
}

static bool verify_game_variant(const std::string& variant, const json& j, std::string& err) {
    if (variant == "1v1") return verify_game<Yams1v1>(j, err);
    return verify_game<Yams2v2>(j, err);
}

// ===========================================================================
// Generation pipeline for one variant.
// ===========================================================================
struct VariantStats {
    int emitted_games = 0;
    long long emitted_decisions = 0;
    int total_simulated = 0;
};

static std::vector<GameSpec> build_candidate_specs(uint64_t master_seed, int num_games) {
    std::vector<GameSpec> specs;
    RNG seeder(master_seed);

    // Targeted: lucky-yams forced first roll, covering every row (x a few faces).
    for (int rep = 0; rep < 3; ++rep)
        for (int row = 0; row < kNumRows; ++row)
            specs.push_back({kLuckyYamsForced, seeder.next(), row});

    // Targeted: turbo-hoarding endgames.
    int turbo_games = std::max(50, num_games / 50);
    for (int i = 0; i < turbo_games; ++i)
        specs.push_back({kTurboHoard, seeder.next(), 0});

    // Bulk mix: ~70% random-legal (breadth), ~30% heuristic (realism).
    for (int i = 0; i < num_games; ++i) {
        int kind = (seeder.uniform_int(0, 9) < 7) ? kRandom : kHeuristic;
        specs.push_back({kind, seeder.next(), 0});
    }
    return specs;
}

static void generate_variant(const std::string& variant, uint64_t master_seed,
                             int num_games, int max_emit, const fs::path& out_dir,
                             VariantStats& stats,
                             std::map<std::string, int>& global_tag_counts) {
    const std::vector<GameSpec> specs = build_candidate_specs(master_seed, num_games);
    stats.total_simulated = (int)specs.size();

    std::cout << "[" << variant << "] simulating " << specs.size()
              << " candidate games...\n";

    // Pass 1: simulate every candidate, keep only a lightweight coverage
    // signature + the spec (so selected games are deterministically regenerable).
    struct Cand { GameSpec spec; std::set<int> cov; };
    std::vector<Cand> cands;
    cands.reserve(specs.size());
    for (const auto& s : specs) {
        GameRec g = play_game_variant(variant, s);
        cands.push_back({s, std::move(g.coverage)});
    }

    // Coverage-guided greedy selection: keep a game if it adds new coverage.
    std::set<int> covered;
    std::vector<int> selected;
    for (int i = 0; i < (int)cands.size(); ++i) {
        bool adds = false;
        for (int t : cands[i].cov) if (!covered.count(t)) { adds = true; break; }
        if (adds) {
            for (int t : cands[i].cov) covered.insert(t);
            selected.push_back(i);
        }
    }
    std::cout << "[" << variant << "] coverage-guided picks: " << selected.size()
              << " (" << covered.size() << " distinct tags)\n";

    // Random breadth sample on top, up to max_emit total.
    RNG sampler(master_seed ^ 0x9E3779B97F4A7C15ULL);
    std::vector<int> in_selected(cands.size(), 0);
    for (int idx : selected) in_selected[idx] = 1;
    std::vector<int> pool;
    for (int i = 0; i < (int)cands.size(); ++i) if (!in_selected[i]) pool.push_back(i);
    // Fisher-Yates partial shuffle.
    for (int i = (int)pool.size() - 1; i > 0; --i) {
        int j = sampler.uniform_int(0, i);
        std::swap(pool[i], pool[j]);
    }
    for (int k = 0; k < (int)pool.size() && (int)selected.size() < max_emit; ++k)
        selected.push_back(pool[k]);

    std::sort(selected.begin(), selected.end());

    // Pass 2: re-simulate selected games and write JSONL + verify.
    fs::create_directories(out_dir);
    const fs::path file = out_dir / (variant + ".jsonl");
    std::ofstream ofs(file);
    if (!ofs) { std::cerr << "FATAL: cannot open " << file << "\n"; std::exit(3); }

    int emitted = 0;
    long long decisions = 0;
    for (int idx : selected) {
        GameRec g = play_game_variant(variant, cands[idx].spec);
        for (int t : g.coverage) global_tag_counts[g_tags.name_of[t]]++;
        const json j = to_json(g);
        const std::string line = j.dump();

        // Self-verify by re-parsing the emitted line and replaying it.
        json reparsed = json::parse(line);
        std::string err;
        if (!verify_game_variant(variant, reparsed, err)) {
            std::cerr << "FATAL: self-verify failed (game spec kind="
                      << cands[idx].spec.kind << " seed=" << cands[idx].spec.seed
                      << "): " << err << "\n";
            std::exit(4);
        }
        ofs << line << "\n";
        ++emitted;
        decisions += (long long)g.decisions.size();
    }
    ofs.close();

    stats.emitted_games = emitted;
    stats.emitted_decisions = decisions;
    std::cout << "[" << variant << "] emitted " << emitted << " games, "
              << decisions << " decisions -> " << file << " (self-verified OK)\n";
}

// ---------------------------------------------------------------------------
// CLI.
// ---------------------------------------------------------------------------
static void usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "  --seed N           master seed (default 20260530)\n"
        "  --games N          candidate games to simulate per variant (default 30000)\n"
        "  --max-emit N       max games to emit per variant (default 1500)\n"
        "  --variant V        '1v1', '2v2', or 'both' (default both)\n"
        "  --out DIR          output directory (default ./golden-out)\n";
}

int main(int argc, char** argv) {
    uint64_t seed = 20260530ULL;
    int games = 30000;
    int max_emit = 1500;
    std::string variant = "both";
    fs::path out = "golden-out";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--seed") seed = std::stoull(next());
        else if (a == "--games") games = std::stoi(next());
        else if (a == "--max-emit") max_emit = std::stoi(next());
        else if (a == "--variant") variant = next();
        else if (a == "--out") out = next();
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::cerr << "unknown arg: " << a << "\n"; usage(argv[0]); return 1; }
    }

    // Production rule configuration: Lucky Yams ON.
    GameRules rules;
    rules.yams_first_roll_bonus = true;
    set_game_rules(rules);

    // Build solver tables once (used only by the heuristic action policy).
    PrecomputedTables tables;
    init_precomputed_tables(tables);
    g_tables = &tables;

    std::vector<std::string> variants;
    if (variant == "both") variants = {"1v1", "2v2"};
    else variants = {variant};

    std::map<std::string, std::map<std::string, int>> tag_counts;  // variant -> tag -> count
    std::map<std::string, VariantStats> stats;

    for (const auto& v : variants) {
        VariantStats st;
        generate_variant(v, seed, games, max_emit, out / v, st, tag_counts[v]);
        stats[v] = st;
    }

    // -----------------------------------------------------------------------
    // Coverage report (stdout + coverage.md).
    // -----------------------------------------------------------------------
    std::ostringstream md;
    md << "# Golden Corpus Coverage Report\n\n";
    md << "Generated by `golden_gen` (master seed " << seed << ").\n\n";
    md << "Each game is a pure function of (policy kind, seed, param), so the "
          "corpus regenerates bit-for-bit. Every recorded value is the engine's "
          "own output; the generator never hand-derives scores or duel points.\n\n";

    md << "## Schema notes / additions\n\n"
       << "- `result.finalScores[p]` is the team-0-perspective duel total signed "
          "per player: team-0 players (1v1: P0; 2v2: P0,P2) get `+duelTotal`, "
          "team-1 players get `-duelTotal`.\n"
       << "- `result.pairings` is emitted for **both** variants (the spec marked "
          "it 2v2-only). In 1v1 it is the single (P0,P1) pairing per column; this "
          "lets the TS port check duel crush/clean-bonus math in 1v1 too. Each "
          "entry: `{column,t0Player,t1Player,adjustedA,adjustedB,crush,points}` "
          "where `adjustedA/B` are raw+clean-bonus for the team-0/team-1 player "
          "and `points = (adjustedA-adjustedB)*crush*coefficient`. The per-column "
          "sum of pairing `points` equals `duelColumns[column]` (asserted at "
          "generation time).\n"
       << "- `dice` are recorded exactly as the engine holds them (sorted "
          "ascending); `holdMask` is positional over that array (bit i keeps "
          "dice[i]).\n\n";

    std::cout << "\n================ SUMMARY ================\n";
    for (const auto& v : variants) {
        std::cout << "[" << v << "] simulated " << stats[v].total_simulated
                  << ", emitted " << stats[v].emitted_games << " games, "
                  << stats[v].emitted_decisions << " decisions, "
                  << tag_counts[v].size() << " distinct coverage tags\n";
        md << "## Variant " << v << "\n\n";
        md << "- Candidate games simulated: " << stats[v].total_simulated << "\n";
        md << "- Games emitted: " << stats[v].emitted_games << "\n";
        md << "- Decisions emitted: " << stats[v].emitted_decisions << "\n";
        md << "- Distinct coverage tags: " << tag_counts[v].size() << "\n\n";

        // Group tags by family for readability.
        md << "| coverage tag | count |\n|---|---|\n";
        for (const auto& [tag, cnt] : tag_counts[v])
            md << "| `" << tag << "` | " << cnt << " |\n";
        md << "\n";

        // Spotlight the required rule branches.
        auto cnt = [&](const std::string& t) {
            auto it = tag_counts[v].find(t);
            return it == tag_counts[v].end() ? 0 : it->second;
        };
        std::cout << "    required-branch coverage:\n";
        const std::vector<std::pair<std::string,std::string>> required = {
            {"first-roll Lucky Yams", "lucky_yams_active"},
            {"SS/LS mutual destruction", "ss_ls_mutual_destruction"},
            {"SS/LS forced scratch (interlock)", "ss_ls_forced_scratch"},
            {"forced scratch (all-zero)", "forced_scratch"},
            {"Turbo 2-roll placement (rolls_left==1)", "turbo_place_r1"},
            {"Turbo reroll illegal", "turbo_reroll_illegal"},
            {"upper-bonus >=60 tier", "upper_tier_30"},
            {"upper near 60 boundary", "upper_near_60"},
            {"Golden-Rule tie (meets bar)", "golden_tie_meet"},
            {"Golden-Rule miss (below bar)", "golden_miss"},
            {"crush x5", "crush_5"},
            {"clean-column bonus (crushed, 100)", "clean_bonus_100"},
            {"clean-column bonus (uncrushed, 200)", "clean_bonus_200"},
        };
        md << "### Required-branch spotlight (" << v << ")\n\n| branch | tag | count |\n|---|---|---|\n";
        for (const auto& [label, tag] : required) {
            int c = cnt(tag);
            std::cout << "      " << (c > 0 ? "OK " : "!! ") << label << ": " << c << "\n";
            md << "| " << label << " | `" << tag << "` | " << c << " |\n";
        }
        md << "\n";
    }

    std::ofstream cov(out / "coverage.md");
    cov << md.str();
    cov.close();
    std::cout << "\nWrote coverage report -> " << (out / "coverage.md") << "\n";

    return 0;
}
