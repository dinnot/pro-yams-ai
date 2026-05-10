#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>

#include "heuristic/heuristic_bot.h"
#include "model/pro_yams_net.h"
#include "solver/precomputed_tables.h"

// ---------------------------------------------------------------------------
// AgentType — what kind of bot drives a tournament participant.
// ---------------------------------------------------------------------------
enum class AgentType {
    kHeuristicV1,
    kHeuristicV2,
    kHeuristicV3,
    kHeuristicV4,
    kHeuristicV5,
    kHeuristicV6,
    kHeuristicV7,
    kHeuristicV8,
    kHeuristicV9,
    kHeuristicV10,
    kHeuristicV11,
    kHeuristicV12,
    kHeuristicV13,
    kHeuristicV14,
    kHeuristicV15,
    kNN,
};

const char* agent_type_to_string(AgentType t);
bool        parse_agent_type(const std::string& s, AgentType& out);

// ---------------------------------------------------------------------------
// TournamentParticipant — one bot in the tournament.
//
// `path` is required for kNN agents (checkpoint stem, no .model suffix).
// `model` is populated by the manager after deduplicated load; participants
// pointing at the same path share the same shared_ptr.
// ---------------------------------------------------------------------------
struct TournamentParticipant {
    std::string id;     // unique within this tournament (e.g. "heur_v2", "nn_step_50000")
    std::string name;   // display name shown in the UI
    AgentType   type   = AgentType::kHeuristicV2;
    std::string path;   // checkpoint stem (kNN only)
    std::shared_ptr<ProYamsNet> model;  // populated post-load (kNN only)
};

// ---------------------------------------------------------------------------
// MatchupResult — aggregated record between two participants A vs B.
// avg_margin_a is the mean duel margin from A's perspective (positive = A wins).
// ---------------------------------------------------------------------------
struct MatchupResult {
    int    wins_a       = 0;
    int    wins_b       = 0;
    int    draws        = 0;
    double avg_margin_a = 0.0;  // computed lazily as sum/games when displayed
    double sum_margin_a = 0.0;
    int    games        = 0;
};

// ---------------------------------------------------------------------------
// TournamentState — snapshot of running/completed tournament.
// ---------------------------------------------------------------------------
struct TournamentState {
    bool   is_running       = false;
    int    games_completed  = 0;
    int    total_games      = 0;
    int    games_per_matchup = 0;
    std::string error;        // non-empty if the tournament failed to start

    // Wall-clock progress.
    std::chrono::steady_clock::time_point started_at{};

    std::vector<TournamentParticipant> participants;

    // Sparse matrix: grid[a_id][b_id] holds A-vs-B aggregate.
    std::map<std::string, std::map<std::string, MatchupResult>> grid;
};

// ---------------------------------------------------------------------------
// TournamentManager — owns a single background-running tournament.
//
// One tournament at a time. start_tournament() returns immediately (false
// if one is already running or input was invalid); the worker thread plays
// every unique pair `games_per_matchup` times, alternating who is P0.
//
// All public methods are thread-safe.
// ---------------------------------------------------------------------------
class TournamentManager {
public:
    TournamentManager(const PrecomputedTables& tables, torch::Device device);
    ~TournamentManager();

    /// Start a tournament. Returns false (and sets state.error) if one is
    /// already running, participants are < 2, or NN checkpoints fail to load.
    bool start_tournament(std::vector<TournamentParticipant> participants,
                          int games_per_matchup,
                          std::string& out_error);

    /// Snapshot of current tournament state — safe to call any time.
    TournamentState get_status() const;

    /// True iff a tournament thread is currently running.
    bool is_running() const;

    /// Stop the running tournament early. No-op if none is running.
    void stop();

private:
    void run_tournament(std::vector<TournamentParticipant> participants,
                        int games_per_matchup);

    int play_one_game(const TournamentParticipant& a,
                      const TournamentParticipant& b,
                      int a_player,            // 0 or 1
                      uint64_t seed,
                      int& out_duel_margin) const;

    void play_turn(const TournamentParticipant& p,
                   GameState& state, GameContext& ctx,
                   SolverBuffers& buffers,
                   std::vector<float>& tensor_buffer,
                   RNG& rng) const;

    const PrecomputedTables& tables_;
    torch::Device device_;

    mutable std::mutex mutex_;
    TournamentState    state_;
    std::atomic<bool>  stop_flag_{false};
    std::thread        thread_;
};
