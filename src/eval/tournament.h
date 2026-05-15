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

#include "engine/game_traits.h"
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
    kHeuristicV16,
    kHeuristicV17,
    kNN,
};

const char* agent_type_to_string(AgentType t);
bool        parse_agent_type(const std::string& s, AgentType& out);

// ---------------------------------------------------------------------------
// TournamentParticipant — one bot in the tournament.
// ---------------------------------------------------------------------------
struct TournamentParticipant {
    std::string id;
    std::string name;
    AgentType   type   = AgentType::kHeuristicV2;
    std::string path;
    std::shared_ptr<ProYamsNet> model;
};

// ---------------------------------------------------------------------------
// MatchupResult — aggregated record between two participants A vs B.
// In 1v1, A vs B is one player vs one player.
// In 2v2, A controls one team and B controls the other (A's bot drives both
// of its team's seats, B's drives both of the other team's seats).
// avg_margin_a is the mean duel margin from A's perspective.
// ---------------------------------------------------------------------------
struct MatchupResult {
    int    wins_a       = 0;
    int    wins_b       = 0;
    int    draws        = 0;
    double avg_margin_a = 0.0;
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
    std::string error;

    std::chrono::steady_clock::time_point started_at{};

    std::vector<TournamentParticipant> participants;
    std::map<std::string, std::map<std::string, MatchupResult>> grid;
};

// ---------------------------------------------------------------------------
// TournamentManagerT<Traits> — owns a single background-running tournament.
//
// Templated on the game variant. In Yams1v1 mode each game is a 1v1 between
// two participants; in Yams2v2 mode each game is a 2v2 with one participant
// controlling both seats of one team and the other controlling the other team.
// ---------------------------------------------------------------------------
template <typename Traits>
class TournamentManagerT {
public:
    TournamentManagerT(const PrecomputedTables& tables, torch::Device device);
    ~TournamentManagerT();

    bool start_tournament(std::vector<TournamentParticipant> participants,
                          int games_per_matchup,
                          std::string& out_error);

    TournamentState get_status() const;
    bool is_running() const;
    void stop();

private:
    void run_tournament(std::vector<TournamentParticipant> participants,
                        int games_per_matchup);

    int play_one_game(const TournamentParticipant& a,
                      const TournamentParticipant& b,
                      int a_player,
                      uint64_t seed,
                      int& out_duel_margin) const;

    void play_turn(const TournamentParticipant& p,
                   GameStateT<Traits>& state, GameContextT<Traits>& ctx,
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

// Backward-compat aliases.
using TournamentManager    = TournamentManagerT<Yams1v1>;
using TournamentManager2v2 = TournamentManagerT<Yams2v2>;
