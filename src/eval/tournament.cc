#include "eval/tournament.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

#include "engine/duel.h"
#include "engine/game_flow.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "eval/evaluator.h"
#include "model/trainer.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// AgentType helpers
// ---------------------------------------------------------------------------

const char* agent_type_to_string(AgentType t) {
    switch (t) {
        case AgentType::kHeuristicV1:  return "kHeuristicV1";
        case AgentType::kHeuristicV2:  return "kHeuristicV2";
        case AgentType::kHeuristicV3:  return "kHeuristicV3";
        case AgentType::kHeuristicV4:  return "kHeuristicV4";
        case AgentType::kHeuristicV5:  return "kHeuristicV5";
        case AgentType::kHeuristicV6:  return "kHeuristicV6";
        case AgentType::kHeuristicV7:  return "kHeuristicV7";
        case AgentType::kHeuristicV8:  return "kHeuristicV8";
        case AgentType::kHeuristicV9:  return "kHeuristicV9";
        case AgentType::kHeuristicV10: return "kHeuristicV10";
        case AgentType::kHeuristicV11: return "kHeuristicV11";
        case AgentType::kHeuristicV12: return "kHeuristicV12";
        case AgentType::kHeuristicV13: return "kHeuristicV13";
        case AgentType::kHeuristicV14: return "kHeuristicV14";
        case AgentType::kHeuristicV15: return "kHeuristicV15";
        case AgentType::kHeuristicV16: return "kHeuristicV16";
        case AgentType::kHeuristicV17: return "kHeuristicV17";
        case AgentType::kNN:           return "kNN";
    }
    return "unknown";
}

bool parse_agent_type(const std::string& s, AgentType& out) {
    if (s == "kHeuristicV1" || s == "heuristic_v1" || s == "v1") {
        out = AgentType::kHeuristicV1;
        return true;
    }
    if (s == "kHeuristicV2" || s == "heuristic_v2" || s == "v2") {
        out = AgentType::kHeuristicV2;
        return true;
    }
    if (s == "kHeuristicV3" || s == "heuristic_v3" || s == "v3") {
        out = AgentType::kHeuristicV3;
        return true;
    }
    static const struct { const char* tag; AgentType t; HeuristicVersion v; } kVN[] = {
        {"4",  AgentType::kHeuristicV4,  HeuristicVersion::V4},
        {"5",  AgentType::kHeuristicV5,  HeuristicVersion::V5},
        {"6",  AgentType::kHeuristicV6,  HeuristicVersion::V6},
        {"7",  AgentType::kHeuristicV7,  HeuristicVersion::V7},
        {"8",  AgentType::kHeuristicV8,  HeuristicVersion::V8},
        {"9",  AgentType::kHeuristicV9,  HeuristicVersion::V9},
        {"10", AgentType::kHeuristicV10, HeuristicVersion::V10},
        {"11", AgentType::kHeuristicV11, HeuristicVersion::V11},
        {"12", AgentType::kHeuristicV12, HeuristicVersion::V12},
        {"13", AgentType::kHeuristicV13, HeuristicVersion::V13},
        {"14", AgentType::kHeuristicV14, HeuristicVersion::V14},
        {"15", AgentType::kHeuristicV15, HeuristicVersion::V15},
        {"16", AgentType::kHeuristicV16, HeuristicVersion::V16},
        {"17", AgentType::kHeuristicV17, HeuristicVersion::V17},
    };
    for (const auto& e : kVN) {
        std::string s1 = std::string("kHeuristicV") + e.tag;
        std::string s2 = std::string("heuristic_v") + e.tag;
        std::string s3 = std::string("v") + e.tag;
        if (s == s1 || s == s2 || s == s3) { out = e.t; return true; }
    }
    if (s == "kNN" || s == "nn") {
        out = AgentType::kNN;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// TournamentManager
// ---------------------------------------------------------------------------

TournamentManager::TournamentManager(const PrecomputedTables& tables,
                                     torch::Device device)
    : tables_(tables), device_(device) {}

TournamentManager::~TournamentManager() {
    stop();
    if (thread_.joinable()) thread_.join();
}

bool TournamentManager::is_running() const {
    std::lock_guard<std::mutex> g(mutex_);
    return state_.is_running;
}

void TournamentManager::stop() {
    stop_flag_.store(true);
    if (thread_.joinable()) thread_.join();
    stop_flag_.store(false);
}

TournamentState TournamentManager::get_status() const {
    std::lock_guard<std::mutex> g(mutex_);
    TournamentState copy = state_;
    // Materialize avg_margin_a from the running sums for the snapshot.
    for (auto& [a_id, row] : copy.grid) {
        for (auto& [b_id, m] : row) {
            m.avg_margin_a = (m.games > 0)
                ? m.sum_margin_a / static_cast<double>(m.games) : 0.0;
        }
    }
    return copy;
}

bool TournamentManager::start_tournament(std::vector<TournamentParticipant> participants,
                                         int games_per_matchup,
                                         std::string& out_error) {
    {
        std::lock_guard<std::mutex> g(mutex_);
        if (state_.is_running) {
            out_error = "tournament already running";
            return false;
        }
    }

    if (thread_.joinable()) thread_.join();
    stop_flag_.store(false);

    if (participants.size() < 2) {
        out_error = "need at least 2 participants";
        return false;
    }
    if (games_per_matchup <= 0) {
        out_error = "games_per_matchup must be > 0";
        return false;
    }

    // Reject duplicate IDs.
    {
        std::unordered_map<std::string, int> seen;
        for (const auto& p : participants) {
            if (p.id.empty()) {
                out_error = "participant id cannot be empty";
                return false;
            }
            if (++seen[p.id] > 1) {
                out_error = "duplicate participant id: " + p.id;
                return false;
            }
        }
    }

    // Load NN models (deduplicated by checkpoint path).
    std::unordered_map<std::string, std::shared_ptr<ProYamsNet>> model_cache;
    for (auto& p : participants) {
        if (p.type != AgentType::kNN) continue;
        if (p.path.empty()) {
            out_error = "NN participant '" + p.id + "' has empty checkpoint path";
            return false;
        }
        auto it = model_cache.find(p.path);
        if (it != model_cache.end()) {
            p.model = it->second;
            continue;
        }
        try {
            ModelConfig cfg = ModelTrainer::config_from_checkpoint(p.path);
            if (cfg.input_size != kTensorSize) {
                out_error = "checkpoint '" + p.path + "' has input_size=" +
                            std::to_string(cfg.input_size) +
                            " but the current tensor format is " +
                            std::to_string(kTensorSize) + " — incompatible.";
                return false;
            }
            ModelTrainer trainer(cfg, device_);
            trainer.load_weights(p.path);
            auto model = trainer.clone_for_inference(device_);
            model->to(device_);
            model->eval();
            model_cache[p.path] = model;
            p.model = model;
        } catch (const std::exception& e) {
            out_error = "failed to load checkpoint '" + p.path + "': " + e.what();
            return false;
        }
    }

    const int n = static_cast<int>(participants.size());
    const int total_pairs = n * (n - 1) / 2;
    const int total_games = total_pairs * games_per_matchup;

    {
        std::lock_guard<std::mutex> g(mutex_);
        state_ = TournamentState{};
        state_.is_running = true;
        state_.total_games = total_games;
        state_.games_per_matchup = games_per_matchup;
        state_.participants = participants;
        state_.started_at = std::chrono::steady_clock::now();
        // Pre-create the grid cells so the UI can render immediately.
        for (const auto& a : participants) {
            for (const auto& b : participants) {
                if (a.id == b.id) continue;
                state_.grid[a.id][b.id] = MatchupResult{};
            }
        }
    }

    thread_ = std::thread(&TournamentManager::run_tournament, this,
                          std::move(participants), games_per_matchup);
    return true;
}

void TournamentManager::run_tournament(std::vector<TournamentParticipant> participants,
                                       int games_per_matchup) {
    const int n = static_cast<int>(participants.size());

    auto record_error = [&](const std::string& msg) {
        std::lock_guard<std::mutex> g(mutex_);
        if (state_.error.empty()) state_.error = msg;
    };

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int g = 0; g < games_per_matchup; ++g) {
                if (stop_flag_.load()) goto done;

                // Alternate who plays as P0.
                const int a_player = g % 2;  // 0 → A is P0, 1 → A is P1
                const uint64_t seed = static_cast<uint64_t>(i) * 1'000'003ull +
                                      static_cast<uint64_t>(j) * 31ull +
                                      static_cast<uint64_t>(g);

                int duel_a = 0;
                try {
                    play_one_game(participants[i], participants[j],
                                  a_player, seed, duel_a);
                } catch (const std::exception& e) {
                    record_error(std::string("game error (") +
                                 participants[i].id + " vs " +
                                 participants[j].id + "): " + e.what());
                    goto done;
                } catch (...) {
                    record_error(std::string("game error (") +
                                 participants[i].id + " vs " +
                                 participants[j].id + "): unknown exception");
                    goto done;
                }

                // Update the grid, mirrored from each side's perspective.
                {
                    std::lock_guard<std::mutex> guard(mutex_);
                    auto& AB = state_.grid[participants[i].id][participants[j].id];
                    auto& BA = state_.grid[participants[j].id][participants[i].id];

                    AB.games++; BA.games++;
                    AB.sum_margin_a += duel_a;
                    BA.sum_margin_a += -duel_a;

                    if (duel_a > 0) {
                        AB.wins_a++;
                        BA.wins_b++;
                    } else if (duel_a < 0) {
                        AB.wins_b++;
                        BA.wins_a++;
                    } else {
                        AB.draws++;
                        BA.draws++;
                    }
                    state_.games_completed++;
                }
            }
        }
    }

done:
    std::lock_guard<std::mutex> g(mutex_);
    state_.is_running = false;
}

void TournamentManager::play_turn(const TournamentParticipant& p,
                                  GameState& state, GameContext& ctx,
                                  SolverBuffers& buffers,
                                  std::vector<float>& tensor_buffer,
                                  RNG& rng) const {
    SolverConfig greedy_cfg;
    switch (p.type) {
        case AgentType::kHeuristicV1:
            heuristic_play_turn(state, ctx, tables_, buffers, rng,
                                HeuristicVersion::V1);
            return;
        case AgentType::kHeuristicV2:
            heuristic_play_turn(state, ctx, tables_, buffers, rng,
                                HeuristicVersion::V2);
            return;
        case AgentType::kHeuristicV3:
            heuristic_play_turn(state, ctx, tables_, buffers, rng,
                                HeuristicVersion::V3);
            return;
        case AgentType::kHeuristicV4:
        case AgentType::kHeuristicV5:
        case AgentType::kHeuristicV6:
        case AgentType::kHeuristicV7:
        case AgentType::kHeuristicV8:
        case AgentType::kHeuristicV9:
        case AgentType::kHeuristicV10:
        case AgentType::kHeuristicV11:
        case AgentType::kHeuristicV12:
        case AgentType::kHeuristicV13:
        case AgentType::kHeuristicV14:
        case AgentType::kHeuristicV15:
        case AgentType::kHeuristicV16:
        case AgentType::kHeuristicV17: {
            const int delta = static_cast<int>(p.type) -
                              static_cast<int>(AgentType::kHeuristicV4);
            const HeuristicVersion v = static_cast<HeuristicVersion>(
                static_cast<int>(HeuristicVersion::V4) + delta);
            heuristic_play_turn(state, ctx, tables_, buffers, rng, v);
            return;
        }
        case AgentType::kNN:
            nn_play_turn(*p.model, device_, state, ctx, tables_,
                         buffers, tensor_buffer, greedy_cfg, rng);
            return;
    }
}

int TournamentManager::play_one_game(const TournamentParticipant& a,
                                     const TournamentParticipant& b,
                                     int a_player,
                                     uint64_t seed,
                                     int& out_duel_margin) const {
    GameState state;
    GameContext ctx;
    SolverBuffers buffers{};
    std::vector<float> tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize);

    RNG rng(seed);
    init_game(state, ctx, rng);

    while (!is_terminal(state.board)) {
        const int player = state.board.current_player;
        const TournamentParticipant& mover = (player == a_player) ? a : b;
        play_turn(mover, state, ctx, buffers, tensor_buffer, rng);
    }

    int duel = compute_duel(state.board, ctx);
    // Convert to A's perspective.
    if (a_player == 1) duel = -duel;
    out_duel_margin = duel;
    return duel;
}
