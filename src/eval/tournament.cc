#include "eval/tournament.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

#include "engine/duel.h"
#include "engine/game_flow.h"
#include "engine/game_traits.h"
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
// TournamentManagerT<Traits>
// ---------------------------------------------------------------------------

template <typename Traits>
TournamentManagerT<Traits>::TournamentManagerT(const PrecomputedTables& tables,
                                                torch::Device device)
    : tables_(tables), device_(device) {}

template <typename Traits>
TournamentManagerT<Traits>::~TournamentManagerT() {
    stop();
    if (thread_.joinable()) thread_.join();
}

template <typename Traits>
bool TournamentManagerT<Traits>::is_running() const {
    std::lock_guard<std::mutex> g(mutex_);
    return state_.is_running;
}

template <typename Traits>
void TournamentManagerT<Traits>::stop() {
    stop_flag_.store(true);
    if (thread_.joinable()) thread_.join();
    stop_flag_.store(false);
}

template <typename Traits>
TournamentState TournamentManagerT<Traits>::get_status() const {
    std::lock_guard<std::mutex> g(mutex_);
    TournamentState copy = state_;
    for (auto& [a_id, row] : copy.grid) {
        for (auto& [b_id, m] : row) {
            m.avg_margin_a = (m.games > 0)
                ? m.sum_margin_a / static_cast<double>(m.games) : 0.0;
        }
    }
    return copy;
}

template <typename Traits>
bool TournamentManagerT<Traits>::start_tournament(
        std::vector<TournamentParticipant> participants,
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

    // Load NN models (deduplicated by checkpoint path). NN checkpoints must
    // match this tournament's tensor format: 1v1 checkpoints (input_size=986)
    // for Yams1v1 tournaments, 2v2 checkpoints (input_size=2126) for Yams2v2.
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
            if (cfg.input_size != Traits::kTensorSize) {
                out_error = "checkpoint '" + p.path + "' has input_size=" +
                            std::to_string(cfg.input_size) +
                            " but this tournament expects input_size=" +
                            std::to_string(Traits::kTensorSize) +
                            " — incompatible variant.";
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
        for (const auto& a : participants) {
            for (const auto& b : participants) {
                if (a.id == b.id) continue;
                state_.grid[a.id][b.id] = MatchupResult{};
            }
        }
    }

    thread_ = std::thread(&TournamentManagerT::run_tournament, this,
                          std::move(participants), games_per_matchup);
    return true;
}

template <typename Traits>
void TournamentManagerT<Traits>::run_tournament(
        std::vector<TournamentParticipant> participants,
        int games_per_matchup) {
    const int n = static_cast<int>(participants.size());

    auto record_error = [&](const std::string& msg) {
        std::lock_guard<std::mutex> g(mutex_);
        if (state_.error.empty()) state_.error = msg;
    };

    // Flatten every (i, j, game) into an independent task. Each game owns its
    // own state/buffers/RNG and reads only the shared (eval-mode) models and
    // const tables, so games run concurrently with no per-game locking. The
    // seed is derived from (i, j, g) alone, so results don't depend on which
    // worker runs which game or in what order.
    struct GameTask { int i, j, g; };
    std::vector<GameTask> tasks;
    tasks.reserve(static_cast<size_t>(n) * n * games_per_matchup);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            for (int g = 0; g < games_per_matchup; ++g)
                tasks.push_back({i, j, g});

    std::atomic<size_t> next_task{0};
    std::atomic<bool>   abort{false};  // set on error to halt remaining workers

    auto worker = [&]() {
        size_t idx;
        while ((idx = next_task.fetch_add(1)) < tasks.size()) {
            if (stop_flag_.load() || abort.load()) return;

            const GameTask& tk = tasks[idx];
            // a_player alternates: in 1v1, who's P0; in 2v2, which team A
            // anchors (0 = Team 0 = seats {0,2}; 1 = Team 1 = seats {1,3}).
            const int a_player = tk.g % 2;
            const uint64_t seed = static_cast<uint64_t>(tk.i) * 1'000'003ull +
                                  static_cast<uint64_t>(tk.j) * 31ull +
                                  static_cast<uint64_t>(tk.g);

            int duel_a = 0;
            try {
                play_one_game(participants[tk.i], participants[tk.j],
                              a_player, seed, duel_a);
            } catch (const std::exception& e) {
                record_error(std::string("game error (") +
                             participants[tk.i].id + " vs " +
                             participants[tk.j].id + "): " + e.what());
                abort.store(true);
                return;
            } catch (...) {
                record_error(std::string("game error (") +
                             participants[tk.i].id + " vs " +
                             participants[tk.j].id + "): unknown exception");
                abort.store(true);
                return;
            }

            std::lock_guard<std::mutex> guard(mutex_);
            auto& AB = state_.grid[participants[tk.i].id][participants[tk.j].id];
            auto& BA = state_.grid[participants[tk.j].id][participants[tk.i].id];

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
    };

    const int nthreads = std::max<int>(1,
        std::min<int>(kDefaultEvalThreads, static_cast<int>(tasks.size())));
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    std::lock_guard<std::mutex> g(mutex_);
    state_.is_running = false;
}

template <typename Traits>
void TournamentManagerT<Traits>::play_turn(
        const TournamentParticipant& p,
        GameStateT<Traits>& state, GameContextT<Traits>& ctx,
        SolverBuffers& buffers,
        std::vector<float>& tensor_buffer,
        RNG& rng) const {
    SolverConfig greedy_cfg;
    switch (p.type) {
        case AgentType::kHeuristicV1:
            heuristic_play_turn<Traits>(state, ctx, tables_, buffers, rng,
                                        HeuristicVersion::V1);
            return;
        case AgentType::kHeuristicV2:
            heuristic_play_turn<Traits>(state, ctx, tables_, buffers, rng,
                                        HeuristicVersion::V2);
            return;
        case AgentType::kHeuristicV3:
            heuristic_play_turn<Traits>(state, ctx, tables_, buffers, rng,
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
            heuristic_play_turn<Traits>(state, ctx, tables_, buffers, rng, v);
            return;
        }
        case AgentType::kNN:
            nn_play_turn<Traits>(*p.model, device_, state, ctx, tables_,
                                 buffers, tensor_buffer, greedy_cfg, rng);
            return;
    }
}

template <typename Traits>
int TournamentManagerT<Traits>::play_one_game(
        const TournamentParticipant& a,
        const TournamentParticipant& b,
        int a_player,
        uint64_t seed,
        int& out_duel_margin) const {
    GameStateT<Traits> state;
    GameContextT<Traits> ctx;
    SolverBuffers buffers{};
    std::vector<float> tensor_buffer(
        static_cast<size_t>(kMaxAfterstateRequests) * Traits::kTensorSize);

    RNG rng(seed);
    init_game<Traits>(state, ctx, rng);

    while (!is_terminal<Traits>(state.board)) {
        const int player = state.board.current_player;
        // 1v1: A plays the seat == a_player; B plays the other seat.
        // 2v2: A anchors team `a_player`; A drives both seats on that team via
        // Traits::are_teammates; B drives the other team.
        const bool is_a_seat = Traits::are_teammates(player, a_player);
        const TournamentParticipant& mover = is_a_seat ? a : b;
        play_turn(mover, state, ctx, buffers, tensor_buffer, rng);
    }

    int duel = compute_duel<Traits>(state.board, ctx);
    // compute_duel returns Team-0 margin. Convert to A's perspective: if A
    // anchors Team 0 (a_player == 0 in both variants thanks to are_teammates),
    // no flip; otherwise flip.
    if (!Traits::are_teammates(a_player, 0)) duel = -duel;
    out_duel_margin = duel;
    return duel;
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template class TournamentManagerT<Yams1v1>;
template class TournamentManagerT<Yams2v2>;
