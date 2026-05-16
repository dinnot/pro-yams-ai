#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "engine/game_flow.h"
#include "engine/game_traits.h"
#include "engine/tensor.h"
#include "eval/tournament.h"
#include "ui/json_serialization.h"
#include "ui/session_manager.h"

// ---------------------------------------------------------------------------
// UIServerT<Traits> — HTTP server for the Pro Yams AI web interface.
//
// Serves:
//   REST API  →  /api/**
//   Frontend  →  / and all static files
//
// Templated on the game variant so 1v1 and 2v2 servers are distinct types.
// Header-only to keep template specialisations together with the routes.
// ---------------------------------------------------------------------------
template <typename Traits>
class UIServerT {
public:
    using Sessions = SessionManagerT<Traits>;
    using Session  = GameSessionT<Traits>;

    UIServerT(int port,
              const std::string& static_dir,
              Sessions& sessions,
              const std::string& log_dir,
              const std::string& checkpoints_dir,
              TournamentManagerT<Traits>* tournament)
        : sessions_(sessions),
          log_dir_(log_dir),
          static_dir_(static_dir),
          checkpoints_dir_(checkpoints_dir),
          tournament_(tournament),
          port_(port) {
        register_routes();
    }

    void start() { server_.listen("0.0.0.0", port_); }
    void stop()  { server_.stop(); }

    int port() const { return port_; }

private:
    void register_routes() {
        namespace fs = std::filesystem;

        server_.set_default_headers({
            {"Access-Control-Allow-Origin",  "*"},
            {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
            {"Access-Control-Allow-Headers", "Content-Type"}
        });
        server_.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            res.status = 204;
        });

        server_.Get("/api/info",
            [this](const httplib::Request& req, httplib::Response& res) { handle_info(req, res); });
        server_.Post("/api/game/new",
            [this](const httplib::Request& req, httplib::Response& res) { handle_new_game(req, res); });
        server_.Get(R"(/api/game/(\d+))",
            [this](const httplib::Request& req, httplib::Response& res) { handle_get_game(req, res); });
        server_.Post(R"(/api/game/(\d+)/step)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_step(req, res); });
        server_.Post(R"(/api/game/(\d+)/play_all)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_play_all(req, res); });
        server_.Delete(R"(/api/game/(\d+))",
            [this](const httplib::Request& req, httplib::Response& res) { handle_delete_game(req, res); });

        server_.Get(R"(/api/game/(\d+)/options)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_options(req, res); });
        server_.Post(R"(/api/game/(\d+)/hold)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_hold(req, res); });
        server_.Post(R"(/api/game/(\d+)/place)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_place(req, res); });
        server_.Get(R"(/api/game/(\d+)/can_reroll)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_can_reroll(req, res); });
        server_.Get(R"(/api/game/(\d+)/tensor)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_tensor(req, res); });

        server_.Get("/api/logs/training",
            [this](const httplib::Request& req, httplib::Response& res) { handle_training_log(req, res); });
        server_.Get("/api/logs/eval",
            [this](const httplib::Request& req, httplib::Response& res) { handle_eval_log(req, res); });
        server_.Get("/api/logs/list",
            [this](const httplib::Request& req, httplib::Response& res) { handle_log_list(req, res); });

        server_.Get("/api/models/list",
            [this](const httplib::Request& req, httplib::Response& res) { handle_models_list(req, res); });
        server_.Post("/api/tournament/start",
            [this](const httplib::Request& req, httplib::Response& res) { handle_tournament_start(req, res); });
        server_.Get("/api/tournament/status",
            [this](const httplib::Request& req, httplib::Response& res) { handle_tournament_status(req, res); });
        server_.Post("/api/tournament/stop",
            [this](const httplib::Request& req, httplib::Response& res) { handle_tournament_stop(req, res); });

        if (!static_dir_.empty() && fs::exists(static_dir_)) {
            server_.set_mount_point("/", static_dir_);
        }
        server_.Get("/", [this](const httplib::Request&, httplib::Response& res) {
            std::ifstream f(fs::path(static_dir_) / "index.html");
            if (!f) {
                res.status = 404;
                res.set_content("index.html not found. Use --static_dir to specify the frontend directory.",
                                "text/plain");
                return;
            }
            std::ostringstream ss;
            ss << f.rdbuf();
            res.set_content(ss.str(), "text/html");
        });
    }

    // ---- Helpers ----
    static int parse_session_id(const httplib::Request& req) {
        try { return std::stoi(req.matches[1].str()); }
        catch (...) { return -1; }
    }
    static void json_response(httplib::Response& res, const nlohmann::json& j, int status = 200) {
        res.status = status;
        res.set_content(j.dump(), "application/json");
    }
    static void error_response(httplib::Response& res, const std::string& msg, int status = 400) {
        res.status = status;
        res.set_content(nlohmann::json{{"error", msg}}.dump(), "application/json");
    }
    std::string read_log_file(const std::string& dir, const std::string& filename) const {
        namespace fs = std::filesystem;
        fs::path p = fs::path(dir) / filename;
        std::ifstream f(p);
        if (!f) return "";
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    static PlayerType parse_player(const std::string& s) {
        if (s == "human")         return PlayerType::kHuman;
        if (s == "nn")            return PlayerType::kNNSolver;
        if (s == "mc")            return PlayerType::kMCRollout;
        if (s == "heuristic_v17") return PlayerType::kHeuristicV17;
        if (s == "heuristic_v16") return PlayerType::kHeuristicV16;
        if (s == "heuristic_v15") return PlayerType::kHeuristicV15;
        if (s == "heuristic_v14") return PlayerType::kHeuristicV14;
        if (s == "heuristic_v13") return PlayerType::kHeuristicV13;
        if (s == "heuristic_v12") return PlayerType::kHeuristicV12;
        if (s == "heuristic_v11") return PlayerType::kHeuristicV11;
        if (s == "heuristic_v10") return PlayerType::kHeuristicV10;
        if (s == "heuristic_v9")  return PlayerType::kHeuristicV9;
        if (s == "heuristic_v8")  return PlayerType::kHeuristicV8;
        if (s == "heuristic_v7")  return PlayerType::kHeuristicV7;
        if (s == "heuristic_v6")  return PlayerType::kHeuristicV6;
        if (s == "heuristic_v5")  return PlayerType::kHeuristicV5;
        if (s == "heuristic_v4")  return PlayerType::kHeuristicV4;
        if (s == "heuristic_v3")  return PlayerType::kHeuristicV3;
        if (s == "heuristic_v2")  return PlayerType::kHeuristicV2;
        if (s == "heuristic_v1")  return PlayerType::kHeuristic;
        return PlayerType::kHeuristic;
    }

    // ---- Handlers ----
    // GET /api/info — static server metadata. The variant is fixed at launch,
    // so the frontend can use this on page load to decide which player-type
    // dropdowns to expose (P0/P1 for 1v1, P0..P3 for 2v2).
    void handle_info(const httplib::Request&, httplib::Response& res) {
        json_response(res, {
            {"num_players",  Traits::kNumPlayers},
            {"game_variant", (Traits::kNumPlayers == 4) ? "2v2" : "1v1"},
        });
    }

    void handle_new_game(const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) { error_response(res, "invalid JSON body"); return; }

        // Read player0..playerN-1. Missing slots default to heuristic.
        PlayerType pts[Traits::kNumPlayers];
        for (int p = 0; p < Traits::kNumPlayers; ++p) {
            std::string key = "player" + std::to_string(p);
            std::string val = body.value(key, std::string{"heuristic"});
            pts[p] = parse_player(val);
        }
        uint64_t seed       = body.value("seed",        static_cast<uint64_t>(42));
        bool     debug_mode = body.value("debug_mode",  false);

        int id = sessions_.create_session(pts, Traits::kNumPlayers, seed, debug_mode);

        Session copy;
        sessions_.get_session_copy(id, copy);
        json_response(res, {{"session_id", id},
                             {"game_state", game_state_to_json<Traits>(copy)}});
    }

    void handle_get_game(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        json_response(res, game_state_to_json<Traits>(copy));
    }

    void handle_step(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        sessions_.advance_turn(id);
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        json_response(res, game_state_to_json<Traits>(copy));
    }

    void handle_play_all(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        sessions_.play_to_completion(id);
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        json_response(res, game_state_to_json<Traits>(copy));
    }

    void handle_delete_game(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        sessions_.remove_session(id);
        json_response(res, {{"ok", true}});
    }

    void handle_options(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        bool cr = false;
        auto opts = sessions_.get_human_options(id, cr);
        if (opts.empty() && !cr) {
            error_response(res, "session not found or not human turn", 404); return;
        }
        json_response(res, options_to_json(opts, cr));
    }

    void handle_hold(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) { error_response(res, "invalid JSON body"); return; }
        uint8_t mask = static_cast<uint8_t>(body.value("hold_mask", 0));
        if (!sessions_.human_hold(id, mask)) {
            error_response(res, "cannot hold: invalid session or state"); return;
        }
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        json_response(res, game_state_to_json<Traits>(copy));
    }

    void handle_place(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) { error_response(res, "invalid JSON body"); return; }
        int column = body.value("column", -1);
        int row    = body.value("row", -1);
        if (column < 0 || row < 0) {
            error_response(res, "missing column/row"); return;
        }
        if (!sessions_.human_place(id, column, row)) {
            error_response(res, "illegal placement"); return;
        }
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        json_response(res, game_state_to_json<Traits>(copy));
    }

    void handle_can_reroll(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        json_response(res, {{"can_reroll", can_reroll<Traits>(copy.state, copy.ctx)},
                             {"rolls_left", static_cast<int>(copy.state.rolls_left)}});
    }

    void handle_tensor(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);

        int player = -1;
        if (req.has_param("player")) {
            try { player = std::stoi(req.get_param_value("player")); }
            catch (...) { player = -1; }
        }

        std::vector<float> tensor;
        float nn_val = 0.0f;
        bool  has_nn = false;
        if (!sessions_.compute_current_tensor(id, player, tensor, nn_val, has_nn)) {
            error_response(res, "session not found", 404);
            return;
        }

        Session copy;
        sessions_.get_session_copy(id, copy);
        int perspective = (player >= 0 && player < Traits::kNumPlayers)
                            ? player
                            : static_cast<int>(copy.state.board.current_player);

        nlohmann::json values = nlohmann::json::array();
        for (float v : tensor) values.push_back(v);

        nlohmann::json groups = nlohmann::json::array();
        // The detailed group breakdown is only emitted for the 1v1 layout the
        // frontend currently knows about. For 2v2 we ship the raw values and
        // size — the frontend can still display them as a flat list.
        if constexpr (std::is_same_v<Traits, Yams1v1>) {
            groups.push_back({{"name", "A"}, {"start", 0},   {"size", 312},
                              {"description", "Cell occupancy and normalized value (per player x column x row x 2)"}});
            groups.push_back({{"name", "B"}, {"start", 312}, {"size", 108},
                              {"description", "Per-column derived scores and crush/duel projections"}});
            groups.push_back({{"name", "C"}, {"start", 420}, {"size", 156},
                              {"description", "1-turn non-scratch probability per cell (player x column x row)"}});
            groups.push_back({{"name", "D"}, {"start", 576}, {"size", 14},
                              {"description", "Global aggregates"}});
            groups.push_back({{"name", "E"}, {"start", 590}, {"size", 216},
                              {"description", "Upper-section probabilities and EV"}});
            groups.push_back({{"name", "F"}, {"start", 806}, {"size", 180},
                              {"description", "Middle/lower P/EV and clean-column probability"}});
        }

        nlohmann::json out = {
            {"size",        Traits::kTensorSize},
            {"perspective", perspective},
            {"values",      values},
            {"groups",      groups},
        };
        if (has_nn) out["nn_value"] = nn_val;

        json_response(res, out);
    }

    void handle_training_log(const httplib::Request& req, httplib::Response& res) {
        std::string dir = log_dir_;
        if (req.has_param("dir")) dir = req.get_param_value("dir");
        std::string content = read_log_file(dir, "training_log.csv");
        if (content.empty()) { error_response(res, "log not found", 404); return; }
        res.status = 200;
        res.set_content(content, "text/csv");
    }

    void handle_eval_log(const httplib::Request& req, httplib::Response& res) {
        std::string dir = log_dir_;
        if (req.has_param("dir")) dir = req.get_param_value("dir");
        std::string content = read_log_file(dir, "eval_log.csv");
        if (content.empty()) { error_response(res, "log not found", 404); return; }
        res.status = 200;
        res.set_content(content, "text/csv");
    }

    void handle_log_list(const httplib::Request&, httplib::Response& res) {
        namespace fs = std::filesystem;
        nlohmann::json dirs = nlohmann::json::array();
        std::error_code ec;
        if (fs::exists(log_dir_, ec)) {
            for (const auto& entry : fs::directory_iterator(log_dir_, ec)) {
                if (entry.is_directory())
                    dirs.push_back(entry.path().string());
            }
        }
        dirs.push_back(log_dir_);
        json_response(res, {{"directories", dirs}});
    }

    void handle_models_list(const httplib::Request& req, httplib::Response& res) {
        namespace fs = std::filesystem;
        std::string dir = checkpoints_dir_;
        if (req.has_param("dir")) dir = req.get_param_value("dir");

        nlohmann::json models = nlohmann::json::array();
        std::error_code ec;
        if (fs::exists(dir, ec) && fs::is_directory(dir, ec)) {
            for (auto it = fs::recursive_directory_iterator(dir, ec);
                 it != fs::recursive_directory_iterator();
                 it.increment(ec)) {
                if (ec) break;
                if (!it->is_regular_file(ec)) continue;
                const auto& path = it->path();
                if (path.extension() != ".model") continue;
                auto stem = path.parent_path() / path.stem();

                fs::path rel = fs::relative(stem, dir, ec);
                std::string display = rel.empty() ? path.stem().string() : rel.string();

                nlohmann::json m;
                m["path"] = stem.string();
                m["filename"] = path.filename().string();
                m["display_name"] = display;
                models.push_back(m);
            }
        }
        std::sort(models.begin(), models.end(),
                  [](const nlohmann::json& a, const nlohmann::json& b) {
                      return a.value("display_name", "") < b.value("display_name", "");
                  });
        json_response(res, {{"models", models}, {"directory", dir}});
    }

    void handle_tournament_start(const httplib::Request& req, httplib::Response& res) {
        if (tournament_ == nullptr) {
            error_response(res, "tournament manager unavailable", 503);
            return;
        }
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) { error_response(res, "invalid JSON body"); return; }

        int games_per_matchup = body.value("games_per_matchup", 100);
        if (!body.contains("participants") || !body["participants"].is_array()) {
            error_response(res, "missing 'participants' array");
            return;
        }
        std::vector<TournamentParticipant> participants;
        for (const auto& p : body["participants"]) {
            TournamentParticipant tp;
            tp.id   = p.value("id",   std::string{});
            tp.name = p.value("name", tp.id);
            std::string type_str = p.value("type", std::string{});
            if (!parse_agent_type(type_str, tp.type)) {
                error_response(res, "unknown agent type: " + type_str);
                return;
            }
            tp.path = p.value("path", std::string{});
            participants.push_back(std::move(tp));
        }
        std::string err;
        if (!tournament_->start_tournament(std::move(participants),
                                           games_per_matchup, err)) {
            error_response(res, err);
            return;
        }
        json_response(res, {{"ok", true}});
    }

    void handle_tournament_status(const httplib::Request&, httplib::Response& res) {
        if (tournament_ == nullptr) {
            error_response(res, "tournament manager unavailable", 503);
            return;
        }
        TournamentState st = tournament_->get_status();

        nlohmann::json participants = nlohmann::json::array();
        for (const auto& p : st.participants) {
            participants.push_back({
                {"id",   p.id},
                {"name", p.name},
                {"type", agent_type_to_string(p.type)},
                {"path", p.path},
            });
        }
        nlohmann::json grid = nlohmann::json::object();
        for (const auto& [a_id, row] : st.grid) {
            nlohmann::json row_obj = nlohmann::json::object();
            for (const auto& [b_id, m] : row) {
                row_obj[b_id] = {
                    {"wins_a",       m.wins_a},
                    {"wins_b",       m.wins_b},
                    {"draws",        m.draws},
                    {"games",        m.games},
                    {"avg_margin_a", m.avg_margin_a},
                };
            }
            grid[a_id] = row_obj;
        }
        nlohmann::json out = {
            {"is_running",        st.is_running},
            {"games_completed",   st.games_completed},
            {"total_games",       st.total_games},
            {"games_per_matchup", st.games_per_matchup},
            {"participants",      participants},
            {"grid",              grid},
        };
        if (!st.error.empty()) out["error"] = st.error;
        json_response(res, out);
    }

    void handle_tournament_stop(const httplib::Request&, httplib::Response& res) {
        if (tournament_ == nullptr) {
            error_response(res, "tournament manager unavailable", 503);
            return;
        }
        tournament_->stop();
        json_response(res, {{"ok", true}});
    }

    httplib::Server server_;
    Sessions&       sessions_;
    std::string     log_dir_;
    std::string     static_dir_;
    std::string     checkpoints_dir_;
    TournamentManagerT<Traits>* tournament_;
    int             port_;
};

// Backward-compat aliases.
using UIServer    = UIServerT<Yams1v1>;
using UIServer2v2 = UIServerT<Yams2v2>;
