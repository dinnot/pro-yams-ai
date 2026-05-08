#include "ui/server.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "engine/game_flow.h"
#include "engine/tensor.h"
#include "ui/json_serialization.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

UIServer::UIServer(int port, const std::string& static_dir,
                   SessionManager& sessions, const std::string& log_dir,
                   const std::string& checkpoints_dir,
                   TournamentManager* tournament)
    : sessions_(sessions), log_dir_(log_dir), static_dir_(static_dir),
      checkpoints_dir_(checkpoints_dir), tournament_(tournament),
      port_(port) {
    register_routes();
}

// ---------------------------------------------------------------------------
// start / stop
// ---------------------------------------------------------------------------

void UIServer::start() {
    server_.listen("0.0.0.0", port_);
}

void UIServer::stop() {
    server_.stop();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

int UIServer::parse_session_id(const httplib::Request& req) {
    try {
        return std::stoi(req.matches[1].str());
    } catch (...) {
        return -1;
    }
}

void UIServer::json_response(httplib::Response& res, const json& j, int status) {
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

void UIServer::error_response(httplib::Response& res,
                               const std::string& msg, int status) {
    res.status = status;
    res.set_content(json{{"error", msg}}.dump(), "application/json");
}

std::string UIServer::read_log_file(const std::string& dir,
                                     const std::string& filename) const {
    fs::path p = fs::path(dir) / filename;
    std::ifstream f(p);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ---------------------------------------------------------------------------
// Route registration
// ---------------------------------------------------------------------------

void UIServer::register_routes() {
    // CORS headers for browser access.
    server_.set_default_headers({
        {"Access-Control-Allow-Origin",  "*"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });
    server_.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    // --- Game management ---
    server_.Post("/api/game/new",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_new_game(req, res);
        });
    server_.Get(R"(/api/game/(\d+))",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_get_game(req, res);
        });
    server_.Post(R"(/api/game/(\d+)/step)",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_step(req, res);
        });
    server_.Post(R"(/api/game/(\d+)/play_all)",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_play_all(req, res);
        });
    server_.Delete(R"(/api/game/(\d+))",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_delete_game(req, res);
        });

    // --- Human interaction ---
    server_.Get(R"(/api/game/(\d+)/options)",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_options(req, res);
        });
    server_.Post(R"(/api/game/(\d+)/hold)",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_hold(req, res);
        });
    server_.Post(R"(/api/game/(\d+)/place)",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_place(req, res);
        });
    server_.Get(R"(/api/game/(\d+)/can_reroll)",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_can_reroll(req, res);
        });
    server_.Get(R"(/api/game/(\d+)/tensor)",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_tensor(req, res);
        });

    // --- Training logs ---
    server_.Get("/api/logs/training",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_training_log(req, res);
        });
    server_.Get("/api/logs/eval",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_eval_log(req, res);
        });
    server_.Get("/api/logs/list",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_log_list(req, res);
        });

    // --- Tournament ---
    server_.Get("/api/models/list",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_models_list(req, res);
        });
    server_.Post("/api/tournament/start",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_tournament_start(req, res);
        });
    server_.Get("/api/tournament/status",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_tournament_status(req, res);
        });
    server_.Post("/api/tournament/stop",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_tournament_stop(req, res);
        });

    // --- Static files ---
    // Mount the static directory; files are served as-is.
    if (!static_dir_.empty() && fs::exists(static_dir_)) {
        server_.set_mount_point("/", static_dir_);
    }
    // Explicit handler for "/" to serve index.html.
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

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

void UIServer::handle_new_game(const httplib::Request& req,
                                httplib::Response& res) {
    json body;
    try { body = json::parse(req.body); }
    catch (...) { error_response(res, "invalid JSON body"); return; }

    auto parse_player = [](const std::string& s) -> PlayerType {
        if (s == "human")        return PlayerType::kHuman;
        if (s == "nn")           return PlayerType::kNNSolver;
        if (s == "mc")           return PlayerType::kMCRollout;
        if (s == "heuristic_v2") return PlayerType::kHeuristicV2;
        if (s == "heuristic_v1") return PlayerType::kHeuristic;
        // "heuristic" legacy alias defaults to V1.
        return PlayerType::kHeuristic;
    };

    std::string p0_str    = body.value("player0",    "heuristic");
    std::string p1_str    = body.value("player1",    "heuristic");
    uint64_t    seed      = body.value("seed",        static_cast<uint64_t>(42));
    bool        debug_mode = body.value("debug_mode", false);

    int id = sessions_.create_session(parse_player(p0_str),
                                       parse_player(p1_str), seed, debug_mode);

    GameSession copy;
    sessions_.get_session_copy(id, copy);
    json_response(res, {{"session_id", id},
                         {"game_state", game_state_to_json(copy)}});
}

void UIServer::handle_get_game(const httplib::Request& req,
                                httplib::Response& res) {
    int id = parse_session_id(req);
    GameSession copy;
    if (!sessions_.get_session_copy(id, copy)) {
        error_response(res, "session not found", 404); return;
    }
    json_response(res, game_state_to_json(copy));
}

void UIServer::handle_step(const httplib::Request& req,
                            httplib::Response& res) {
    int id = parse_session_id(req);
    sessions_.advance_turn(id);

    GameSession copy;
    if (!sessions_.get_session_copy(id, copy)) {
        error_response(res, "session not found", 404); return;
    }
    json_response(res, game_state_to_json(copy));
}

void UIServer::handle_play_all(const httplib::Request& req,
                                httplib::Response& res) {
    int id = parse_session_id(req);
    sessions_.play_to_completion(id);

    GameSession copy;
    if (!sessions_.get_session_copy(id, copy)) {
        error_response(res, "session not found", 404); return;
    }
    json_response(res, game_state_to_json(copy));
}

void UIServer::handle_delete_game(const httplib::Request& req,
                                   httplib::Response& res) {
    int id = parse_session_id(req);
    sessions_.remove_session(id);
    json_response(res, {{"ok", true}});
}

void UIServer::handle_options(const httplib::Request& req,
                               httplib::Response& res) {
    int id = parse_session_id(req);
    bool cr = false;
    auto opts = sessions_.get_human_options(id, cr);

    if (opts.empty() && !cr) {
        error_response(res, "session not found or not human turn", 404); return;
    }
    json_response(res, options_to_json(opts, cr));
}

void UIServer::handle_hold(const httplib::Request& req,
                            httplib::Response& res) {
    int id = parse_session_id(req);
    json body;
    try { body = json::parse(req.body); }
    catch (...) { error_response(res, "invalid JSON body"); return; }

    uint8_t mask = static_cast<uint8_t>(body.value("hold_mask", 0));
    if (!sessions_.human_hold(id, mask)) {
        error_response(res, "cannot hold: invalid session or state"); return;
    }

    GameSession copy;
    if (!sessions_.get_session_copy(id, copy)) {
        error_response(res, "session not found", 404); return;
    }
    json_response(res, game_state_to_json(copy));
}

void UIServer::handle_place(const httplib::Request& req,
                             httplib::Response& res) {
    int id = parse_session_id(req);
    json body;
    try { body = json::parse(req.body); }
    catch (...) { error_response(res, "invalid JSON body"); return; }

    int column = body.value("column", -1);
    int row    = body.value("row", -1);
    if (column < 0 || row < 0) {
        error_response(res, "missing column/row"); return;
    }
    if (!sessions_.human_place(id, column, row)) {
        error_response(res, "illegal placement"); return;
    }

    GameSession copy;
    if (!sessions_.get_session_copy(id, copy)) {
        error_response(res, "session not found", 404); return;
    }
    json_response(res, game_state_to_json(copy));
}

void UIServer::handle_can_reroll(const httplib::Request& req,
                                  httplib::Response& res) {
    int id = parse_session_id(req);
    GameSession copy;
    if (!sessions_.get_session_copy(id, copy)) {
        error_response(res, "session not found", 404); return;
    }
    json_response(res, {{"can_reroll", can_reroll(copy.state, copy.ctx)},
                         {"rolls_left", static_cast<int>(copy.state.rolls_left)}});
}

void UIServer::handle_tensor(const httplib::Request& req,
                              httplib::Response& res) {
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

    GameSession copy;
    sessions_.get_session_copy(id, copy);
    int perspective = (player == 0 || player == 1)
                        ? player : static_cast<int>(copy.state.board.current_player);

    json values = json::array();
    for (float v : tensor) values.push_back(v);

    // Layout described in src/engine/tensor.h:
    //   A (312): per-player x col x row x [present, normalized_value]
    //   B (108): B.1 = pi x col x [upper_sum, e_raw] (24)
    //            B.2 = col x [rem_me, rem_opp, margin_now, margin_E,
    //                          d_crush_now, d_crush_E,
    //                          pts_to_2x, pts_to_3x, pts_to_4x, pts_to_5x (now),
    //                          pts_to_2x, pts_to_3x, pts_to_4x, pts_to_5x (E)] (84)
    //   C (156): per-player x col x row 1-turn non-scratch probability
    //   D (14):  col coeffs (6), filled_frac, duel_now, duel_E,
    //            dominance_my, dominance_opp, phase_50, phase_100, phase_140
    //   E (216): per-player x col x 18 (T_min/mid/max x [P60,P70,P80,P90,P100,EU/100])
    //   F (180): per-player x col x 15 (T_min/mid/max x [P_mid, EV_mid/60, P_low, EV_low/250, P_clean])
    json groups = json::array();
    groups.push_back({{"name", "A"}, {"start", 0},   {"size", 312},
                      {"description", "Cell occupancy and normalized value (per player x column x row x 2)"}});
    groups.push_back({{"name", "B"}, {"start", 312}, {"size", 108},
                      {"description", "Per-column derived scores and crush/duel projections"}});
    groups.push_back({{"name", "C"}, {"start", 420}, {"size", 156},
                      {"description", "1-turn non-scratch probability per cell (player x column x row)"}});
    groups.push_back({{"name", "D"}, {"start", 576}, {"size", 14},
                      {"description", "Global aggregates: column coefficients, fill ratio, total duel margins, dominance and phase flags"}});
    groups.push_back({{"name", "E"}, {"start", 590}, {"size", 216},
                      {"description", "Upper-section probabilities and EV across T_min/T_mid/T_max horizons (player x column x 18)"}});
    groups.push_back({{"name", "F"}, {"start", 806}, {"size", 180},
                      {"description", "Middle/lower P/EV and clean-column probability across horizons (player x column x 15)"}});

    json out = {
        {"size",        kTensorSize},
        {"perspective", perspective},
        {"values",      values},
        {"groups",      groups},
    };
    if (has_nn) out["nn_value"] = nn_val;

    json_response(res, out);
}

void UIServer::handle_training_log(const httplib::Request& req,
                                    httplib::Response& res) {
    std::string dir = log_dir_;
    if (req.has_param("dir")) dir = req.get_param_value("dir");
    std::string content = read_log_file(dir, "training_log.csv");
    if (content.empty()) { error_response(res, "log not found", 404); return; }
    res.status = 200;
    res.set_content(content, "text/csv");
}

void UIServer::handle_eval_log(const httplib::Request& req,
                                httplib::Response& res) {
    std::string dir = log_dir_;
    if (req.has_param("dir")) dir = req.get_param_value("dir");
    std::string content = read_log_file(dir, "eval_log.csv");
    if (content.empty()) { error_response(res, "log not found", 404); return; }
    res.status = 200;
    res.set_content(content, "text/csv");
}

void UIServer::handle_log_list(const httplib::Request&,
                                httplib::Response& res) {
    json dirs = json::array();
    std::error_code ec;
    if (fs::exists(log_dir_, ec)) {
        for (const auto& entry : fs::directory_iterator(log_dir_, ec)) {
            if (entry.is_directory())
                dirs.push_back(entry.path().string());
        }
    }
    // Always include the default log dir itself.
    dirs.push_back(log_dir_);
    json_response(res, {{"directories", dirs}});
}

// ---------------------------------------------------------------------------
// Tournament endpoints
// ---------------------------------------------------------------------------

void UIServer::handle_models_list(const httplib::Request& req,
                                   httplib::Response& res) {
    std::string dir = checkpoints_dir_;
    if (req.has_param("dir")) dir = req.get_param_value("dir");

    json models = json::array();
    std::error_code ec;
    if (fs::exists(dir, ec) && fs::is_directory(dir, ec)) {
        // Recursively walk the directory so nested run folders (e.g.
        // checkpoints/<run>/checkpoint_step_N.model) all show up.
        for (auto it = fs::recursive_directory_iterator(dir, ec);
             it != fs::recursive_directory_iterator();
             it.increment(ec)) {
            if (ec) break;
            if (!it->is_regular_file(ec)) continue;
            const auto& path = it->path();
            if (path.extension() != ".model") continue;
            auto stem = path.parent_path() / path.stem();

            // Build a display name relative to the root: "<run>/<step>" when
            // the file lives under a subdirectory, just "<step>" otherwise.
            fs::path rel = fs::relative(stem, dir, ec);
            std::string display = rel.empty() ? path.stem().string() : rel.string();

            json m;
            m["path"] = stem.string();
            m["filename"] = path.filename().string();
            m["display_name"] = display;
            models.push_back(m);
        }
    }
    // Sort by display name for stable display.
    std::sort(models.begin(), models.end(),
              [](const json& a, const json& b) {
                  return a.value("display_name", "") < b.value("display_name", "");
              });
    json_response(res, {{"models", models}, {"directory", dir}});
}

void UIServer::handle_tournament_start(const httplib::Request& req,
                                        httplib::Response& res) {
    if (tournament_ == nullptr) {
        error_response(res, "tournament manager unavailable", 503);
        return;
    }

    json body;
    try { body = json::parse(req.body); }
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

void UIServer::handle_tournament_status(const httplib::Request&,
                                         httplib::Response& res) {
    if (tournament_ == nullptr) {
        error_response(res, "tournament manager unavailable", 503);
        return;
    }
    TournamentState st = tournament_->get_status();

    json participants = json::array();
    for (const auto& p : st.participants) {
        participants.push_back({
            {"id",   p.id},
            {"name", p.name},
            {"type", agent_type_to_string(p.type)},
            {"path", p.path},
        });
    }

    json grid = json::object();
    for (const auto& [a_id, row] : st.grid) {
        json row_obj = json::object();
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

    json out = {
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

void UIServer::handle_tournament_stop(const httplib::Request&,
                                       httplib::Response& res) {
    if (tournament_ == nullptr) {
        error_response(res, "tournament manager unavailable", 503);
        return;
    }
    tournament_->stop();
    json_response(res, {{"ok", true}});
}
