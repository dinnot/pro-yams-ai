#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "engine/duel.h"
#include "engine/game_flow.h"
#include "engine/game_rules.h"
#include "engine/game_traits.h"
#include "engine/tensor.h"
#include "eval/tournament.h"
#include "ui/game_recorder.h"
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
              TournamentManagerT<Traits>* tournament,
              GameRecorder* recorder = nullptr,
              const std::string& games_dir = "")
        : sessions_(sessions),
          log_dir_(log_dir),
          static_dir_(static_dir),
          checkpoints_dir_(checkpoints_dir),
          games_dir_(games_dir),
          tournament_(tournament),
          recorder_(recorder),
          port_(port) {
        register_routes();
    }

    void start() { server_.listen("0.0.0.0", port_); }
    void stop()  { server_.stop(); }

    int port() const { return port_; }

private:
    void register_routes() {
        namespace fs = std::filesystem;

        // Long-poll requests hold a worker thread open for up to kLongPollMs, so
        // give the server a generous pool (the public play app has few players).
        server_.new_task_queue = [] { return new httplib::ThreadPool(64); };

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
        server_.Get(R"(/api/game/(\d+)/longpoll)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_longpoll(req, res); });
        server_.Post(R"(/api/game/(\d+)/step)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_step(req, res); });
        server_.Post(R"(/api/game/(\d+)/bot_step)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_bot_step(req, res); });
        server_.Post(R"(/api/game/(\d+)/play_all)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_play_all(req, res); });
        server_.Delete(R"(/api/game/(\d+))",
            [this](const httplib::Request& req, httplib::Response& res) { handle_delete_game(req, res); });
        server_.Post(R"(/api/game/(\d+)/flag)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_flag(req, res); });
        server_.Post(R"(/api/game/(\d+)/heartbeat)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_heartbeat(req, res); });
        server_.Post(R"(/api/game/(\d+)/hold_preview)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_hold_preview(req, res); });
        server_.Post(R"(/api/game/(\d+)/takeover)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_takeover(req, res); });

        // Matchmaking (2v2 "play with a friend": two humans share one session).
        server_.Post("/api/matchmake/join",
            [this](const httplib::Request& req, httplib::Response& res) { handle_matchmake_join(req, res); });
        server_.Get(R"(/api/matchmake/(\d+))",
            [this](const httplib::Request& req, httplib::Response& res) { handle_matchmake_poll(req, res); });
        server_.Post(R"(/api/matchmake/(\d+)/cancel)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_matchmake_cancel(req, res); });

        server_.Get(R"(/api/game/(\d+)/options)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_options(req, res); });
        server_.Post(R"(/api/game/(\d+)/hold)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_hold(req, res); });
        server_.Post(R"(/api/game/(\d+)/place)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_place(req, res); });
        server_.Get(R"(/api/game/(\d+)/can_reroll)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_can_reroll(req, res); });
        server_.Get(R"(/api/game/(\d+)/luck)",
            [this](const httplib::Request& req, httplib::Response& res) { handle_game_luck(req, res); });
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

        // Recorded games (admin viewer). Available whenever a games dir is set.
        server_.Get("/api/games/stats",
            [this](const httplib::Request& req, httplib::Response& res) { handle_games_stats(req, res); });
        server_.Get("/api/games/list",
            [this](const httplib::Request& req, httplib::Response& res) { handle_games_list(req, res); });
        server_.Post("/api/games/eval",
            [this](const httplib::Request& req, httplib::Response& res) { handle_games_eval(req, res); });
        server_.Post("/api/games/eval_moves",
            [this](const httplib::Request& req, httplib::Response& res) { handle_games_eval_moves(req, res); });
        server_.Post("/api/games/accuracy_turn",
            [this](const httplib::Request& req, httplib::Response& res) { handle_games_accuracy_turn(req, res); });
        server_.Post("/api/games/accuracy_game",
            [this](const httplib::Request& req, httplib::Response& res) { handle_games_accuracy_game(req, res); });
        server_.Get(R"(/api/games/([0-9a-fA-F\-]+))",
            [this](const httplib::Request& req, httplib::Response& res) { handle_game_record(req, res); });

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

    // Parse a starting board position (the structured "position" object the
    // frontend sends) into a BoardStateT. Fills cells, coefficients and
    // current_player; cells_filled is recomputed by the session manager. On
    // any malformed/out-of-range input, returns false and sets `err`.
    static bool parse_position(const nlohmann::json& pos,
                               BoardStateT<Traits>& board, std::string& err) {
        const std::string my_variant = (Traits::kNumPlayers == 4) ? "2v2" : "1v1";
        const std::string variant = pos.value("variant", my_variant);
        if (variant != my_variant) {
            err = "position variant " + variant + " does not match server " + my_variant;
            return false;
        }

        if (!pos.contains("coefficients") || !pos["coefficients"].is_array() ||
            pos["coefficients"].size() != kNumColumns) {
            err = "position: coefficients must be an array of 6";
            return false;
        }
        for (int c = 0; c < kNumColumns; ++c)
            board.coefficients[c] = static_cast<int8_t>(pos["coefficients"][c].get<int>());

        int cp = pos.value("current_player", 0);
        if (cp < 0 || cp >= Traits::kNumPlayers) {
            err = "position: current_player out of range";
            return false;
        }
        board.current_player = static_cast<int8_t>(cp);
        board.cells_filled = 0;  // recomputed in create_session_from_board

        if (!pos.contains("boards") || !pos["boards"].is_object()) {
            err = "position: missing boards";
            return false;
        }
        const auto& boards = pos["boards"];
        for (int p = 0; p < Traits::kNumPlayers; ++p) {
            const std::string pkey = "player" + std::to_string(p);
            if (!boards.contains(pkey)) {
                err = "position: missing " + pkey;
                return false;
            }
            const auto& pboard = boards[pkey];
            for (int c = 0; c < kNumColumns; ++c) {
                const auto col_it = pboard.find(column_name(c));
                for (int r = 0; r < kNumRows; ++r) {
                    int v = kCellEmpty;
                    if (col_it != pboard.end()) {
                        const auto cell_it = col_it->find(row_name(r));
                        if (cell_it != col_it->end() && cell_it->is_number())
                            v = cell_it->get<int>();
                    }
                    if (v != kCellEmpty && (v < 0 || v > 100)) {
                        err = "position: cell value out of range";
                        return false;
                    }
                    board.cells[p][c][r] = static_cast<int8_t>(v);
                }
            }
        }
        return true;
    }

    // ---- Handlers ----
    // GET /api/info — static server metadata. The variant is fixed at launch,
    // so the frontend can use this on page load to decide which player-type
    // dropdowns to expose (P0/P1 for 1v1, P0..P3 for 2v2).
    void handle_info(const httplib::Request&, httplib::Response& res) {
        json_response(res, {
            {"num_players",  Traits::kNumPlayers},
            {"game_variant", (Traits::kNumPlayers == 4) ? "2v2" : "1v1"},
            {"yams_first_roll_bonus", game_rules().yams_first_roll_bonus},
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

        int id;
        if (body.contains("position") && body["position"].is_object()) {
            BoardStateT<Traits> board;
            std::string err;
            if (!parse_position(body["position"], board, err)) {
                error_response(res, err);
                return;
            }
            id = sessions_.create_session_from_board(pts, Traits::kNumPlayers,
                                                     board, seed, debug_mode);
        } else {
            id = sessions_.create_session(pts, Traits::kNumPlayers, seed, debug_mode);
        }

        Session copy;
        sessions_.get_session_copy(id, copy);

        // Record the game start (play server only). Human seats are those
        // configured as kHuman; coefficients come from the freshly dealt board.
        std::vector<int> human_seats;
        for (int p = 0; p < Traits::kNumPlayers; ++p)
            if (pts[p] == PlayerType::kHuman) human_seats.push_back(p);
        record_start(id, copy, human_seats, req);

        json_response(res, {{"session_id", id},
                             {"game_state", game_state_to_json<Traits>(copy)}});
    }

    // Record a game start with the recorder (play server only). Shared by the
    // normal new-game path and matchmaking. No-op if no recorder is configured.
    void record_start(int id, const Session& copy,
                      const std::vector<int>& human_seats,
                      const httplib::Request& req) {
        if (!recorder_) return;
        GameStartInfo info;
        info.variant     = (Traits::kNumPlayers == 4) ? "2v2" : "1v1";
        info.num_players = Traits::kNumPlayers;
        info.human_seats = human_seats;
        for (int c = 0; c < kNumColumns; ++c)
            info.coefficients.push_back(
                static_cast<int>(copy.state.board.coefficients[c]));
        info.ip              = req.remote_addr;
        info.x_forwarded_for = req.get_header_value("X-Forwarded-For");
        info.user_agent      = req.get_header_value("User-Agent");
        recorder_->start_game(id, info);
    }

    void handle_get_game(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        sessions_.refresh_disconnects(id, kSharedTimeoutMs);
        Session copy;
        uint64_t version = 0;
        if (!sessions_.get_session_copy(id, copy, version)) {
            error_response(res, "session not found", 404); return;
        }
        nlohmann::json out = game_state_to_json<Traits>(copy);
        out["version"] = version;
        json_response(res, out);
    }

    // GET /api/game/:id/longpoll?since=N — block until the game state changes
    // past version N (or ~kLongPollMs elapses), then return the full state plus
    // its `version`. With no `since`, returns the current state immediately (for
    // the initial fetch). Powers the shared-game live loop: the client re-issues
    // with the version it last saw, so partner/bot moves arrive near-instantly
    // without a fixed polling delay.
    void handle_longpoll(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        // Refresh disconnect flags first; a flip self-bumps the version so a
        // waiting client returns promptly when the grace period elapses.
        sessions_.refresh_disconnects(id, kSharedTimeoutMs);

        Session copy;
        uint64_t version = 0;
        if (req.has_param("since")) {
            uint64_t since = 0;
            try { since = std::stoull(req.get_param_value("since")); }
            catch (...) { since = 0; }
            if (!sessions_.wait_for_change(id, since, kLongPollMs, copy, version)) {
                error_response(res, "session not found", 404); return;
            }
        } else {
            if (!sessions_.get_session_copy(id, copy, version)) {
                error_response(res, "session not found", 404); return;
            }
        }
        nlohmann::json out = game_state_to_json<Traits>(copy);
        out["version"] = version;
        json_response(res, out);
    }

    // GET /api/game/:id/luck — per-seat luck for the (finished) game, computed
    // from the recorded move history with the server's play model. Returns
    // { seat_luck: [v0, v1, ...] } where each entry is a mean-zero index in
    // [-100,100] (null for a seat that took no turns).
    void handle_game_luck(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        std::vector<double> seat_luck;
        std::string err;
        if (!sessions_.compute_game_luck(id, seat_luck, err)) {
            error_response(res, err.empty() ? "could not compute luck" : err,
                           err == "session not found" ? 404 : 400);
            return;
        }
        nlohmann::json arr = nlohmann::json::array();
        for (double v : seat_luck) {
            if (std::isnan(v)) arr.push_back(nullptr);
            else               arr.push_back(v);
        }
        json_response(res, {{"seat_luck", arr}});
    }

    void handle_step(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        sessions_.refresh_disconnects(id, kSharedTimeoutMs);
        sessions_.advance_turn(id);
        sessions_.notify_changed(id);
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        record_if_finished(id, copy);
        json_response(res, game_state_to_json<Traits>(copy));
    }

    void handle_bot_step(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        sessions_.advance_turn_bot_override(id);
        sessions_.notify_changed(id);
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        record_if_finished(id, copy);
        json_response(res, game_state_to_json<Traits>(copy));
    }

    void handle_play_all(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        sessions_.refresh_disconnects(id, kSharedTimeoutMs);
        sessions_.play_to_completion(id);
        sessions_.notify_changed(id);
        Session copy;
        uint64_t version = 0;
        if (!sessions_.get_session_copy(id, copy, version)) {
            error_response(res, "session not found", 404); return;
        }
        record_if_finished(id, copy);
        nlohmann::json out = game_state_to_json<Traits>(copy);
        out["version"] = version;
        json_response(res, out);
    }

    void handle_delete_game(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        sessions_.remove_session(id);
        // A game deleted before completion counts as abandoned (its started.jsonl
        // line has no matching finished.jsonl entry → drop rate).
        if (recorder_) recorder_->forget(id);
        json_response(res, {{"ok", true}});
    }

    // POST /api/game/:id/flag — the human flags this game for review (e.g. the
    // AI played weird moves). Only meaningful on the play server, which owns a
    // recorder; the admin UI has none and returns 503.
    void handle_flag(const httplib::Request& req, httplib::Response& res) {
        if (!recorder_) { error_response(res, "recording not enabled", 503); return; }
        int id = parse_session_id(req);
        std::string note = "ai_weird_moves";
        if (!req.body.empty()) {
            try {
                nlohmann::json body = nlohmann::json::parse(req.body);
                note = body.value("note", note);
            } catch (...) { /* keep default note */ }
        }
        bool ok = recorder_->flag_game(id, note);
        if (!ok) { error_response(res, "unknown game", 404); return; }
        json_response(res, {{"ok", true}});
    }

    // POST /api/game/:id/heartbeat  body { seat } — a client controlling `seat`
    // in a shared two-human game reports it is still alive. Also runs the
    // disconnect→NN-takeover check so a stale partner is converted promptly.
    void handle_heartbeat(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        int seat = -1;
        if (!req.body.empty()) {
            try {
                nlohmann::json body = nlohmann::json::parse(req.body);
                seat = body.value("seat", -1);
            } catch (...) { /* keep seat = -1 */ }
        }
        sessions_.heartbeat(id, seat);
        sessions_.refresh_disconnects(id, kSharedTimeoutMs);
        json_response(res, {{"ok", true}});
    }

    // POST /api/game/:id/hold_preview  body { seat, mask } — the active player
    // streams their tentative (uncommitted) hold selection so a shared-game
    // teammate can watch the hold pattern form. Best-effort.
    void handle_hold_preview(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        int     seat = -1;
        uint8_t mask = 0;
        if (!req.body.empty()) {
            try {
                nlohmann::json body = nlohmann::json::parse(req.body);
                seat = body.value("seat", -1);
                mask = static_cast<uint8_t>(body.value("mask", 0));
            } catch (...) { /* ignore malformed */ }
        }
        if (sessions_.set_hold_preview(id, seat, mask))
            sessions_.notify_changed(id);  // wake the spectating teammate's long-poll
        json_response(res, {{"ok", true}});
    }

    // POST /api/game/:id/takeover  body { seat } — the surviving teammate chooses
    // to hand a disconnected seat to the NN. Only succeeds if that seat is a
    // human gone silent past the grace period; returns the updated game state.
    void handle_takeover(const httplib::Request& req, httplib::Response& res) {
        int id   = parse_session_id(req);
        int seat = -1;
        if (!req.body.empty()) {
            try {
                nlohmann::json body = nlohmann::json::parse(req.body);
                seat = body.value("seat", -1);
            } catch (...) { /* seat stays -1 */ }
        }
        if (!sessions_.takeover_seat(id, seat, kSharedTimeoutMs)) {
            error_response(res, "cannot take over seat (not a disconnected human)", 400);
            return;
        }
        sessions_.refresh_disconnects(id, kSharedTimeoutMs);
        sessions_.notify_changed(id);
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        json_response(res, game_state_to_json<Traits>(copy));
    }

    // ---- Matchmaking (2v2 "play with a friend") ----
    // Two browsers are paired into one shared session, each owning a single
    // human seat of the same (randomly chosen) team; the other team is NN.
    // The first to arrive waits (polling its ticket); the second completes the
    // match synchronously. State is in-memory and lost on restart — fine for a
    // casual feature; stale waiters are pruned lazily on the next join.

    void handle_matchmake_join(const httplib::Request& req, httplib::Response& res) {
        if constexpr (Traits::kNumPlayers != 4) {
            error_response(res, "matchmaking is only available in 2v2", 400);
        } else {
            std::lock_guard<std::mutex> lock(match_mutex_);
            prune_matchmaking();

            if (!waiting_.empty()) {
                MatchTicket w = waiting_.front();
                waiting_.erase(waiting_.begin());

                // Coin-flip the human team: Team 0 = seats {0,2}, Team 1 = {1,3}.
                int team = static_cast<int>(match_rng_() & 1ULL);
                int hs0  = (team == 0) ? 0 : 1;   // first waiter's seat
                int hs1  = (team == 0) ? 2 : 3;   // caller's seat
                std::vector<int> human_seats = {hs0, hs1};

                PlayerType pts[Traits::kNumPlayers];
                for (int p = 0; p < Traits::kNumPlayers; ++p)
                    pts[p] = PlayerType::kNNSolver;
                pts[hs0] = PlayerType::kHuman;
                pts[hs1] = PlayerType::kHuman;

                uint64_t seed = match_rng_();
                int id = sessions_.create_session(pts, Traits::kNumPlayers, seed, false);
                sessions_.mark_shared(id, human_seats);

                Session copy;
                sessions_.get_session_copy(id, copy);
                record_start(id, copy, human_seats, req);

                w.matched_session = id;
                w.seat            = hs0;
                resolved_[w.id]   = w;

                json_response(res, {{"status", "matched"},
                                    {"session_id", id},
                                    {"seat", hs1}});
            } else {
                MatchTicket t;
                t.id         = next_ticket_id_++;
                t.created_ms = Sessions::now_ms();
                waiting_.push_back(t);
                json_response(res, {{"status", "waiting"}, {"ticket", t.id}});
            }
        }
    }

    void handle_matchmake_poll(const httplib::Request& req, httplib::Response& res) {
        int ticket = parse_session_id(req);
        std::lock_guard<std::mutex> lock(match_mutex_);
        prune_matchmaking();
        auto it = resolved_.find(ticket);
        if (it != resolved_.end()) {
            json_response(res, {{"status", "matched"},
                                {"session_id", it->second.matched_session},
                                {"seat", it->second.seat}});
            return;
        }
        for (const auto& w : waiting_) {
            if (w.id == ticket) {
                json_response(res, {{"status", "waiting"}, {"ticket", ticket}});
                return;
            }
        }
        json_response(res, {{"status", "expired"}});
    }

    void handle_matchmake_cancel(const httplib::Request& req, httplib::Response& res) {
        int ticket = parse_session_id(req);
        std::lock_guard<std::mutex> lock(match_mutex_);
        for (auto it = waiting_.begin(); it != waiting_.end(); ++it) {
            if (it->id == ticket) { waiting_.erase(it); break; }
        }
        json_response(res, {{"ok", true}});
    }

    // Drop waiting tickets and resolved entries older than the match TTL so an
    // abandoned waiter never pairs with a fresh joiner. Caller holds match_mutex_.
    void prune_matchmaking() {
        int64_t cutoff = Sessions::now_ms() - kMatchTtlMs;
        waiting_.erase(std::remove_if(waiting_.begin(), waiting_.end(),
                          [&](const MatchTicket& t) { return t.created_ms < cutoff; }),
                       waiting_.end());
        for (auto it = resolved_.begin(); it != resolved_.end(); ) {
            if (it->second.created_ms < cutoff) it = resolved_.erase(it);
            else ++it;
        }
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
        sessions_.refresh_disconnects(id, kSharedTimeoutMs);
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) { error_response(res, "invalid JSON body"); return; }
        uint8_t mask = static_cast<uint8_t>(body.value("hold_mask", 0));
        if (!sessions_.human_hold(id, mask)) {
            error_response(res, "cannot hold: invalid session or state"); return;
        }
        sessions_.notify_changed(id);
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        json_response(res, game_state_to_json<Traits>(copy));
    }

    void handle_place(const httplib::Request& req, httplib::Response& res) {
        int id = parse_session_id(req);
        sessions_.refresh_disconnects(id, kSharedTimeoutMs);
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
        sessions_.notify_changed(id);
        Session copy;
        if (!sessions_.get_session_copy(id, copy)) {
            error_response(res, "session not found", 404); return;
        }
        record_if_finished(id, copy);
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

    // ---- Recorded games ----
    static int team_of(int player) {
        return (Traits::kNumPlayers == 4) ? (player & 1) : player;
    }

    // If the game just ended (and recording is enabled), compute the outcome
    // and hand the full record to the recorder. Idempotent: the recorder drops
    // its per-session metadata on the first call, so later calls are no-ops.
    void record_if_finished(int id, const Session& copy) {
        if (!recorder_ || !copy.game_over) return;

        GameOutcome out;
        out.result      = copy.result;
        out.winner_team = (copy.result > 0.5) ? 0 : (copy.result < 0.5) ? 1 : -1;

        out.human_team = -1;
        for (int p = 0; p < Traits::kNumPlayers; ++p) {
            if (copy.player_types[p] == PlayerType::kHuman) {
                out.human_team = team_of(p);
                break;
            }
        }

        const auto cols = compute_duel_columns<Traits>(copy.state.board, copy.ctx);
        long total = 0;
        out.column_margins.resize(kNumColumns);
        for (int c = 0; c < kNumColumns; ++c) {
            out.column_margins[c] = cols[c];
            total += cols[c];
        }
        out.total_margin = total;

        out.player_column_scores.resize(Traits::kNumPlayers);
        for (int p = 0; p < Traits::kNumPlayers; ++p) {
            out.player_column_scores[p].resize(kNumColumns);
            for (int c = 0; c < kNumColumns; ++c) {
                int cell_sum = 0;
                for (int r = 0; r < kNumRows; ++r) {
                    int8_t v = copy.state.board.cells[p][c][r];
                    if (v > 0) cell_sum += v;
                }
                out.player_column_scores[p][c] =
                    cell_sum + upper_section_bonus(copy.ctx.upper_sum[p][c]);
            }
        }

        recorder_->finish_game(id, game_state_to_json<Traits>(copy), out);
    }

    std::vector<nlohmann::json> read_jsonl(const std::string& path) const {
        std::vector<nlohmann::json> out;
        std::ifstream f(path);
        if (!f) return out;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            try { out.push_back(nlohmann::json::parse(line)); }
            catch (...) { /* skip malformed lines */ }
        }
        return out;
    }

    // Set of uuids the player flagged for review (from flagged.jsonl). Used to
    // annotate the games list and the per-game record.
    std::set<std::string> read_flagged_uuids() const {
        std::set<std::string> out;
        if (games_dir_.empty()) return out;
        namespace fs = std::filesystem;
        for (const auto& f : read_jsonl((fs::path(games_dir_) / "flagged.jsonl").string())) {
            std::string uuid = f.value("uuid", "");
            if (!uuid.empty()) out.insert(uuid);
        }
        return out;
    }

    // Is this finished-game entry a human win / loss / draw?
    static std::string human_result(const nlohmann::json& g) {
        double result = g.value("result", 0.5);
        int team = g.value("human_team", -1);
        if (result == 0.5 || team < 0) return "draw";
        bool team0_won = (result > 0.5);
        bool human_won = (team0_won && team == 0) || (!team0_won && team == 1);
        return human_won ? "win" : "loss";
    }
    static long human_margin(const nlohmann::json& g) {
        long m = g.value("total_margin", 0);
        return (g.value("human_team", 0) == 0) ? m : -m;
    }

    // Effective client IP for an entry: the proxy-forwarded address when set
    // (the real client behind our reverse proxy), otherwise the socket peer.
    static std::string client_ip(const nlohmann::json& g) {
        std::string xff = g.value("x_forwarded_for", "");
        return xff.empty() ? g.value("ip", "") : xff;
    }

    void handle_games_stats(const httplib::Request&, httplib::Response& res) {
        if (games_dir_.empty()) { error_response(res, "no games directory configured", 404); return; }
        namespace fs = std::filesystem;
        auto started  = read_jsonl((fs::path(games_dir_) / "started.jsonl").string());
        auto finished = read_jsonl((fs::path(games_dir_) / "finished.jsonl").string());

        struct Agg {
            std::string checkpoint, variant;
            long started = 0, finished = 0;
            long wins = 0, losses = 0, draws = 0;
            double margin_sum = 0.0;
        };
        std::map<std::string, Agg> groups;  // key: checkpoint \n variant
        auto key_of = [](const std::string& ckpt, const std::string& var) {
            return ckpt + "\n" + var;
        };

        for (const auto& g : started) {
            std::string ckpt = g.value("checkpoint", "");
            std::string var  = g.value("variant", "");
            auto& a = groups[key_of(ckpt, var)];
            a.checkpoint = ckpt; a.variant = var; a.started++;
        }
        for (const auto& g : finished) {
            std::string ckpt = g.value("checkpoint", "");
            std::string var  = g.value("variant", "");
            auto& a = groups[key_of(ckpt, var)];
            a.checkpoint = ckpt; a.variant = var; a.finished++;
            std::string hr = human_result(g);
            if      (hr == "win")  a.wins++;
            else if (hr == "loss") a.losses++;
            else                   a.draws++;
            a.margin_sum += static_cast<double>(human_margin(g));
        }

        nlohmann::json by_ckpt = nlohmann::json::array();
        long tot_started = 0, tot_finished = 0, tot_wins = 0, tot_losses = 0, tot_draws = 0;
        double tot_margin = 0.0;
        for (const auto& [k, a] : groups) {
            long dropped = a.started - a.finished;
            if (dropped < 0) dropped = 0;
            nlohmann::json row;
            row["checkpoint"]      = a.checkpoint;
            row["variant"]         = a.variant;
            row["started"]         = a.started;
            row["finished"]        = a.finished;
            row["dropped"]         = dropped;
            row["drop_rate"]       = a.started > 0 ? static_cast<double>(dropped) / a.started : 0.0;
            row["human_wins"]      = a.wins;
            row["human_losses"]    = a.losses;
            row["draws"]           = a.draws;
            row["human_win_rate"]  = a.finished > 0 ? static_cast<double>(a.wins) / a.finished : 0.0;
            row["avg_human_margin"]= a.finished > 0 ? a.margin_sum / a.finished : 0.0;
            by_ckpt.push_back(row);
            tot_started += a.started; tot_finished += a.finished;
            tot_wins += a.wins; tot_losses += a.losses; tot_draws += a.draws;
            tot_margin += a.margin_sum;
        }

        // ---- Per-human aggregation ----
        // A "human" is keyed by effective client IP + user-agent. started lines
        // carry that identity directly; finished lines do not, so we join them
        // back to their started line by uuid.
        struct HumanAgg {
            std::string ip, user_agent;
            long started = 0, finished = 0;
            long wins = 0, losses = 0, draws = 0;
            double margin_sum = 0.0;
            int64_t last_ts_ms = 0;
        };
        std::map<std::string, HumanAgg> humans;          // key: ip \n user_agent
        std::map<std::string, std::string> uuid_to_human;  // finished -> identity
        auto human_key = [](const std::string& ip, const std::string& ua) {
            return ip + "\n" + ua;
        };

        for (const auto& g : started) {
            std::string ip = client_ip(g);
            std::string ua = g.value("user_agent", "");
            std::string key = human_key(ip, ua);
            auto& h = humans[key];
            h.ip = ip; h.user_agent = ua; h.started++;
            int64_t ts = g.value("start_ts_ms", (int64_t)0);
            if (ts > h.last_ts_ms) h.last_ts_ms = ts;
            std::string uuid = g.value("uuid", "");
            if (!uuid.empty()) uuid_to_human[uuid] = key;
        }
        for (const auto& g : finished) {
            auto it = uuid_to_human.find(g.value("uuid", ""));
            if (it == uuid_to_human.end()) continue;  // no matching started line
            auto& h = humans[it->second];
            h.finished++;
            std::string hr = human_result(g);
            if      (hr == "win")  h.wins++;
            else if (hr == "loss") h.losses++;
            else                   h.draws++;
            h.margin_sum += static_cast<double>(human_margin(g));
            int64_t ts = g.value("finish_ts_ms", (int64_t)0);
            if (ts > h.last_ts_ms) h.last_ts_ms = ts;
        }

        nlohmann::json by_human = nlohmann::json::array();
        for (const auto& [k, h] : humans) {
            long dropped = h.started - h.finished;
            if (dropped < 0) dropped = 0;
            nlohmann::json row;
            row["ip"]               = h.ip;
            row["user_agent"]       = h.user_agent;
            row["started"]          = h.started;
            row["finished"]         = h.finished;
            row["dropped"]          = dropped;
            row["drop_rate"]        = h.started > 0 ? static_cast<double>(dropped) / h.started : 0.0;
            row["human_wins"]       = h.wins;
            row["human_losses"]     = h.losses;
            row["draws"]            = h.draws;
            row["human_win_rate"]   = h.finished > 0 ? static_cast<double>(h.wins) / h.finished : 0.0;
            row["avg_human_margin"] = h.finished > 0 ? h.margin_sum / h.finished : 0.0;
            row["last_ts_ms"]       = h.last_ts_ms;
            by_human.push_back(row);
        }

        long tot_dropped = tot_started - tot_finished;
        if (tot_dropped < 0) tot_dropped = 0;
        nlohmann::json overall;
        overall["started"]          = tot_started;
        overall["finished"]         = tot_finished;
        overall["dropped"]          = tot_dropped;
        overall["drop_rate"]        = tot_started > 0 ? static_cast<double>(tot_dropped) / tot_started : 0.0;
        overall["human_wins"]       = tot_wins;
        overall["human_losses"]     = tot_losses;
        overall["draws"]            = tot_draws;
        overall["human_win_rate"]   = tot_finished > 0 ? static_cast<double>(tot_wins) / tot_finished : 0.0;
        overall["avg_human_margin"] = tot_finished > 0 ? tot_margin / tot_finished : 0.0;

        json_response(res, {{"overall", overall}, {"by_checkpoint", by_ckpt},
                            {"by_human", by_human}});
    }

    void handle_games_list(const httplib::Request& req, httplib::Response& res) {
        if (games_dir_.empty()) { error_response(res, "no games directory configured", 404); return; }
        namespace fs = std::filesystem;
        auto finished = read_jsonl((fs::path(games_dir_) / "finished.jsonl").string());

        const std::string f_ckpt   = req.has_param("checkpoint") ? req.get_param_value("checkpoint") : "";
        const std::string f_var    = req.has_param("variant")    ? req.get_param_value("variant")    : "";
        const std::string f_out    = req.has_param("outcome")    ? req.get_param_value("outcome")    : "";
        const bool         f_flag  = req.has_param("flagged") &&
                                     (req.get_param_value("flagged") == "1" ||
                                      req.get_param_value("flagged") == "true");

        const std::set<std::string> flagged = read_flagged_uuids();

        std::vector<const nlohmann::json*> rows;
        for (const auto& g : finished) {
            if (!f_ckpt.empty() && g.value("checkpoint", "") != f_ckpt) continue;
            if (!f_var.empty()  && g.value("variant", "")    != f_var)  continue;
            if (!f_out.empty()  && human_result(g)           != f_out)  continue;
            if (f_flag && !flagged.count(g.value("uuid", "")))          continue;
            rows.push_back(&g);
        }
        // Newest first.
        std::sort(rows.begin(), rows.end(), [](const nlohmann::json* a, const nlohmann::json* b) {
            return a->value("finish_ts_ms", (int64_t)0) > b->value("finish_ts_ms", (int64_t)0);
        });

        long total  = static_cast<long>(rows.size());
        long offset = req.has_param("offset") ? std::stol(req.get_param_value("offset")) : 0;
        long limit  = req.has_param("limit")  ? std::stol(req.get_param_value("limit"))  : 50;
        if (offset < 0) offset = 0;
        if (limit  < 0) limit  = 0;

        nlohmann::json games = nlohmann::json::array();
        for (long i = offset; i < total && i < offset + limit; ++i) {
            nlohmann::json row = *rows[i];
            row["flagged"] = flagged.count(row.value("uuid", "")) > 0;
            games.push_back(std::move(row));
        }

        json_response(res, {{"total", total}, {"offset", offset},
                            {"limit", limit}, {"games", games}});
    }

    // POST /api/games/eval — evaluate a single replay position with the
    // checkpoint that played the game. Body:
    //   { "checkpoint": "<path>", "player": <seat>,
    //     "position": { variant, coefficients, current_player, boards } }
    // The position's variant must match this server's variant (parse_position
    // enforces it), so a 2v2 game can only be evaluated by the 2v2 server.
    // Returns { nn_value, player, checkpoint } where nn_value is the win-rate
    // the checkpoint assigns to `player` (0..1).
    void handle_games_eval(const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) { error_response(res, "invalid JSON body"); return; }

        std::string checkpoint = body.value("checkpoint", "");
        if (checkpoint.empty()) { error_response(res, "missing checkpoint"); return; }
        if (!body.contains("position") || !body["position"].is_object()) {
            error_response(res, "missing position"); return;
        }

        BoardStateT<Traits> board;
        std::string err;
        if (!parse_position(body["position"], board, err)) {
            error_response(res, err); return;
        }

        int player = body.value("player", -1);
        if (player < 0) player = static_cast<int>(board.current_player);
        if (player >= Traits::kNumPlayers) { error_response(res, "player out of range"); return; }

        float value = 0.0f;
        if (!sessions_.evaluate_position(checkpoint, board, player, value, err)) {
            // libtorch load failures carry a full backtrace in what(); keep only
            // the first line so the UI shows a readable message.
            std::string brief = err.substr(0, err.find('\n'));
            error_response(res, "could not load checkpoint: " + brief); return;
        }
        json_response(res, {{"nn_value", value}, {"player", player},
                            {"checkpoint", checkpoint}});
    }

    // POST /api/games/eval_moves — evaluate every possible next move from a
    // replay position with the game's checkpoint. Body:
    //   { "checkpoint": "<path>", "player": <seat>,
    //     "dice": [d0..d4],            // final dice that turn (for the valid filter)
    //     "position": { variant, coefficients, current_player, boards } }
    // The position is the board *before* the move; `player` is the seat about to
    // play. Returns every (column,row,score) afterstate the solver enumerates
    // (including the Turbo column) with the model's evaluation, each tagged
    // `valid_for_dice` = whether that (row, score) is reachable with the supplied
    // dice and is a legal final placement (the placement-step subset shown in
    // debug mode), and `requires_roll` = whether it is a Turbo afterstate that
    // can only be reached with a roll left. `dice` may be omitted (all invalid).
    void handle_games_eval_moves(const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) { error_response(res, "invalid JSON body"); return; }

        std::string checkpoint = body.value("checkpoint", "");
        if (checkpoint.empty()) { error_response(res, "missing checkpoint"); return; }
        if (!body.contains("position") || !body["position"].is_object()) {
            error_response(res, "missing position"); return;
        }

        BoardStateT<Traits> board;
        std::string err;
        if (!parse_position(body["position"], board, err)) {
            error_response(res, err); return;
        }

        int player = body.value("player", -1);
        if (player < 0) player = static_cast<int>(board.current_player);
        if (player >= Traits::kNumPlayers) { error_response(res, "player out of range"); return; }

        int8_t dice[kNumDice];
        bool have_dice = false;
        if (body.contains("dice") && body["dice"].is_array() &&
            body["dice"].size() == kNumDice) {
            have_dice = true;
            for (int i = 0; i < kNumDice; ++i)
                dice[i] = static_cast<int8_t>(body["dice"][i].get<int>());
        }

        std::vector<typename Sessions::MoveEval> moves;
        if (!sessions_.evaluate_moves(checkpoint, board, player,
                                      have_dice ? dice : nullptr, moves, err)) {
            std::string brief = err.substr(0, err.find('\n'));
            error_response(res, "could not evaluate moves: " + brief); return;
        }

        nlohmann::json arr = nlohmann::json::array();
        for (const auto& m : moves) {
            arr.push_back({
                {"column",         static_cast<int>(m.placement.column)},
                {"row",            static_cast<int>(m.placement.row)},
                {"column_name",    column_name(static_cast<int>(m.placement.column))},
                {"row_name",       row_name(static_cast<int>(m.placement.row))},
                {"score",          static_cast<int>(m.score)},
                {"eval_value",     m.eval_value},
                {"valid_for_dice", m.valid_for_dice},
                {"requires_roll",  m.requires_roll},
            });
        }
        json_response(res, {{"moves", arr}, {"player", player},
                            {"checkpoint", checkpoint}});
    }

    // POST /api/games/accuracy_turn — score one recorded turn's human decisions
    // against the checkpoint. Body:
    //   { "checkpoint", "player", "position" (board before the turn),
    //     "initial_dice": [d0..d4],
    //     "holds": [ { "mask": m, "dice_after": [d0..d4] }, ... ],
    //     "placement": { "column", "row" }, "score" }
    // Returns { hold_deltas: [Δwin-prob...], place_delta, has_place,
    // roll_lucks: [-100..100 per roll] } where each delta is the win-probability
    // the player gave up vs the solver's best action at that decision point, and
    // each roll_luck is a mean-zero index in [-100,100] = how the dice deviated
    // from expectation, scaled by the roll's [worst,best] spread (0 = rolled as
    // expected, + = luckier than expected, - = unluckier).
    void handle_games_accuracy_turn(const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) { error_response(res, "invalid JSON body"); return; }

        std::string checkpoint = body.value("checkpoint", "");
        if (checkpoint.empty()) { error_response(res, "missing checkpoint"); return; }
        if (!body.contains("position") || !body["position"].is_object()) {
            error_response(res, "missing position"); return;
        }

        BoardStateT<Traits> board;
        std::string err;
        if (!parse_position(body["position"], board, err)) {
            error_response(res, err); return;
        }

        int player = body.value("player", -1);
        if (player < 0) player = static_cast<int>(board.current_player);
        if (player >= Traits::kNumPlayers) { error_response(res, "player out of range"); return; }

        auto read_dice = [](const nlohmann::json& arr, int8_t out[kNumDice]) -> bool {
            if (!arr.is_array() || arr.size() != static_cast<size_t>(kNumDice)) return false;
            for (int i = 0; i < kNumDice; ++i) out[i] = static_cast<int8_t>(arr[i].get<int>());
            return true;
        };

        int8_t initial[kNumDice];
        if (!body.contains("initial_dice") || !read_dice(body["initial_dice"], initial)) {
            error_response(res, "missing/invalid initial_dice"); return;
        }

        std::vector<std::array<int8_t, kNumDice>> reroll_dice;
        std::vector<uint8_t> hold_masks;
        if (body.contains("holds") && body["holds"].is_array()) {
            for (const auto& h : body["holds"]) {
                std::array<int8_t, kNumDice> da{};
                if (!read_dice(h.value("dice_after", nlohmann::json::array()), da.data()))
                    continue;  // skip malformed reroll step
                reroll_dice.push_back(da);
                hold_masks.push_back(static_cast<uint8_t>(h.value("mask", 0)));
            }
        }

        Placement placement{
            static_cast<int8_t>(body.contains("placement") ? body["placement"].value("column", 0) : 0),
            static_cast<int8_t>(body.contains("placement") ? body["placement"].value("row", 0) : 0)};
        int8_t place_score = static_cast<int8_t>(body.value("score", 0));

        std::vector<double> hold_deltas;
        std::vector<double> roll_lucks;
        double place_delta = 0.0;
        bool   has_place   = false;
        if (!sessions_.evaluate_turn_accuracy(checkpoint, board, player, initial,
                                              reroll_dice, hold_masks, placement,
                                              place_score, hold_deltas, place_delta,
                                              has_place, roll_lucks, err)) {
            std::string brief = err.substr(0, err.find('\n'));
            error_response(res, "could not evaluate accuracy: " + brief); return;
        }

        nlohmann::json hd = nlohmann::json::array();
        for (double d : hold_deltas) hd.push_back(d);
        nlohmann::json rl = nlohmann::json::array();
        for (double d : roll_lucks) rl.push_back(d);
        nlohmann::json out = {{"hold_deltas", hd}, {"has_place", has_place},
                              {"roll_lucks", rl}, {"player", player}};
        if (has_place) out["place_delta"] = place_delta;
        json_response(res, out);
    }

    // POST /api/games/accuracy_game — score a whole recorded game in one request
    // (one batched NN pass over every turn's afterstates). Body:
    //   { "checkpoint", "turns": [ {player, position, initial_dice, holds,
    //     placement, score}, ... ] }   (each turn shaped like accuracy_turn)
    // Returns { turns: [ {hold_deltas, place_delta, has_place, roll_lucks,
    // player}, ... ] } in the same order.
    void handle_games_accuracy_game(const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) { error_response(res, "invalid JSON body"); return; }

        std::string checkpoint = body.value("checkpoint", "");
        if (checkpoint.empty()) { error_response(res, "missing checkpoint"); return; }
        if (!body.contains("turns") || !body["turns"].is_array()) {
            error_response(res, "missing turns"); return;
        }

        auto read_dice = [](const nlohmann::json& arr, int8_t out[kNumDice]) -> bool {
            if (!arr.is_array() || arr.size() != static_cast<size_t>(kNumDice)) return false;
            for (int i = 0; i < kNumDice; ++i) out[i] = static_cast<int8_t>(arr[i].get<int>());
            return true;
        };

        std::vector<typename Sessions::TurnEval> turns;
        std::string err;
        for (const auto& jt : body["turns"]) {
            typename Sessions::TurnEval te;
            if (!jt.contains("position") || !jt["position"].is_object() ||
                !parse_position(jt["position"], te.board, err)) {
                error_response(res, "bad turn position: " + err); return;
            }
            te.player = jt.value("player", static_cast<int>(te.board.current_player));
            if (te.player < 0 || te.player >= Traits::kNumPlayers) {
                error_response(res, "player out of range"); return;
            }
            if (!jt.contains("initial_dice") || !read_dice(jt["initial_dice"], te.initial_dice)) {
                error_response(res, "missing/invalid initial_dice"); return;
            }
            if (jt.contains("holds") && jt["holds"].is_array()) {
                for (const auto& h : jt["holds"]) {
                    std::array<int8_t, kNumDice> da{};
                    if (!read_dice(h.value("dice_after", nlohmann::json::array()), da.data()))
                        continue;  // skip malformed reroll step
                    te.reroll_dice.push_back(da);
                    te.hold_masks.push_back(static_cast<uint8_t>(h.value("mask", 0)));
                }
            }
            te.placement = Placement{
                static_cast<int8_t>(jt.contains("placement") ? jt["placement"].value("column", 0) : 0),
                static_cast<int8_t>(jt.contains("placement") ? jt["placement"].value("row", 0) : 0)};
            te.place_score = static_cast<int8_t>(jt.value("score", 0));
            turns.push_back(std::move(te));
        }

        std::vector<typename Sessions::TurnResult> results;
        if (!sessions_.evaluate_game_accuracy(checkpoint, turns, results, err)) {
            std::string brief = err.substr(0, err.find('\n'));
            error_response(res, "could not evaluate game: " + brief); return;
        }

        nlohmann::json jturns = nlohmann::json::array();
        for (const auto& tr : results) {
            nlohmann::json hd = nlohmann::json::array();
            for (double d : tr.hold_deltas) hd.push_back(d);
            nlohmann::json rl = nlohmann::json::array();
            for (double d : tr.roll_lucks) rl.push_back(d);
            nlohmann::json jt = {{"hold_deltas", hd}, {"has_place", tr.has_place},
                                 {"roll_lucks", rl}, {"player", tr.player}};
            if (tr.has_place) jt["place_delta"] = tr.place_delta;
            jturns.push_back(jt);
        }
        json_response(res, {{"turns", jturns}});
    }

    void handle_game_record(const httplib::Request& req, httplib::Response& res) {
        if (games_dir_.empty()) { error_response(res, "no games directory configured", 404); return; }
        std::string uuid = req.matches[1].str();
        // The route regex already restricts to [0-9a-fA-F-]; double-check to be
        // safe against path traversal.
        for (char c : uuid) {
            if (!(std::isxdigit(static_cast<unsigned char>(c)) || c == '-')) {
                error_response(res, "invalid id"); return;
            }
        }
        namespace fs = std::filesystem;
        fs::path p = fs::path(games_dir_) / "games" / (uuid + ".json");
        std::ifstream f(p);
        if (!f) { error_response(res, "game not found", 404); return; }
        std::ostringstream ss;
        ss << f.rdbuf();

        // Merge the flag from flagged.jsonl in case the per-game file wasn't
        // stamped (e.g. flagged by a different process before the file write).
        if (read_flagged_uuids().count(uuid)) {
            try {
                nlohmann::json full = nlohmann::json::parse(ss.str());
                full["flagged"] = true;
                res.status = 200;
                res.set_content(full.dump(), "application/json");
                return;
            } catch (...) { /* fall through to raw passthrough */ }
        }
        res.status = 200;
        res.set_content(ss.str(), "application/json");
    }

    // ---- Matchmaking / shared-session state ----
    // A pending or completed "play with a friend" pairing.
    struct MatchTicket {
        int     id             = 0;
        int64_t created_ms     = 0;
        int     matched_session = -1;  // set once paired
        int     seat            = -1;  // this ticket-holder's seat once paired
    };
    // Grace period before a silent human seat is handed to the NN. Generous so a
    // player who briefly switches apps / backgrounds the tab (mobile timers get
    // throttled) can return and keep playing; the 5s heartbeat keeps an active
    // player well inside it.
    static constexpr int64_t kSharedTimeoutMs = 120000;  // 2 min → NN takeover
    static constexpr int64_t kMatchTtlMs      = 600000;  // stale waiter/resolved TTL
    static constexpr int64_t kLongPollMs      = 25000;   // max hold for /longpoll

    std::vector<MatchTicket>      waiting_;    // FIFO of unmatched joiners
    std::map<int, MatchTicket>    resolved_;   // ticket id → match result (for poll)
    int                           next_ticket_id_ = 1;
    std::mt19937_64               match_rng_{std::random_device{}()};
    std::mutex                    match_mutex_;

    httplib::Server server_;
    Sessions&       sessions_;
    std::string     log_dir_;
    std::string     static_dir_;
    std::string     checkpoints_dir_;
    std::string     games_dir_;
    TournamentManagerT<Traits>* tournament_;
    GameRecorder*   recorder_;
    int             port_;
};

// Backward-compat aliases.
using UIServer    = UIServerT<Yams1v1>;
using UIServer2v2 = UIServerT<Yams2v2>;
