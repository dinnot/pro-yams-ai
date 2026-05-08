#pragma once

#include <memory>
#include <string>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "eval/tournament.h"
#include "ui/session_manager.h"

// ---------------------------------------------------------------------------
// UIServer — HTTP server for the Pro Yams AI web interface.
//
// Serves:
//   REST API  →  /api/**
//   Frontend  →  / and all static files
// ---------------------------------------------------------------------------
class UIServer {
public:
    UIServer(int port,
             const std::string& static_dir,
             SessionManager& sessions,
             const std::string& log_dir,
             const std::string& checkpoints_dir,
             TournamentManager* tournament);

    /// Start serving (blocks until stop() is called).
    void start();

    /// Stop the server (can be called from another thread).
    void stop();

    /// Port the server listens on.
    int port() const { return port_; }

private:
    void register_routes();

    // Route handlers
    void handle_new_game    (const httplib::Request&, httplib::Response&);
    void handle_get_game    (const httplib::Request&, httplib::Response&);
    void handle_step        (const httplib::Request&, httplib::Response&);
    void handle_play_all    (const httplib::Request&, httplib::Response&);
    void handle_delete_game (const httplib::Request&, httplib::Response&);
    void handle_options     (const httplib::Request&, httplib::Response&);
    void handle_hold        (const httplib::Request&, httplib::Response&);
    void handle_place       (const httplib::Request&, httplib::Response&);
    void handle_can_reroll  (const httplib::Request&, httplib::Response&);
    void handle_tensor      (const httplib::Request&, httplib::Response&);
    void handle_training_log(const httplib::Request&, httplib::Response&);
    void handle_eval_log    (const httplib::Request&, httplib::Response&);
    void handle_log_list    (const httplib::Request&, httplib::Response&);
    void handle_models_list (const httplib::Request&, httplib::Response&);
    void handle_tournament_start (const httplib::Request&, httplib::Response&);
    void handle_tournament_status(const httplib::Request&, httplib::Response&);
    void handle_tournament_stop  (const httplib::Request&, httplib::Response&);

    // Helpers
    static int  parse_session_id(const httplib::Request& req);
    static void json_response(httplib::Response& res,
                              const nlohmann::json& j,
                              int status = 200);
    static void error_response(httplib::Response& res,
                               const std::string& msg,
                               int status = 400);
    std::string read_log_file(const std::string& dir,
                               const std::string& filename) const;

    httplib::Server server_;
    SessionManager& sessions_;
    std::string     log_dir_;
    std::string     static_dir_;
    std::string     checkpoints_dir_;
    TournamentManager* tournament_;
    int             port_;
};
