#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// GameRecorder — persists games played against the NN for later analysis.
//
// Used only by the public play server (pro_yams_play); the admin UI constructs
// its UIServerT with a null recorder. One JSON file is written per finished
// game, plus two append-only index files:
//
//   <games_dir>/started.jsonl    one line appended when a game is created
//   <games_dir>/finished.jsonl   one line appended when a game completes
//   <games_dir>/games/<uuid>.json full record (final state + move history)
//
// Multiple play-server processes (e.g. a 1v1 instance and a 2v2 instance on
// different ports) may share one games_dir. Index lines are appended with a
// single write(2) to an O_APPEND fd, which POSIX guarantees is atomic for
// writes below PIPE_BUF (4096 bytes); our lines stay well under that. Per-game
// files use a random v4 UUID, so filenames never collide across processes.
// The session_id -> StartedMeta map is per-process state.
//
// The recorder is intentionally engine-agnostic: callers that hold the game
// Traits build the final-state JSON and outcome scalars and hand them over.
// ---------------------------------------------------------------------------

// Captured when a game is created.
struct GameStartInfo {
    std::string      variant;          // "1v1" or "2v2"
    int              num_players = 0;
    std::vector<int> human_seats;      // seats controlled by a human
    std::vector<int> coefficients;     // per-column multipliers (randomized)
    std::string      ip;               // socket remote_addr
    std::string      x_forwarded_for;  // proxy header, may be empty
    std::string      user_agent;
};

// Computed when a game completes (all from Team 0's perspective unless noted).
struct GameOutcome {
    double           result = 0.0;     // 1.0 Team0 win / 0.0 Team1 win / 0.5 draw
    int              winner_team = -1; // 0, 1, or -1 for draw
    int              human_team = -1;  // team the human(s) played on
    long             total_margin = 0; // signed duel points (Team0 - Team1)
    std::vector<int> column_margins;   // per-column duel points (sums to total)
    // player_column_scores[p][c] = player p's final raw score in column c
    // (cell sum + upper-section bonus).
    std::vector<std::vector<int>> player_column_scores;
};

class GameRecorder {
public:
    // games_dir: shared output directory. checkpoint_label: identifier for the
    // NN checkpoint this server instance loaded. port: the server's port,
    // stamped on every record to disambiguate concurrent instances.
    GameRecorder(std::string games_dir, std::string checkpoint_label, int port);

    // Register a new game; appends to started.jsonl and returns its UUID.
    std::string start_game(int session_id, const GameStartInfo& info);

    // Record a completed game; writes <uuid>.json and appends to
    // finished.jsonl. No-op (and safe to call repeatedly) if the session has
    // already finished or was never started.
    void finish_game(int session_id,
                     const nlohmann::json& final_state,
                     const GameOutcome& outcome);

    // Drop in-memory state for a game that was abandoned/deleted before
    // finishing. The started.jsonl line remains, so it counts toward the drop
    // rate (started but never finished).
    void forget(int session_id);

private:
    struct StartedMeta {
        std::string      uuid;
        int64_t          start_ts_ms = 0;
        std::string      variant;
        int              num_players = 0;
        std::vector<int> human_seats;
        std::vector<int> coefficients;
        std::string      ip;
        std::string      x_forwarded_for;
        std::string      user_agent;
    };

    // Append one line (a single atomic write) to an O_APPEND file.
    static void append_line(const std::string& path, const std::string& line);

    std::string games_dir_;
    std::string games_subdir_;     // games_dir_/games
    std::string started_path_;
    std::string finished_path_;
    std::string checkpoint_;
    int         port_;

    std::mutex                    mu_;
    std::map<int, StartedMeta>    active_;  // session_id -> meta
};
