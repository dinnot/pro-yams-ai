#include "ui/game_recorder.h"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>

namespace {

int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
               system_clock::now().time_since_epoch())
        .count();
}

std::string iso8601_utc(int64_t ms) {
    std::time_t secs = static_cast<std::time_t>(ms / 1000);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &secs);
#else
    gmtime_r(&secs, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

// RFC 4122 version-4 UUID from a thread-local PRNG seeded once per thread.
std::string make_uuid_v4() {
    static thread_local std::mt19937_64 rng(
        std::random_device{}() ^
        static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()));

    uint64_t hi = rng();
    uint64_t lo = rng();
    // Set version (4) and variant (10xx) bits.
    hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
    lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;

    char buf[37];
    std::snprintf(
        buf, sizeof(buf),
        "%08x-%04x-%04x-%04x-%012llx",
        static_cast<unsigned>(hi >> 32),
        static_cast<unsigned>((hi >> 16) & 0xFFFF),
        static_cast<unsigned>(hi & 0xFFFF),
        static_cast<unsigned>(lo >> 48),
        static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
    return std::string(buf);
}

}  // namespace

GameRecorder::GameRecorder(std::string games_dir, std::string checkpoint_label,
                           int port)
    : games_dir_(std::move(games_dir)),
      checkpoint_(std::move(checkpoint_label)),
      port_(port) {
    namespace fs = std::filesystem;
    games_subdir_  = (fs::path(games_dir_) / "games").string();
    started_path_  = (fs::path(games_dir_) / "started.jsonl").string();
    finished_path_ = (fs::path(games_dir_) / "finished.jsonl").string();
    flagged_path_  = (fs::path(games_dir_) / "flagged.jsonl").string();

    std::error_code ec;
    fs::create_directories(games_subdir_, ec);  // also creates games_dir_
}

void GameRecorder::append_line(const std::string& path, const std::string& line) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;  // best-effort: never break gameplay on a logging error
    std::string buf = line;
    buf.push_back('\n');
    ssize_t n = ::write(fd, buf.data(), buf.size());
    (void)n;
    ::close(fd);
}

std::string GameRecorder::start_game(int session_id, const GameStartInfo& info) {
    StartedMeta meta;
    meta.uuid            = make_uuid_v4();
    meta.start_ts_ms     = now_ms();
    meta.variant         = info.variant;
    meta.num_players     = info.num_players;
    meta.human_seats     = info.human_seats;
    meta.coefficients    = info.coefficients;
    meta.ip              = info.ip;
    meta.x_forwarded_for = info.x_forwarded_for;
    meta.user_agent      = info.user_agent;

    nlohmann::json j;
    j["uuid"]            = meta.uuid;
    j["session_id"]      = session_id;
    j["port"]            = port_;
    j["start_ts_ms"]     = meta.start_ts_ms;
    j["start_iso"]       = iso8601_utc(meta.start_ts_ms);
    j["variant"]         = meta.variant;
    j["num_players"]     = meta.num_players;
    j["human_seats"]     = meta.human_seats;
    j["coefficients"]    = meta.coefficients;
    j["checkpoint"]      = checkpoint_;
    j["ip"]              = meta.ip;
    j["x_forwarded_for"] = meta.x_forwarded_for;
    j["user_agent"]      = meta.user_agent;

    std::string uuid = meta.uuid;
    {
        std::lock_guard<std::mutex> lock(mu_);
        active_[session_id] = std::move(meta);
        session_uuid_[session_id] = uuid;
        // Bound the retained-uuid window. session_ids increase monotonically,
        // so erasing the lowest keys drops the oldest games first.
        constexpr size_t kMaxRetainedUuids = 8192;
        while (session_uuid_.size() > kMaxRetainedUuids)
            session_uuid_.erase(session_uuid_.begin());
        append_line(started_path_, j.dump());
    }
    return uuid;
}

void GameRecorder::finish_game(int session_id,
                               const nlohmann::json& final_state,
                               const GameOutcome& outcome) {
    StartedMeta meta;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = active_.find(session_id);
        if (it == active_.end()) return;  // never started, or already finished
        meta = std::move(it->second);
        active_.erase(it);
    }

    const int64_t finish_ts = now_ms();

    // Summary line for the query index.
    nlohmann::json summary;
    summary["uuid"]                 = meta.uuid;
    summary["port"]                 = port_;
    summary["start_ts_ms"]          = meta.start_ts_ms;
    summary["finish_ts_ms"]         = finish_ts;
    summary["duration_ms"]          = finish_ts - meta.start_ts_ms;
    summary["variant"]              = meta.variant;
    summary["num_players"]          = meta.num_players;
    summary["human_seats"]          = meta.human_seats;
    summary["human_team"]           = outcome.human_team;
    summary["coefficients"]         = meta.coefficients;
    summary["checkpoint"]           = checkpoint_;
    summary["result"]               = outcome.result;
    summary["winner_team"]          = outcome.winner_team;
    summary["total_margin"]         = outcome.total_margin;
    summary["column_margins"]       = outcome.column_margins;
    summary["player_column_scores"] = outcome.player_column_scores;

    // Full record = summary fields + start metadata + complete final state.
    nlohmann::json full   = summary;
    full["start_iso"]       = iso8601_utc(meta.start_ts_ms);
    full["finish_iso"]      = iso8601_utc(finish_ts);
    full["ip"]              = meta.ip;
    full["x_forwarded_for"] = meta.x_forwarded_for;
    full["user_agent"]      = meta.user_agent;
    full["final_state"]     = final_state;

    namespace fs = std::filesystem;
    std::string file_path = (fs::path(games_subdir_) / (meta.uuid + ".json")).string();
    {
        std::ofstream out(file_path, std::ios::trunc);
        if (out) out << full.dump();
    }

    append_line(finished_path_, summary.dump());
}

bool GameRecorder::flag_game(int session_id, const std::string& note) {
    std::string uuid;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = session_uuid_.find(session_id);
        if (it == session_uuid_.end()) return false;
        uuid = it->second;
    }

    const int64_t ts = now_ms();

    nlohmann::json line;
    line["uuid"]    = uuid;
    line["port"]    = port_;
    line["ts_ms"]   = ts;
    line["iso"]     = iso8601_utc(ts);
    if (!note.empty()) line["note"] = note;
    append_line(flagged_path_, line.dump());

    // Best-effort: stamp the per-game record so the replay viewer can show the
    // flag even without consulting flagged.jsonl. The play and admin servers
    // share games_dir, so this file is also what the admin UI reads.
    namespace fs = std::filesystem;
    std::string file_path =
        (fs::path(games_subdir_) / (uuid + ".json")).string();
    try {
        nlohmann::json full;
        {
            std::ifstream in(file_path);
            if (in) in >> full;
        }
        if (full.is_object()) {
            full["flagged"] = true;
            if (!note.empty()) full["flag_note"] = note;
            std::ofstream out(file_path, std::ios::trunc);
            if (out) out << full.dump();
        }
    } catch (...) {
        // never let a logging error propagate; flagged.jsonl already has it.
    }
    return true;
}

void GameRecorder::forget(int session_id) {
    std::lock_guard<std::mutex> lock(mu_);
    active_.erase(session_id);
    session_uuid_.erase(session_id);
}
