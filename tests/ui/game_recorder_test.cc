#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ui/game_recorder.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// Count non-empty lines in a JSONL file (0 if it does not exist).
int count_lines(const std::string& path) {
    std::ifstream f(path);
    if (!f) return 0;
    int n = 0;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) ++n;
    return n;
}

std::vector<json> read_lines(const std::string& path) {
    std::vector<json> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) out.push_back(json::parse(line));
    return out;
}

class GameRecorderTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = (fs::temp_directory_path() /
                ("yams_rec_" + std::to_string(::testing::UnitTest::GetInstance()
                                                  ->random_seed()) +
                 "_" + std::to_string(reinterpret_cast<uintptr_t>(this))))
                   .string();
        fs::remove_all(dir_);
    }
    void TearDown() override { fs::remove_all(dir_); }

    GameStartInfo make_start() {
        GameStartInfo info;
        info.variant         = "1v1";
        info.num_players     = 2;
        info.human_seats     = {0};
        info.coefficients    = {1, 2, 3, 4, 5, 6};
        info.ip              = "203.0.113.7";
        info.x_forwarded_for = "198.51.100.1";
        info.user_agent      = "UnitTest/1.0";
        return info;
    }

    GameOutcome make_outcome() {
        GameOutcome out;
        out.result               = 1.0;  // Team 0 win
        out.winner_team          = 0;
        out.human_team           = 0;
        out.total_margin         = 42;
        out.column_margins       = {10, 0, 5, 7, 20, 0};
        out.player_column_scores = {{30, 0, 12, 18, 40, 0}, {20, 0, 7, 11, 20, 0}};
        return out;
    }

    std::string dir_;
    std::string started()  { return (fs::path(dir_) / "started.jsonl").string(); }
    std::string finished() { return (fs::path(dir_) / "finished.jsonl").string(); }
    std::string game_file(const std::string& uuid) {
        return (fs::path(dir_) / "games" / (uuid + ".json")).string();
    }
};

TEST_F(GameRecorderTest, StartGameAppendsStartedLine) {
    GameRecorder rec(dir_, "checkpoints/mlp/step_50000", 8091);
    std::string uuid = rec.start_game(1, make_start());

    EXPECT_FALSE(uuid.empty());
    ASSERT_EQ(count_lines(started()), 1);

    auto lines = read_lines(started());
    EXPECT_EQ(lines[0]["uuid"], uuid);
    EXPECT_EQ(lines[0]["port"], 8091);
    EXPECT_EQ(lines[0]["variant"], "1v1");
    EXPECT_EQ(lines[0]["checkpoint"], "checkpoints/mlp/step_50000");
    EXPECT_EQ(lines[0]["ip"], "203.0.113.7");
    EXPECT_EQ(lines[0]["x_forwarded_for"], "198.51.100.1");
    EXPECT_EQ(lines[0]["human_seats"], json::array({0}));
    EXPECT_EQ(lines[0]["coefficients"], json::array({1, 2, 3, 4, 5, 6}));

    // No finished record yet (this game is still "in progress" → drop case).
    EXPECT_EQ(count_lines(finished()), 0);
}

TEST_F(GameRecorderTest, FinishGameWritesRecordAndIndex) {
    GameRecorder rec(dir_, "ckpt", 8090);
    std::string uuid = rec.start_game(1, make_start());

    json final_state = {{"num_players", 2}, {"history", json::array()}};
    rec.finish_game(1, final_state, make_outcome());

    // Per-game file.
    ASSERT_TRUE(fs::exists(game_file(uuid)));
    std::ifstream gf(game_file(uuid));
    json full; gf >> full;
    EXPECT_EQ(full["uuid"], uuid);
    EXPECT_EQ(full["total_margin"], 42);
    EXPECT_EQ(full["column_margins"], json::array({10, 0, 5, 7, 20, 0}));
    EXPECT_EQ(full["player_column_scores"][0], json::array({30, 0, 12, 18, 40, 0}));
    EXPECT_EQ(full["final_state"]["num_players"], 2);
    EXPECT_TRUE(full.contains("user_agent"));

    // Finished index.
    ASSERT_EQ(count_lines(finished()), 1);
    auto fl = read_lines(finished());
    EXPECT_EQ(fl[0]["uuid"], uuid);
    EXPECT_EQ(fl[0]["winner_team"], 0);
    EXPECT_EQ(fl[0]["human_team"], 0);
    EXPECT_TRUE(fl[0].contains("duration_ms"));
}

TEST_F(GameRecorderTest, FinishGameIsIdempotent) {
    GameRecorder rec(dir_, "ckpt", 8090);
    rec.start_game(1, make_start());
    json final_state = {{"num_players", 2}};

    rec.finish_game(1, final_state, make_outcome());
    rec.finish_game(1, final_state, make_outcome());  // second call: no-op

    EXPECT_EQ(count_lines(finished()), 1);
}

TEST_F(GameRecorderTest, ForgottenGameIsNotRecorded) {
    GameRecorder rec(dir_, "ckpt", 8090);
    rec.start_game(7, make_start());
    rec.forget(7);

    rec.finish_game(7, json::object(), make_outcome());

    EXPECT_EQ(count_lines(started()), 1);    // start still counts toward drops
    EXPECT_EQ(count_lines(finished()), 0);   // but it was never finished
}

}  // namespace
