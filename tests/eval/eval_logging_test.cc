#include <gtest/gtest.h>

#include "eval/eval_logging.h"
#include "eval/evaluator.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
static EvalResult make_result(int total, int nn_wins, int heur_wins, int draws) {
    EvalResult r{};
    r.total_games    = total;
    r.nn_wins        = nn_wins;
    r.heuristic_wins = heur_wins;
    r.draws          = draws;
    r.games_as_p0    = total / 2;
    r.games_as_p1    = total - total / 2;
    r.nn_wins_as_p0  = nn_wins / 2;
    r.nn_wins_as_p1  = nn_wins - nn_wins / 2;
    r.avg_duel_margin = nn_wins > 0 ? 100.0 : 0.0;
    return r;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(EvalLoggingTest, CreatesFileWithHeader) {
    fs::path tmp = fs::temp_directory_path() / "eval_log_test_header";
    fs::remove_all(tmp);

    EvalResult r = make_result(10, 6, 4, 0);
    log_evaluation(tmp.string(), 1000, r);

    fs::path csv = tmp / "eval_log.csv";
    ASSERT_TRUE(fs::exists(csv));

    std::ifstream f(csv);
    std::string header;
    std::getline(f, header);
    EXPECT_NE(header.find("timestamp"), std::string::npos);
    EXPECT_NE(header.find("step"), std::string::npos);
    EXPECT_NE(header.find("win_rate"), std::string::npos);

    fs::remove_all(tmp);
}

TEST(EvalLoggingTest, AppendsMultipleRows) {
    fs::path tmp = fs::temp_directory_path() / "eval_log_test_append";
    fs::remove_all(tmp);

    for (int step = 1000; step <= 3000; step += 1000) {
        EvalResult r = make_result(10, 5, 5, 0);
        log_evaluation(tmp.string(), step, r);
    }

    fs::path csv = tmp / "eval_log.csv";
    std::ifstream f(csv);
    int lines = 0;
    std::string line;
    while (std::getline(f, line)) ++lines;
    // 1 header + 3 data rows
    EXPECT_EQ(lines, 4);

    fs::remove_all(tmp);
}

TEST(EvalLoggingTest, StepValueInRow) {
    fs::path tmp = fs::temp_directory_path() / "eval_log_test_step";
    fs::remove_all(tmp);

    EvalResult r = make_result(20, 10, 9, 1);
    log_evaluation(tmp.string(), 42000, r);

    fs::path csv = tmp / "eval_log.csv";
    std::ifstream f(csv);
    std::string header, data;
    std::getline(f, header);
    std::getline(f, data);

    EXPECT_NE(data.find("42000"), std::string::npos)
        << "Step 42000 not found in CSV row: " << data;

    fs::remove_all(tmp);
}

TEST(EvalLoggingTest, EmptyLogDirUsesDefault) {
    // With empty log_dir, should write to eval_log.csv in current dir
    fs::path csv = fs::current_path() / "eval_log.csv";
    fs::remove(csv);  // clean up if exists

    EvalResult r = make_result(10, 5, 5, 0);
    log_evaluation("", 999, r);

    EXPECT_TRUE(fs::exists(csv));
    fs::remove(csv);
}
