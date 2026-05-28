#include "eval/eval_logging.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

static std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

void log_evaluation(const std::string& log_dir, int training_step,
                     const EvalResult& result) {
    namespace fs = std::filesystem;

    if (!log_dir.empty()) fs::create_directories(log_dir);

    std::string path = log_dir.empty()
        ? "eval_log.csv"
        : log_dir + "/eval_log.csv";

    bool write_header = !fs::exists(path) || fs::file_size(path) == 0;

    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("log_evaluation: cannot open " + path);

    if (write_header) {
        f << "timestamp,step,games,nn_wins,heur_wins,draws,"
             "win_rate,wr_as_p0,wr_as_p1,avg_margin\n";
    }

    f << current_timestamp()       << ','
      << training_step             << ','
      << result.total_games        << ','
      << result.nn_wins            << ','
      << result.heuristic_wins     << ','
      << result.draws              << ','
      << result.nn_win_rate()      << ','
      << result.nn_win_rate_as_p0()<< ','
      << result.nn_win_rate_as_p1()<< ','
      << result.avg_duel_margin    << '\n';
}

void log_lr_backoff(const std::string& log_dir, int training_step,
                    double old_lr, double new_lr, double best_win_rate) {
    namespace fs = std::filesystem;

    if (!log_dir.empty()) fs::create_directories(log_dir);

    std::string path = log_dir.empty()
        ? "lr_backoff_log.csv"
        : log_dir + "/lr_backoff_log.csv";

    bool write_header = !fs::exists(path) || fs::file_size(path) == 0;

    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("log_lr_backoff: cannot open " + path);

    if (write_header) {
        f << "timestamp,step,old_lr,new_lr,best_win_rate\n";
    }

    f << current_timestamp() << ','
      << training_step        << ','
      << old_lr               << ','
      << new_lr               << ','
      << best_win_rate        << '\n';

    std::cout << "[lr-backoff] step " << training_step
              << ": win rate plateaued (best=" << best_win_rate
              << "), lr " << old_lr << " -> " << new_lr << std::endl;
}
