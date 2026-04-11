#include "config/config_printer.h"

#include "self_play/training_data.h"  // TDMode

namespace {

const char* td_mode_str(TDMode m) {
    switch (m) {
        case TDMode::kMC:       return "mc";
        case TDMode::kTD0:      return "td0";
        case TDMode::kTDLambda: return "tdlambda";
    }
    return "unknown";
}

}  // namespace

void print_config(const AppConfig& cfg, std::ostream& out) {
    const TrainingConfig& tc = cfg.training;
    const SelfPlayConfig& sp = tc.self_play;
    const ModelConfig&    m  = tc.model;

    out << "num_steps: " << cfg.num_steps << "\n";
    out << "training:\n";
    out << "  replay_capacity: "      << tc.replay_capacity     << "\n";
    out << "  min_buffer_size: "      << tc.min_buffer_size      << "\n";
    out << "  train_batch_size: "     << tc.train_batch_size     << "\n";
    out << "  train_steps_per_collect: " << tc.train_steps_per_collect << "\n";
    out << "  model_swap_interval: "  << tc.model_swap_interval  << "\n";
    out << "  checkpoint_interval: "  << tc.checkpoint_interval  << "\n";
    out << "  max_checkpoints: "      << tc.max_checkpoints      << "\n";
    out << "  td_mode: "              << td_mode_str(tc.td_mode) << "\n";
    out << "  td_lambda: "            << tc.td_lambda            << "\n";
    out << "  initial_temperature: "  << tc.initial_temperature  << "\n";
    out << "  min_temperature: "      << tc.min_temperature      << "\n";
    out << "  temperature_decay: "    << tc.temperature_decay    << "\n";
    out << "  initial_epsilon: "      << tc.initial_epsilon      << "\n";
    out << "  eval_interval: "        << tc.eval_interval        << "\n";
    out << "  eval_games: "           << tc.eval_games           << "\n";
    out << "  checkpoint_dir: "       << tc.checkpoint_dir       << "\n";
    out << "  log_dir: "              << tc.log_dir              << "\n";
    out << "  log_path: "             << tc.log_path             << "\n";
    out << "  self_play:\n";
    out << "    max_inference_batch: " << sp.max_inference_batch << "\n";
    out << "    min_games_per_batch: " << sp.min_games_per_batch << "\n";
    out << "    batch_timeout_ms: "    << sp.batch_timeout_ms    << "\n";
    out << "    num_workers: "         << sp.num_workers         << "\n";
    out << "    num_games: "           << sp.num_games           << "\n";
    out << "    num_coordinators: "    << sp.num_coordinators    << "\n";
    out << "  model:\n";
    out << "    input_size: "    << m.input_size    << "\n";
    out << "    hidden_layers: " << m.hidden_layers << "\n";
    out << "    hidden_width: "  << m.hidden_width  << "\n";
    out << "    learning_rate: " << m.learning_rate << "\n";
}
