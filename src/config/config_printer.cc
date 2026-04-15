#include "config/config_printer.h"
#include <fstream>
#include <yaml-cpp/yaml.h>
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

void emit_config(YAML::Emitter& out, const AppConfig& cfg) {
    const TrainingConfig& tc = cfg.training;
    const SelfPlayConfig& sp = tc.self_play;
    const ModelConfig&    m  = tc.model;

    out << YAML::BeginMap;
    out << YAML::Key << "num_steps" << YAML::Value << cfg.num_steps;
    
    out << YAML::Key << "training";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "replay_capacity"      << YAML::Value << tc.replay_capacity;
    out << YAML::Key << "min_buffer_size"      << YAML::Value << tc.min_buffer_size;
    out << YAML::Key << "train_batch_size"     << YAML::Value << tc.train_batch_size;
    out << YAML::Key << "train_steps_per_collect" << YAML::Value << tc.train_steps_per_collect;
    out << YAML::Key << "model_swap_interval"  << YAML::Value << tc.model_swap_interval;
    out << YAML::Key << "checkpoint_interval"  << YAML::Value << tc.checkpoint_interval;
    out << YAML::Key << "max_checkpoints"      << YAML::Value << tc.max_checkpoints;
    out << YAML::Key << "td_mode"              << YAML::Value << td_mode_str(tc.td_mode);
    out << YAML::Key << "td_lambda"            << YAML::Value << tc.td_lambda;
    out << YAML::Key << "initial_temperature"  << YAML::Value << tc.initial_temperature;
    out << YAML::Key << "min_temperature"      << YAML::Value << tc.min_temperature;
    out << YAML::Key << "temperature_decay"    << YAML::Value << tc.temperature_decay;
    out << YAML::Key << "temperature_decay_start_step"  << YAML::Value << tc.temperature_decay_start_step;
    out << YAML::Key << "temperature_decay_start_value" << YAML::Value << tc.temperature_decay_start_value;
    out << YAML::Key << "initial_epsilon"      << YAML::Value << tc.initial_epsilon;
    out << YAML::Key << "eval_interval"        << YAML::Value << tc.eval_interval;
    out << YAML::Key << "eval_games"           << YAML::Value << tc.eval_games;
    out << YAML::Key << "checkpoint_dir"       << YAML::Value << tc.checkpoint_dir;
    out << YAML::Key << "log_dir"              << YAML::Value << tc.log_dir;
    out << YAML::Key << "log_path"             << YAML::Value << tc.log_path;
    out << YAML::Key << "debug_mode"           << YAML::Value << tc.debug_mode;
    out << YAML::Key << "initial_heuristic_weight" << YAML::Value << tc.initial_heuristic_weight;
    out << YAML::Key << "heuristic_decay_steps" << YAML::Value << tc.heuristic_decay_steps;
    out << YAML::Key << "use_duel_margin_maximization"   << YAML::Value << tc.use_duel_margin_maximization;
    out << YAML::Key << "duel_margin_maximization_scale" << YAML::Value << tc.duel_margin_maximization_scale;
    out << YAML::Key << "use_pbrs"          << YAML::Value << tc.use_pbrs;
    out << YAML::Key << "pbrs_upper_reward" << YAML::Value << tc.pbrs_upper_reward;
    out << YAML::Key << "pbrs_clean_reward" << YAML::Value << tc.pbrs_clean_reward;

    out << YAML::Key << "self_play";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "max_inference_batch" << YAML::Value << sp.max_inference_batch;
    out << YAML::Key << "min_games_per_batch" << YAML::Value << sp.min_games_per_batch;
    out << YAML::Key << "batch_timeout_ms"    << YAML::Value << sp.batch_timeout_ms;
    out << YAML::Key << "num_workers"         << YAML::Value << sp.num_workers;
    out << YAML::Key << "num_games"           << YAML::Value << sp.num_games;
    out << YAML::Key << "num_coordinators"    << YAML::Value << sp.num_coordinators;
    out << YAML::EndMap;

    out << YAML::Key << "model";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "input_size"    << YAML::Value << m.input_size;
    out << YAML::Key << "hidden_layers" << YAML::Value << m.hidden_layers;
    out << YAML::Key << "hidden_width"  << YAML::Value << m.hidden_width;
    out << YAML::Key << "learning_rate"     << YAML::Value << m.learning_rate;
    out << YAML::Key << "output_activation" << YAML::Value << m.output_activation;
    out << YAML::Key << "loss_function"     << YAML::Value << m.loss_function;
    out << YAML::Key << "architecture"      << YAML::Value << m.architecture;
    out << YAML::EndMap;

    out << YAML::EndMap; // training
    out << YAML::EndMap; // root
}

}  // namespace

void print_config(const AppConfig& cfg, std::ostream& out) {
    YAML::Emitter emitter;
    emitter << YAML::BeginDoc;
    emit_config(emitter, cfg);
    emitter << YAML::EndDoc;
    out << emitter.c_str() << "\n";
}

void save_config(const AppConfig& cfg, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open config file for writing: " + path);
    }
    YAML::Emitter emitter;
    emitter << YAML::BeginDoc;
    emit_config(emitter, cfg);
    emitter << YAML::EndDoc;
    out << emitter.c_str() << "\n";
}
