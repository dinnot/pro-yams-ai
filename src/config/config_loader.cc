#include "config/config_loader.h"

#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "self_play/training_data.h"  // TDMode

static TDMode parse_td_mode(const std::string& s) {
    if (s == "mc")        return TDMode::kMC;
    if (s == "td0")       return TDMode::kTD0;
    if (s == "tdlambda")  return TDMode::kTDLambda;
    throw std::runtime_error("Unknown td_mode value: '" + s +
                             "' (expected: mc, td0, tdlambda)");
}

namespace {

template <typename T>
void maybe_assign(const YAML::Node& node, const char* key, T& out) {
    if (node[key]) out = node[key].as<T>();
}

void load_self_play_config(const YAML::Node& n, SelfPlayConfig& sp) {
    maybe_assign(n, "max_inference_batch", sp.max_inference_batch);
    maybe_assign(n, "min_games_per_batch", sp.min_games_per_batch);
    maybe_assign(n, "batch_timeout_ms",    sp.batch_timeout_ms);
    maybe_assign(n, "num_workers",         sp.num_workers);
    maybe_assign(n, "num_games",           sp.num_games);
    maybe_assign(n, "num_coordinators",    sp.num_coordinators);
    maybe_assign(n, "debug_mode",          sp.debug_mode);
    maybe_assign(n, "debug_log_path",      sp.debug_log_path);
}

void load_model_config(const YAML::Node& n, ModelConfig& m) {
    maybe_assign(n, "input_size",    m.input_size);
    maybe_assign(n, "hidden_layers", m.hidden_layers);
    maybe_assign(n, "hidden_width",  m.hidden_width);
    maybe_assign(n, "learning_rate", m.learning_rate);
}

void load_training_config(const YAML::Node& n, TrainingConfig& tc) {
    maybe_assign(n, "replay_capacity",     tc.replay_capacity);
    maybe_assign(n, "min_buffer_size",     tc.min_buffer_size);
    maybe_assign(n, "train_batch_size",    tc.train_batch_size);
    maybe_assign(n, "train_steps_per_collect", tc.train_steps_per_collect);
    maybe_assign(n, "model_swap_interval", tc.model_swap_interval);
    maybe_assign(n, "checkpoint_interval", tc.checkpoint_interval);
    maybe_assign(n, "max_checkpoints",     tc.max_checkpoints);
    maybe_assign(n, "td_lambda",           tc.td_lambda);
    maybe_assign(n, "initial_temperature", tc.initial_temperature);
    maybe_assign(n, "min_temperature",     tc.min_temperature);
    maybe_assign(n, "temperature_decay",   tc.temperature_decay);
    maybe_assign(n, "initial_epsilon",     tc.initial_epsilon);
    maybe_assign(n, "eval_interval",       tc.eval_interval);
    maybe_assign(n, "eval_games",          tc.eval_games);
    maybe_assign(n, "checkpoint_dir",      tc.checkpoint_dir);
    maybe_assign(n, "log_dir",             tc.log_dir);
    maybe_assign(n, "log_path",            tc.log_path);
    maybe_assign(n, "debug_mode",          tc.debug_mode);
    maybe_assign(n, "initial_heuristic_weight", tc.initial_heuristic_weight);
    maybe_assign(n, "heuristic_decay_steps", tc.heuristic_decay_steps);
    if (n["td_mode"])   tc.td_mode = parse_td_mode(n["td_mode"].as<std::string>());
    if (n["self_play"]) load_self_play_config(n["self_play"], tc.self_play);
    if (n["model"])     load_model_config(n["model"], tc.model);
}

}  // namespace

AppConfig default_config() {
    return AppConfig{};
}

void apply_cli_overrides(AppConfig& config, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) config.mode = argv[++i];
        else if (arg == "--config" && i + 1 < argc) { config.config_path = argv[++i]; }
        else if (arg == "--checkpoint" && i + 1 < argc) config.checkpoint_path = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) config.seed = std::stoull(argv[++i]);
        else if (arg == "--num_steps" && i + 1 < argc) config.num_steps = std::stoi(argv[++i]);
        else if (arg == "--learning_rate" && i + 1 < argc) config.training.model.learning_rate = std::stod(argv[++i]);
        else if (arg == "--hidden_layers" && i + 1 < argc) config.training.model.hidden_layers = std::stoi(argv[++i]);
        else if (arg == "--hidden_width" && i + 1 < argc) config.training.model.hidden_width = std::stoi(argv[++i]);
        else if (arg == "--td_mode" && i + 1 < argc) {
            config.training.td_mode = parse_td_mode(argv[++i]);
        }
        else if (arg == "--batch_size" && i + 1 < argc) config.training.train_batch_size = std::stoi(argv[++i]);
        else if (arg == "--train_steps_per_collect" && i + 1 < argc) config.training.train_steps_per_collect = std::stoi(argv[++i]);
        else if (arg == "--replay_buffer_size" && i + 1 < argc) config.training.replay_capacity = std::stoi(argv[++i]);
        else if (arg == "--placement_temperature" && i + 1 < argc) config.training.initial_temperature = std::stod(argv[++i]);
        else if (arg == "--num_workers" && i + 1 < argc) config.training.self_play.num_workers = std::stoi(argv[++i]);
        else if (arg == "--num_games" && i + 1 < argc) config.training.self_play.num_games = std::stoi(argv[++i]);
        else if (arg == "--num_coordinators" && i + 1 < argc) config.training.self_play.num_coordinators = std::stoi(argv[++i]);
        else if (arg == "--eval_games" && i + 1 < argc) config.training.eval_games = std::stoi(argv[++i]);
        else if (arg == "--eval_interval" && i + 1 < argc) config.training.eval_interval = std::stoi(argv[++i]);
        else if (arg == "--debug_mode" && i + 1 < argc) config.training.debug_mode = (std::stoi(argv[++i]) != 0);
        else if (arg == "--initial_heuristic_weight" && i + 1 < argc) config.training.initial_heuristic_weight = std::stod(argv[++i]);
        else if (arg == "--heuristic_decay_steps" && i + 1 < argc) config.training.heuristic_decay_steps = std::stoi(argv[++i]);
    }
}

AppConfig load_config(int argc, char* argv[]) {
    AppConfig config = default_config();
    // First pass: check for --config to load YAML base.
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config = load_config(std::string(argv[i + 1]));
            break;
        }
    }
    // Second pass: CLI overrides take precedence over YAML.
    apply_cli_overrides(config, argc, argv);
    return config;
}

AppConfig load_config(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(std::string("Failed to load config '") +
                                 path + "': " + e.what());
    }

    AppConfig cfg;
    maybe_assign(root, "num_steps", cfg.num_steps);
    if (root["training"]) load_training_config(root["training"], cfg.training);
    return cfg;
}
