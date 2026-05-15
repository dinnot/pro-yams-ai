#include "config/config_loader.h"

#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "engine/game_traits.h"
#include "model/model_config.h"
#include "self_play/training_data.h"  // TDMode

static TDMode parse_td_mode(const std::string& s) {
    if (s == "mc")        return TDMode::kMC;
    if (s == "td0")       return TDMode::kTD0;
    if (s == "tdlambda")  return TDMode::kTDLambda;
    throw std::runtime_error("Unknown td_mode value: '" + s +
                             "' (expected: mc, td0, tdlambda)");
}

// Map "1v1" / "2v2" → kGameVariant1v1 / kGameVariant2v2.
static int parse_game_variant(const std::string& s) {
    if (s == "1v1") return kGameVariant1v1;
    if (s == "2v2") return kGameVariant2v2;
    throw std::runtime_error("Unknown game_variant value: '" + s +
                             "' (expected: 1v1, 2v2)");
}

// Pick the default tensor size for a given variant. Used as a fallback when
// the YAML / CLI doesn't override `input_size` explicitly.
static int default_input_size_for(int game_variant) {
    return (game_variant == kGameVariant2v2)
        ? Yams2v2::kTensorSize
        : Yams1v1::kTensorSize;
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
    maybe_assign(n, "input_size",         m.input_size);
    maybe_assign(n, "hidden_layers",      m.hidden_layers);
    maybe_assign(n, "hidden_width",       m.hidden_width);
    maybe_assign(n, "learning_rate",      m.learning_rate);
    maybe_assign(n, "output_activation",  m.output_activation);
    maybe_assign(n, "loss_function",      m.loss_function);
    maybe_assign(n, "architecture",       m.architecture);
    // game_variant can be set via a nested model.game_variant key too. Most
    // users will set it at the top level (training.game_variant); the nested
    // form is kept for symmetry with input_size.
    if (n["game_variant"]) {
        m.game_variant = parse_game_variant(n["game_variant"].as<std::string>());
    }
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
    maybe_assign(n, "temperature_decay_start_step", tc.temperature_decay_start_step);
    maybe_assign(n, "temperature_decay_start_value", tc.temperature_decay_start_value);
    maybe_assign(n, "initial_epsilon",     tc.initial_epsilon);
    maybe_assign(n, "eval_interval",       tc.eval_interval);
    maybe_assign(n, "eval_games",          tc.eval_games);
    maybe_assign(n, "checkpoint_dir",      tc.checkpoint_dir);
    maybe_assign(n, "log_dir",             tc.log_dir);
    maybe_assign(n, "log_path",            tc.log_path);
    maybe_assign(n, "debug_mode",          tc.debug_mode);
    maybe_assign(n, "initial_heuristic_weight", tc.initial_heuristic_weight);
    maybe_assign(n, "heuristic_decay_steps", tc.heuristic_decay_steps);
    maybe_assign(n, "heuristic_version", tc.heuristic_version);
    maybe_assign(n, "use_duel_margin_maximization",   tc.use_duel_margin_maximization);
    maybe_assign(n, "duel_margin_maximization_scale", tc.duel_margin_maximization_scale);
    maybe_assign(n, "use_pbrs",          tc.use_pbrs);
    maybe_assign(n, "pbrs_upper_reward", tc.pbrs_upper_reward);
    maybe_assign(n, "pbrs_clean_reward", tc.pbrs_clean_reward);
    maybe_assign(n, "past_opponent_probability", tc.past_opponent_probability);
    maybe_assign(n, "logs_on_start",     tc.logs_on_start);
    if (n["td_mode"])   tc.td_mode = parse_td_mode(n["td_mode"].as<std::string>());
    // game_variant at the training level takes precedence; load it BEFORE the
    // model block so the model block can still override it if explicitly set.
    if (n["game_variant"]) {
        tc.model.game_variant = parse_game_variant(n["game_variant"].as<std::string>());
    }
    if (n["self_play"]) load_self_play_config(n["self_play"], tc.self_play);
    if (n["model"])     load_model_config(n["model"], tc.model);
    // If the user didn't set input_size explicitly, derive it from the variant.
    // We detect "not explicitly set" by checking whether the loaded value still
    // matches the default-for-1v1 — a heuristic, but harmless: when a user
    // explicitly writes input_size: 986 for a 1v1 run, the result is identical.
    if (tc.model.game_variant == kGameVariant2v2 &&
        tc.model.input_size == Yams1v1::kTensorSize) {
        tc.model.input_size = Yams2v2::kTensorSize;
    }
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
        else if (arg == "--resume" && i + 1 < argc) config.resume_path = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) config.seed = std::stoull(argv[++i]);
        else if (arg == "--num_steps" && i + 1 < argc) config.num_steps = std::stoi(argv[++i]);
        else if (arg == "--learning_rate" && i + 1 < argc) config.training.model.learning_rate = std::stod(argv[++i]);
        else if (arg == "--hidden_layers" && i + 1 < argc) config.training.model.hidden_layers = std::stoi(argv[++i]);
        else if (arg == "--hidden_width" && i + 1 < argc) config.training.model.hidden_width = std::stoi(argv[++i]);
        else if (arg == "--output_activation" && i + 1 < argc) config.training.model.output_activation = argv[++i];
        else if (arg == "--td_mode" && i + 1 < argc) {
            config.training.td_mode = parse_td_mode(argv[++i]);
        }
        else if (arg == "--batch_size" && i + 1 < argc) config.training.train_batch_size = std::stoi(argv[++i]);
        else if (arg == "--train_steps_per_collect" && i + 1 < argc) config.training.train_steps_per_collect = std::stod(argv[++i]);
        else if (arg == "--replay_buffer_size" && i + 1 < argc) config.training.replay_capacity = std::stoi(argv[++i]);
        else if (arg == "--placement_temperature" && i + 1 < argc) config.training.initial_temperature = std::stod(argv[++i]);
        else if (arg == "--temperature_decay_start_step" && i + 1 < argc) config.training.temperature_decay_start_step = std::stoi(argv[++i]);
        else if (arg == "--num_workers" && i + 1 < argc) config.training.self_play.num_workers = std::stoi(argv[++i]);
        else if (arg == "--num_games" && i + 1 < argc) config.training.self_play.num_games = std::stoi(argv[++i]);
        else if (arg == "--num_coordinators" && i + 1 < argc) config.training.self_play.num_coordinators = std::stoi(argv[++i]);
        else if (arg == "--eval_games" && i + 1 < argc) config.training.eval_games = std::stoi(argv[++i]);
        else if (arg == "--eval_interval" && i + 1 < argc) config.training.eval_interval = std::stoi(argv[++i]);
        else if (arg == "--debug_mode" && i + 1 < argc) config.training.debug_mode = (std::stoi(argv[++i]) != 0);
        else if (arg == "--initial_heuristic_weight" && i + 1 < argc) config.training.initial_heuristic_weight = std::stod(argv[++i]);
        else if (arg == "--heuristic_decay_steps" && i + 1 < argc) config.training.heuristic_decay_steps = std::stoi(argv[++i]);
        else if (arg == "--heuristic_version" && i + 1 < argc) config.training.heuristic_version = std::stoi(argv[++i]);
        else if (arg == "--use_duel_margin_maximization" && i + 1 < argc) config.training.use_duel_margin_maximization = (std::stoi(argv[++i]) != 0);
        else if (arg == "--duel_margin_maximization_scale" && i + 1 < argc) config.training.duel_margin_maximization_scale = std::stod(argv[++i]);
        else if (arg == "--past_opponent_probability" && i + 1 < argc) config.training.past_opponent_probability = std::stod(argv[++i]);
        else if (arg == "--game_variant" && i + 1 < argc) {
            config.training.model.game_variant = parse_game_variant(argv[++i]);
            // CLI override: if the user didn't also override input_size, snap
            // it to the variant default. A subsequent --hidden_width / etc.
            // call doesn't disturb this; explicit --input_size if added later
            // would.
            if (config.training.model.game_variant == kGameVariant2v2 &&
                config.training.model.input_size == Yams1v1::kTensorSize) {
                config.training.model.input_size = Yams2v2::kTensorSize;
            } else if (config.training.model.game_variant == kGameVariant1v1 &&
                       config.training.model.input_size == Yams2v2::kTensorSize) {
                config.training.model.input_size = Yams1v1::kTensorSize;
            }
        }
    }
}

AppConfig load_config(int argc, char* argv[]) {
    AppConfig config = default_config();

    // Collect both flags before loading anything, so they can be layered
    // instead of being mutually exclusive. The layering order is:
    //   1. defaults  → 2. --resume saved config  → 3. --config overlay
    //   → 4. CLI overrides (apply_cli_overrides below)
    std::string resume_dir;
    std::string config_file;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--resume" && i + 1 < argc) {
            resume_dir = argv[i + 1];
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[i + 1];
        }
    }

    if (!resume_dir.empty()) {
        config = load_config(resume_dir + "/config.yaml");
        config.resume_path = resume_dir;
    }

    if (!config_file.empty()) {
        // Overlay user-specified YAML on top of the resumed config. We must
        // preserve resume_path through the swap so apply_cli_overrides still
        // sees it (the overlay YAML almost never carries resume_path itself).
        std::string preserved_resume_path = config.resume_path;
        config = load_config(config_file);
        config.resume_path = preserved_resume_path;
    }

    // Final layer: CLI overrides take precedence over YAML / saved config.
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
