#include "config/config_loader.h"

#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "distil/distil_config.h"
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

static TeacherKind parse_teacher_kind(const std::string& s) {
    if (s == "heuristic") return TeacherKind::kHeuristic;
    if (s == "nn")        return TeacherKind::kNN;
    throw std::runtime_error("Unknown teacher_kind value: '" + s +
                             "' (expected: heuristic, nn)");
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
    maybe_assign(n, "lr_backoff_enabled",  tc.lr_backoff_enabled);
    maybe_assign(n, "lr_backoff_factor",   tc.lr_backoff_factor);
    maybe_assign(n, "lr_backoff_min_lr",   tc.lr_backoff_min_lr);
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

// ---------------------------------------------------------------------------
// Distil YAML loader. Mirrors load_training_config for the shared fields and
// adds teacher selection, replay-buffer and convergence knobs.
// ---------------------------------------------------------------------------
void load_distil_config(const YAML::Node& n, DistilConfig& dc) {
    if (n["teacher_kind"])
        dc.teacher_kind = parse_teacher_kind(n["teacher_kind"].as<std::string>());
    maybe_assign(n, "teacher_heuristic_version", dc.teacher_heuristic_version);
    maybe_assign(n, "teacher_checkpoint",        dc.teacher_checkpoint_path);

    maybe_assign(n, "train_batch_size",        dc.train_batch_size);
    maybe_assign(n, "replay_buffer_capacity",  dc.replay_buffer_capacity);
    maybe_assign(n, "min_samples_to_start",    dc.min_samples_to_start);
    maybe_assign(n, "samples_per_train",       dc.samples_per_train);
    maybe_assign(n, "samples_per_games_rate",  dc.samples_per_games_rate);

    maybe_assign(n, "checkpoint_interval",     dc.checkpoint_interval);
    maybe_assign(n, "max_checkpoints",         dc.max_checkpoints);

    maybe_assign(n, "max_steps",                  dc.max_steps);
    maybe_assign(n, "min_steps",                  dc.min_steps);
    maybe_assign(n, "eval_interval",              dc.eval_interval);
    maybe_assign(n, "eval_games",                 dc.eval_games);
    maybe_assign(n, "reference_heuristic_version", dc.reference_heuristic_version);
    maybe_assign(n, "convergence_win_rate_delta", dc.convergence_win_rate_delta);
    maybe_assign(n, "convergence_patience",       dc.convergence_patience);
    maybe_assign(n, "convergence_match_mse",      dc.convergence_match_mse);
    maybe_assign(n, "final_eval_games",           dc.final_eval_games);

    maybe_assign(n, "use_duel_margin_maximization",   dc.use_duel_margin_maximization);
    maybe_assign(n, "duel_margin_maximization_scale", dc.duel_margin_maximization_scale);

    maybe_assign(n, "checkpoint_dir", dc.checkpoint_dir);
    maybe_assign(n, "log_dir",        dc.log_dir);
    maybe_assign(n, "log_path",       dc.log_path);
    maybe_assign(n, "debug_mode",     dc.debug_mode);

    // game_variant at the distil level takes precedence; load it BEFORE the
    // student block so the student block can still override it if explicitly set.
    if (n["game_variant"]) {
        dc.student_model.game_variant =
            parse_game_variant(n["game_variant"].as<std::string>());
    }
    if (n["self_play"]) load_self_play_config(n["self_play"], dc.self_play);
    if (n["student"])   load_model_config(n["student"], dc.student_model);

    // Default-promote input_size for 2v2 if the user only set game_variant.
    if (dc.student_model.game_variant == kGameVariant2v2 &&
        dc.student_model.input_size == Yams1v1::kTensorSize) {
        dc.student_model.input_size = Yams2v2::kTensorSize;
    }
}

}  // namespace

AppConfig default_config() {
    return AppConfig{};
}

void apply_cli_overrides(AppConfig& config, int argc, char* argv[]) {
    // Pre-scan for --mode so shared-flag routing (training vs distil) is
    // mode-aware regardless of CLI argument order.
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--mode") {
            config.mode = argv[i + 1];
            break;
        }
    }
    const bool to_distil = (config.mode == "distil");

    // Routing for shared knobs (num_workers, batch_size, learning_rate, etc.).
    SelfPlayConfig& sp_target = to_distil ? config.distil.self_play
                                          : config.training.self_play;
    ModelConfig&    m_target  = to_distil ? config.distil.student_model
                                          : config.training.model;
    int&            batch_target  = to_distil ? config.distil.train_batch_size
                                              : config.training.train_batch_size;
    int&            eval_int_t    = to_distil ? config.distil.eval_interval
                                              : config.training.eval_interval;
    int&            eval_games_t  = to_distil ? config.distil.eval_games
                                              : config.training.eval_games;
    bool&           debug_target  = to_distil ? config.distil.debug_mode
                                              : config.training.debug_mode;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) config.mode = argv[++i];
        else if (arg == "--config" && i + 1 < argc) { config.config_path = argv[++i]; }
        else if (arg == "--checkpoint" && i + 1 < argc) config.checkpoint_path = argv[++i];
        else if (arg == "--resume" && i + 1 < argc) config.resume_path = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) config.seed = std::stoull(argv[++i]);
        else if (arg == "--num_steps" && i + 1 < argc) {
            int v = std::stoi(argv[++i]);
            config.num_steps = v;
            if (to_distil) config.distil.max_steps = v;
        }
        else if (arg == "--learning_rate" && i + 1 < argc) m_target.learning_rate = std::stod(argv[++i]);
        else if (arg == "--hidden_layers" && i + 1 < argc) m_target.hidden_layers = std::stoi(argv[++i]);
        else if (arg == "--hidden_width"  && i + 1 < argc) m_target.hidden_width  = std::stoi(argv[++i]);
        else if (arg == "--output_activation" && i + 1 < argc) m_target.output_activation = argv[++i];
        else if (arg == "--td_mode" && i + 1 < argc) {
            // Training-only knob; silently routes to training.
            config.training.td_mode = parse_td_mode(argv[++i]);
        }
        else if (arg == "--batch_size" && i + 1 < argc) batch_target = std::stoi(argv[++i]);
        else if (arg == "--train_steps_per_collect" && i + 1 < argc) config.training.train_steps_per_collect = std::stod(argv[++i]);
        else if (arg == "--replay_buffer_size" && i + 1 < argc) config.training.replay_capacity = std::stoi(argv[++i]);
        else if (arg == "--placement_temperature" && i + 1 < argc) config.training.initial_temperature = std::stod(argv[++i]);
        else if (arg == "--temperature_decay_start_step" && i + 1 < argc) config.training.temperature_decay_start_step = std::stoi(argv[++i]);
        else if (arg == "--num_workers" && i + 1 < argc) sp_target.num_workers = std::stoi(argv[++i]);
        else if (arg == "--num_games"   && i + 1 < argc) sp_target.num_games   = std::stoi(argv[++i]);
        else if (arg == "--num_coordinators" && i + 1 < argc) sp_target.num_coordinators = std::stoi(argv[++i]);
        else if (arg == "--eval_games"  && i + 1 < argc) eval_games_t = std::stoi(argv[++i]);
        else if (arg == "--eval_interval" && i + 1 < argc) eval_int_t  = std::stoi(argv[++i]);
        else if (arg == "--debug_mode"  && i + 1 < argc) debug_target = (std::stoi(argv[++i]) != 0);
        else if (arg == "--initial_heuristic_weight" && i + 1 < argc) config.training.initial_heuristic_weight = std::stod(argv[++i]);
        else if (arg == "--heuristic_decay_steps" && i + 1 < argc) config.training.heuristic_decay_steps = std::stoi(argv[++i]);
        else if (arg == "--heuristic_version" && i + 1 < argc) config.training.heuristic_version = std::stoi(argv[++i]);
        else if (arg == "--use_duel_margin_maximization" && i + 1 < argc) {
            bool v = (std::stoi(argv[++i]) != 0);
            if (to_distil) config.distil.use_duel_margin_maximization   = v;
            else           config.training.use_duel_margin_maximization = v;
        }
        else if (arg == "--duel_margin_maximization_scale" && i + 1 < argc) {
            double v = std::stod(argv[++i]);
            if (to_distil) config.distil.duel_margin_maximization_scale   = v;
            else           config.training.duel_margin_maximization_scale = v;
        }
        else if (arg == "--past_opponent_probability" && i + 1 < argc) config.training.past_opponent_probability = std::stod(argv[++i]);
        else if (arg == "--game_variant" && i + 1 < argc) {
            int gv = parse_game_variant(argv[++i]);
            m_target.game_variant = gv;
            // Snap input_size to variant default unless the user pinned a custom one.
            if (gv == kGameVariant2v2 && m_target.input_size == Yams1v1::kTensorSize) {
                m_target.input_size = Yams2v2::kTensorSize;
            } else if (gv == kGameVariant1v1 && m_target.input_size == Yams2v2::kTensorSize) {
                m_target.input_size = Yams1v1::kTensorSize;
            }
        }
        // --- Distil-only flags ---
        else if (arg == "--teacher_kind" && i + 1 < argc) {
            config.distil.teacher_kind = parse_teacher_kind(argv[++i]);
        }
        else if (arg == "--teacher_heuristic_version" && i + 1 < argc) {
            config.distil.teacher_heuristic_version = std::stoi(argv[++i]);
        }
        else if (arg == "--teacher_checkpoint" && i + 1 < argc) {
            config.distil.teacher_checkpoint_path = argv[++i];
        }
        else if (arg == "--max_steps" && i + 1 < argc) {
            config.distil.max_steps = std::stoi(argv[++i]);
        }
        else if (arg == "--min_steps" && i + 1 < argc) {
            config.distil.min_steps = std::stoi(argv[++i]);
        }
        else if (arg == "--reference_heuristic_version" && i + 1 < argc) {
            config.distil.reference_heuristic_version = std::stoi(argv[++i]);
        }
        else if (arg == "--convergence_win_rate_delta" && i + 1 < argc) {
            config.distil.convergence_win_rate_delta = std::stod(argv[++i]);
        }
        else if (arg == "--convergence_patience" && i + 1 < argc) {
            config.distil.convergence_patience = std::stoi(argv[++i]);
        }
        else if (arg == "--final_eval_games" && i + 1 < argc) {
            config.distil.final_eval_games = std::stoi(argv[++i]);
        }
        else if (arg == "--samples_per_games_rate" && i + 1 < argc) {
            config.distil.samples_per_games_rate = std::stod(argv[++i]);
        }
        else if (arg == "--replay_buffer_capacity" && i + 1 < argc) {
            config.distil.replay_buffer_capacity = std::stoi(argv[++i]);
        }
        else if (arg == "--min_samples_to_start" && i + 1 < argc) {
            config.distil.min_samples_to_start = std::stoi(argv[++i]);
        }
        else if (arg == "--samples_per_train" && i + 1 < argc) {
            config.distil.samples_per_train = std::stod(argv[++i]);
        }
        else if (arg == "--convergence_match_mse" && i + 1 < argc) {
            config.distil.convergence_match_mse = std::stod(argv[++i]);
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
    if (root["mode"]) cfg.mode = root["mode"].as<std::string>();
    if (root["training"]) load_training_config(root["training"], cfg.training);
    if (root["distil"])   load_distil_config(root["distil"],     cfg.distil);
    return cfg;
}
