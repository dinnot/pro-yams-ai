#include <atomic>
#include <csignal>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>

#include <torch/torch.h>

#include "config/app_config.h"
#include "config/config_loader.h"
#include "config/config_printer.h"
#include "config/config_validator.h"
#include "distil/distil_loop.h"
#include "distil/distil_resume.h"
#include "engine/game_traits.h"
#include "eval/evaluator.h"
#include "model/model_config.h"
#include "model/trainer.h"
#include "solver/precomputed_tables.h"
#include "training/resume.h"
#include "training/training_loop.h"

// ---------------------------------------------------------------------------
// Signal handling — graceful shutdown for training. The active loop is
// variant-dependent, so we hold a type-erased "stop" callback.
// ---------------------------------------------------------------------------
static std::atomic<bool>     g_shutdown{false};
static std::function<void()> g_training_stop;

extern "C" void signal_handler(int /*signal*/) {
    g_shutdown.store(true, std::memory_order_relaxed);
    if (g_training_stop) g_training_stop();
}

// ---------------------------------------------------------------------------
// load_and_validate
// ---------------------------------------------------------------------------
static bool load_and_validate(AppConfig& cfg) {
    ValidationResult vr = validate_config(cfg);
    if (!vr.ok) {
        std::cerr << "Configuration errors:\n";
        for (const auto& err : vr.errors)
            std::cerr << "  - " << err << "\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// mode_info
// ---------------------------------------------------------------------------
static int mode_info() {
    std::cout << "Pro Yams AI\n";
    std::cout << "CUDA available: "
              << (torch::cuda::is_available() ? "yes" : "no") << "\n";
    std::cout << "CUDA devices: " << torch::cuda::device_count() << "\n";
    if (torch::cuda::is_available()) {
        auto t = torch::randn({3, 3}, torch::kCUDA);
        std::cout << "GPU tensor test: OK (" << t.sizes() << ")\n";
    }
    return 0;
}

// ---------------------------------------------------------------------------
// run_training_variant<Traits>
// ---------------------------------------------------------------------------
template <typename Traits>
static int run_training_variant(const AppConfig& cfg,
                                torch::Device train_device,
                                torch::Device infer_device,
                                const PrecomputedTables& tables) {
    TrainingLoopT<Traits> loop(cfg.training, tables, train_device, infer_device);

    if (!cfg.resume_path.empty() && !cfg.checkpoint_path.empty()) {
        std::cerr << "Error: --resume and --checkpoint cannot be used together.\n";
        return 1;
    }

    if (!cfg.resume_path.empty()) {
        std::cout << "Resuming training from: " << cfg.resume_path << "\n";
        if (!resume_from_checkpoint<Traits>(loop, cfg.resume_path)) {
            std::cerr << "Error: no checkpoints found in " << cfg.resume_path << "\n";
            return 1;
        }
    } else if (!cfg.checkpoint_path.empty()) {
        std::cout << "Initializing model weights from: " << cfg.checkpoint_path << "\n";
        if (!init_from_checkpoint<Traits>(loop, cfg.checkpoint_path)) {
            std::cerr << "Error: could not load checkpoint from " << cfg.checkpoint_path << "\n";
            return 1;
        }
    }

    std::cout << "Starting training for " << cfg.num_steps << " steps\n";
    g_training_stop = [&loop] { loop.stop(); };
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);
    loop.run(cfg.num_steps);
    g_training_stop = nullptr;

    TrainingMetrics m = loop.metrics();
    std::cout << "Training complete.  Steps: " << m.training_step
              << "  Loss: "      << m.loss
              << "  Win rate: "  << m.latest_eval_win_rate << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// mode_train
// ---------------------------------------------------------------------------
static int mode_train(const AppConfig& cfg) {
    torch::Device train_device  = torch::cuda::is_available()
                                  ? torch::Device(torch::kCUDA)
                                  : torch::Device(torch::kCPU);
    torch::Device infer_device  = train_device;

    std::cout << "Building solver tables...\n";
    PrecomputedTables tables;
    init_precomputed_tables(tables);

    if (!cfg.training.log_dir.empty()) {
        std::filesystem::create_directories(cfg.training.log_dir);
        save_config(cfg, cfg.training.log_dir + "/config.yaml");
    }
    std::filesystem::create_directories(cfg.training.checkpoint_dir);
    save_config(cfg, cfg.training.checkpoint_dir + "/config.yaml");

    // Dispatch on game variant: a 1v1 ModelConfig drives Yams1v1, a 2v2
    // ModelConfig drives Yams2v2. The TrainingLoopT constructor asserts
    // input_size / variant agreement.
    if (cfg.training.model.game_variant == kGameVariant2v2) {
        std::cout << "Game variant: 2v2 (Yams2v2)\n";
        return run_training_variant<Yams2v2>(cfg, train_device, infer_device, tables);
    } else {
        std::cout << "Game variant: 1v1 (Yams1v1)\n";
        return run_training_variant<Yams1v1>(cfg, train_device, infer_device, tables);
    }
}

// ---------------------------------------------------------------------------
// run_eval_variant<Traits>
// ---------------------------------------------------------------------------
template <typename Traits>
static int run_eval_variant(const AppConfig& cfg, torch::Device device,
                            const PrecomputedTables& tables) {
    ModelTrainer trainer(cfg.training.model, device);
    int step = 0;
    double temp = 1.0, eps = 0.0;
    trainer.load_checkpoint(cfg.checkpoint_path, step, temp, eps);

    EvalResult result = run_evaluation<Traits>(trainer.model_mut(), device, tables,
                                                cfg.training.eval_games, cfg.seed);

    constexpr bool is_2v2 = std::is_same_v<Traits, Yams2v2>;
    const char* p0_label = is_2v2 ? "NN as Team 0" : "NN as P0";
    const char* p1_label = is_2v2 ? "NN as Team 1" : "NN as P1";

    std::cout << "Evaluation results (" << result.total_games << " games"
              << (is_2v2 ? ", 2v2" : ", 1v1") << "):\n"
              << "  NN win rate:        " << result.nn_win_rate() << "\n"
              << "  " << p0_label << " win rate: " << result.nn_win_rate_as_p0() << "\n"
              << "  " << p1_label << " win rate: " << result.nn_win_rate_as_p1() << "\n"
              << "  Avg duel margin (all): " << result.avg_duel_margin << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// mode_eval — dispatch on the checkpoint's own variant tag so the user
// doesn't need to keep --game_variant in sync with --checkpoint.
// ---------------------------------------------------------------------------
static int mode_eval(AppConfig cfg) {
    if (cfg.checkpoint_path.empty()) {
        std::cerr << "Eval mode requires --checkpoint\n";
        return 1;
    }

    torch::Device device = torch::cuda::is_available()
                           ? torch::Device(torch::kCUDA)
                           : torch::Device(torch::kCPU);

    PrecomputedTables tables;
    init_precomputed_tables(tables);

    // Pull the variant + input_size out of the checkpoint and overwrite the
    // cfg's model section so ModelTrainer is constructed with matching shape.
    try {
        ModelConfig ckpt_cfg =
            ModelTrainer::config_from_checkpoint(cfg.checkpoint_path);
        cfg.training.model.game_variant = ckpt_cfg.game_variant;
        cfg.training.model.input_size   = ckpt_cfg.input_size;
        cfg.training.model.hidden_layers   = ckpt_cfg.hidden_layers;
        cfg.training.model.hidden_width    = ckpt_cfg.hidden_width;
        cfg.training.model.output_activation = ckpt_cfg.output_activation;
        cfg.training.model.architecture    = ckpt_cfg.architecture;
    } catch (const std::exception& e) {
        std::cerr << "Eval: failed to read checkpoint metadata: " << e.what() << "\n";
        return 1;
    }

    if (cfg.training.model.game_variant == kGameVariant2v2) {
        std::cout << "Eval variant: 2v2 (Yams2v2)\n";
        return run_eval_variant<Yams2v2>(cfg, device, tables);
    }
    std::cout << "Eval variant: 1v1 (Yams1v1)\n";
    return run_eval_variant<Yams1v1>(cfg, device, tables);
}

// ---------------------------------------------------------------------------
// run_distil_variant<Traits>
// ---------------------------------------------------------------------------
template <typename Traits>
static int run_distil_variant(const AppConfig& cfg,
                              torch::Device train_device,
                              torch::Device infer_device,
                              const PrecomputedTables& tables) {
    DistilLoopT<Traits> loop(cfg.distil, tables, train_device, infer_device);

    if (!cfg.resume_path.empty() && !cfg.checkpoint_path.empty()) {
        std::cerr << "Error: --resume and --checkpoint cannot be used together.\n";
        return 1;
    }

    if (!cfg.resume_path.empty()) {
        std::cout << "Resuming distillation from: " << cfg.resume_path << "\n";
        try {
            if (!resume_distil_from_checkpoint<Traits>(loop, cfg.resume_path)) {
                std::cerr << "Error: no checkpoints found in " << cfg.resume_path << "\n";
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to resume distillation: " << e.what() << "\n";
            return 1;
        }
    } else if (!cfg.checkpoint_path.empty()) {
        std::cout << "Initialising student weights from: "
                  << cfg.checkpoint_path << "\n";
        try {
            loop.trainer().load_weights(cfg.checkpoint_path);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load student checkpoint: "
                      << e.what() << "\n";
            return 1;
        }
    }

    std::cout << "Starting distillation (max_steps="
              << cfg.distil.max_steps << ")\n";
    g_training_stop = [&loop] { loop.stop(); };
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);
    loop.run();
    g_training_stop = nullptr;

    // Throughput / volume summary (the win-rate headline block is printed
    // by DistilLoopT::final_report inside run()).
    TrainingMetrics m = loop.metrics();
    const long turns = loop.total_turns_processed();
    const double per_game = (m.games_played > 0)
        ? static_cast<double>(m.total_samples_emitted) / m.games_played
        : 0.0;
    std::cout << "\nThroughput: " << m.training_step << " steps  "
              << "loss=" << m.loss << "  "
              << "games=" << m.games_played << "  "
              << "placements=" << turns << "  "
              << "trained=" << m.total_samples_trained << "  "
              << "emitted=" << m.total_samples_emitted
              << " (~" << per_game << "/game)\n";
    return 0;
}

// ---------------------------------------------------------------------------
// mode_distil
// ---------------------------------------------------------------------------
static int mode_distil(const AppConfig& cfg) {
    torch::Device train_device = torch::cuda::is_available()
                                 ? torch::Device(torch::kCUDA)
                                 : torch::Device(torch::kCPU);
    torch::Device infer_device = train_device;

    std::cout << "Building solver tables...\n";
    PrecomputedTables tables;
    init_precomputed_tables(tables);

    if (!cfg.distil.log_dir.empty()) {
        std::filesystem::create_directories(cfg.distil.log_dir);
        save_config(cfg, cfg.distil.log_dir + "/config.yaml");
    }
    std::filesystem::create_directories(cfg.distil.checkpoint_dir);
    save_config(cfg, cfg.distil.checkpoint_dir + "/config.yaml");

    if (cfg.distil.student_model.game_variant == kGameVariant2v2) {
        std::cout << "Distil variant: 2v2 (Yams2v2)\n";
        return run_distil_variant<Yams2v2>(cfg, train_device, infer_device, tables);
    }
    std::cout << "Distil variant: 1v1 (Yams1v1)\n";
    return run_distil_variant<Yams1v1>(cfg, train_device, infer_device, tables);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    torch::set_num_threads(1);
    at::set_num_interop_threads(1);

    AppConfig config;
    try {
        config = load_config(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (config.mode == "info") {
        return mode_info();
    }

    if (config.mode == "config") {
        if (!load_and_validate(config)) return 1;
        print_config(config);
        return 0;
    }

    if (config.mode == "train") {
        if (!load_and_validate(config)) return 1;
        return mode_train(config);
    }

    if (config.mode == "eval") {
        if (!load_and_validate(config)) return 1;
        return mode_eval(config);
    }

    if (config.mode == "distil") {
        if (!load_and_validate(config)) return 1;
        return mode_distil(config);
    }

    if (config.mode == "play") {
        std::cout << "Play mode (not yet implemented)\n";
        return 0;
    }

    if (config.mode == "benchmark") {
        std::cout << "Benchmark mode (not yet implemented)\n";
        return 0;
    }

    if (config.mode == "generate") {
        std::cout << "Generate heuristic data mode (not yet implemented)\n";
        return 0;
    }

    std::cerr << "Unknown mode: " << config.mode << "\n";
    std::cerr << "Available: info, config, train, eval, distil, play, benchmark, generate\n";
    return 1;
}
