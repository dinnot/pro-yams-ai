#include <csignal>
#include <iostream>
#include <string>
#include <filesystem>

#include <torch/torch.h>

#include "config/app_config.h"
#include "config/config_loader.h"
#include "config/config_printer.h"
#include "config/config_validator.h"
#include "eval/evaluator.h"
#include "model/trainer.h"
#include "solver/precomputed_tables.h"
#include "training/resume.h"
#include "training/training_loop.h"

// ---------------------------------------------------------------------------
// Signal handling — graceful shutdown for training.
// ---------------------------------------------------------------------------
static std::atomic<bool> g_shutdown{false};
static TrainingLoop*     g_training_loop = nullptr;

extern "C" void signal_handler(int /*signal*/) {
    g_shutdown.store(true, std::memory_order_relaxed);
    if (g_training_loop) {
        g_training_loop->stop();
    }
}

// ---------------------------------------------------------------------------
// load_and_validate — validate a loaded config.
// Returns false on error (message already printed).
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
// mode_info — print build info and CUDA status.
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
// mode_train — run the training loop.
// ---------------------------------------------------------------------------
static int mode_train(const AppConfig& cfg) {
    torch::Device train_device  = torch::cuda::is_available()
                                  ? torch::Device(torch::kCUDA)
                                  : torch::Device(torch::kCPU);
    torch::Device infer_device  = train_device;

    std::cout << "Building solver tables...\n";
    PrecomputedTables tables;
    init_precomputed_tables(tables);

    TrainingLoop loop(cfg.training, tables, train_device, infer_device);

    if (!cfg.training.log_dir.empty()) {
        std::filesystem::create_directories(cfg.training.log_dir);
        save_config(cfg, cfg.training.log_dir + "/config.yaml");
    }

    if (!cfg.checkpoint_path.empty()) {
        std::cout << "Resuming from checkpoint dir: " << cfg.checkpoint_path << "\n";
        resume_from_checkpoint(loop, cfg.checkpoint_path);
    }

    std::cout << "Starting training for " << cfg.num_steps << " steps\n";
    g_training_loop = &loop;
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);
    loop.run(cfg.num_steps);
    g_training_loop = nullptr;

    TrainingMetrics m = loop.metrics();
    std::cout << "Training complete.  Steps: " << m.training_step
              << "  Loss: "      << m.loss
              << "  Win rate: "  << m.latest_eval_win_rate << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// mode_eval — evaluate a checkpoint against the heuristic bot.
// ---------------------------------------------------------------------------
static int mode_eval(const AppConfig& cfg) {
    if (cfg.checkpoint_path.empty()) {
        std::cerr << "Eval mode requires --checkpoint\n";
        return 1;
    }

    torch::Device device = torch::cuda::is_available()
                           ? torch::Device(torch::kCUDA)
                           : torch::Device(torch::kCPU);

    PrecomputedTables tables;
    init_precomputed_tables(tables);

    ModelTrainer trainer(cfg.training.model, device);
    int step = 0;
    double temp = 1.0, eps = 0.0;
    trainer.load_checkpoint(cfg.checkpoint_path, step, temp, eps);

    EvalResult result = run_evaluation(trainer.model_mut(), device, tables,
                                        cfg.training.eval_games, cfg.seed);

    std::cout << "Evaluation results (" << result.total_games << " games):\n"
              << "  NN win rate:      " << result.nn_win_rate() << "\n"
              << "  NN win rate as P0: " << result.nn_win_rate_as_p0() << "\n"
              << "  NN win rate as P1: " << result.nn_win_rate_as_p1() << "\n"
              << "  Avg duel margin (all): " << result.avg_duel_margin << "\n";
    return 0;
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
    std::cerr << "Available: info, config, train, eval, play, benchmark, generate\n";
    return 1;
}
