#include <iostream>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "model/model_config.h"
#include "model/pro_yams_net.h"
#include "model/trainer.h"
#include "solver/precomputed_tables.h"
#include "ui/server.h"
#include "ui/session_manager.h"

int main(int argc, char* argv[]) {
    std::string checkpoint_path;
    std::string log_dir    = ".";
    std::string static_dir = "./static";
    int         port       = 8080;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--log_dir"    && i + 1 < argc) log_dir         = argv[++i];
        else if (arg == "--static_dir" && i + 1 < argc) static_dir      = argv[++i];
        else if (arg == "--port"       && i + 1 < argc) port            = std::stoi(argv[++i]);
    }

    // Limit PyTorch threading for a UI process (not training).
    torch::set_num_threads(1);

    // Build precomputed tables (takes ~350ms).
    std::cout << "Building solver tables...\n";
    PrecomputedTables tables;
    init_precomputed_tables(tables);

    // Load NN model if checkpoint provided.
    torch::Device device = get_device();
    std::shared_ptr<ProYamsNet> model;

    if (!checkpoint_path.empty()) {
        try {
            ModelConfig model_cfg =
                ModelTrainer::config_from_checkpoint(checkpoint_path);
            ModelTrainer trainer(model_cfg, device);
            trainer.load_weights(checkpoint_path);
            model = trainer.clone_for_inference(device);
            model->to(device);
            model->eval();
            std::cout << "Loaded model from: " << checkpoint_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load checkpoint: " << e.what() << "\n";
            std::cerr << "Continuing without NN bot.\n";
        }
    } else {
        std::cout << "No checkpoint — NN bot unavailable (heuristic-only mode).\n";
    }

    // Create session manager.
    SessionManager sessions(tables, model.get(), device);

    // Start HTTP server.
    UIServer server(port, static_dir, sessions, log_dir);
    std::cout << "Pro Yams UI running at http://localhost:" << port << "\n";
    std::cout << "Frontend:  " << static_dir << "\n";
    std::cout << "Log dir:   " << log_dir    << "\n";
    std::cout << "Press Ctrl+C to stop.\n";
    server.start();  // Blocks until stopped.

    return 0;
}
