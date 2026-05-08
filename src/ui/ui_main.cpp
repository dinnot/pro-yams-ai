#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "eval/tournament.h"
#include "model/model_config.h"
#include "model/pro_yams_net.h"
#include "model/trainer.h"
#include "solver/precomputed_tables.h"
#include "ui/server.h"
#include "ui/session_manager.h"

int main(int argc, char* argv[]) {
    std::string checkpoint_path;
    std::string log_dir         = ".";
    std::string static_dir      = "./static";
    std::string checkpoints_dir;   // empty = derive default below
    int         port            = 8080;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--checkpoint"      && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--log_dir"         && i + 1 < argc) log_dir         = argv[++i];
        else if (arg == "--static_dir"      && i + 1 < argc) static_dir      = argv[++i];
        else if (arg == "--checkpoints_dir" && i + 1 < argc) checkpoints_dir = argv[++i];
        else if (arg == "--port"            && i + 1 < argc) port            = std::stoi(argv[++i]);
    }

    // Default the tournament checkpoints directory to the parent of the
    // loaded checkpoint, so users who pass --checkpoint also get a sensible
    // dropdown without having to set --checkpoints_dir explicitly.
    if (checkpoints_dir.empty()) {
        if (!checkpoint_path.empty()) {
            std::filesystem::path p(checkpoint_path);
            std::filesystem::path parent = p.parent_path();
            // If the checkpoint sits inside a "...checkpoints/<run>/..." tree,
            // prefer the "checkpoints" root so all runs show up.
            std::filesystem::path candidate = parent.parent_path();
            if (!candidate.empty() && candidate.filename() == "checkpoints") {
                checkpoints_dir = candidate.string();
            } else {
                checkpoints_dir = parent.empty() ? "." : parent.string();
            }
        } else {
            checkpoints_dir = "./checkpoints";
        }
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

    // Tournament manager — runs games asynchronously off the HTTP thread.
    TournamentManager tournament(tables, device);

    // Start HTTP server.
    UIServer server(port, static_dir, sessions, log_dir, checkpoints_dir,
                    &tournament);
    std::cout << "Pro Yams UI running at http://localhost:" << port << "\n";
    std::cout << "Frontend:        " << static_dir      << "\n";
    std::cout << "Log dir:         " << log_dir         << "\n";
    std::cout << "Checkpoints dir: " << checkpoints_dir << "\n";
    std::cout << "Press Ctrl+C to stop.\n";
    server.start();  // Blocks until stopped.

    return 0;
}
