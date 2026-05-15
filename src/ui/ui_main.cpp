#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "engine/game_traits.h"
#include "eval/tournament.h"
#include "model/model_config.h"
#include "model/pro_yams_net.h"
#include "model/trainer.h"
#include "solver/precomputed_tables.h"
#include "ui/server.h"
#include "ui/session_manager.h"

namespace {

template <typename Traits>
int run_server_variant(int port, const std::string& static_dir,
                       const std::string& log_dir,
                       const std::string& checkpoints_dir,
                       const PrecomputedTables& tables,
                       ProYamsNet* model, torch::Device device) {
    SessionManagerT<Traits> sessions(tables, model, device);

    // Tournament manager is 1v1-only for now (eval/tournament not templated).
    // In 2v2 mode we pass a null tournament manager — the UI server tolerates it.
    if constexpr (std::is_same_v<Traits, Yams1v1>) {
        TournamentManager tournament(tables, device);
        UIServerT<Traits> server(port, static_dir, sessions, log_dir,
                                  checkpoints_dir, &tournament);
        std::cout << "Pro Yams UI running at http://localhost:" << port << "\n";
        std::cout << "Frontend:        " << static_dir      << "\n";
        std::cout << "Log dir:         " << log_dir         << "\n";
        std::cout << "Checkpoints dir: " << checkpoints_dir << "\n";
        std::cout << "Press Ctrl+C to stop.\n";
        server.start();
    } else {
        UIServerT<Traits> server(port, static_dir, sessions, log_dir,
                                  checkpoints_dir, /*tournament=*/nullptr);
        std::cout << "Pro Yams UI (2v2) running at http://localhost:" << port << "\n";
        std::cout << "Frontend:        " << static_dir      << "\n";
        std::cout << "Tournament endpoints disabled in 2v2 mode.\n";
        std::cout << "Press Ctrl+C to stop.\n";
        server.start();
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string checkpoint_path;
    std::string log_dir         = ".";
    std::string static_dir      = "./static";
    std::string checkpoints_dir;
    int         port            = 8080;
    std::string variant         = "1v1";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--checkpoint"      && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--log_dir"         && i + 1 < argc) log_dir         = argv[++i];
        else if (arg == "--static_dir"      && i + 1 < argc) static_dir      = argv[++i];
        else if (arg == "--checkpoints_dir" && i + 1 < argc) checkpoints_dir = argv[++i];
        else if (arg == "--port"            && i + 1 < argc) port            = std::stoi(argv[++i]);
        else if (arg == "--variant"         && i + 1 < argc) variant         = argv[++i];
        else if (arg == "--game_variant"    && i + 1 < argc) variant         = argv[++i];
    }

    if (checkpoints_dir.empty()) {
        if (!checkpoint_path.empty()) {
            std::filesystem::path p(checkpoint_path);
            std::filesystem::path parent = p.parent_path();
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

    torch::set_num_threads(1);

    std::cout << "Building solver tables...\n";
    PrecomputedTables tables;
    init_precomputed_tables(tables);

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

    if (variant == "2v2") {
        return run_server_variant<Yams2v2>(port, static_dir, log_dir,
                                            checkpoints_dir, tables,
                                            model.get(), device);
    }
    return run_server_variant<Yams1v1>(port, static_dir, log_dir,
                                        checkpoints_dir, tables,
                                        model.get(), device);
}
