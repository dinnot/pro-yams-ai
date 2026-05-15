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

// ---------------------------------------------------------------------------
// pro_yams_play — public-facing "play against the NN" web app.
//
// Reuses UIServer + SessionManager from the main UI, but ships a separate
// mobile-first static directory and runs on its own port so it can sit next
// to the dev UI without conflict.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string checkpoint_path;
    std::string static_dir = "./play_static";
    int         port       = 8090;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--static_dir" && i + 1 < argc) static_dir      = argv[++i];
        else if (arg == "--port"       && i + 1 < argc) port            = std::stoi(argv[++i]);
    }

    torch::set_num_threads(1);

    std::cout << "Building solver tables...\n";
    PrecomputedTables tables;
    init_precomputed_tables(tables);

    torch::Device device = get_device();
    std::shared_ptr<ProYamsNet> model;

    if (checkpoint_path.empty()) {
        std::cerr << "Error: --checkpoint is required (NN bot must be available).\n";
        return 1;
    }
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
        std::cerr << "Error: failed to load checkpoint: " << e.what() << "\n";
        return 1;
    }

    SessionManager sessions(tables, model.get(), device);

    // No tournament/log dirs needed for the play app — pass empty strings and
    // a null tournament manager. UIServer tolerates a nullptr tournament_.
    UIServer server(port, static_dir, sessions, /*log_dir=*/".",
                    /*checkpoints_dir=*/".", /*tournament=*/nullptr);
    std::cout << "Pro Yams Play running at http://localhost:" << port << "\n";
    std::cout << "Frontend:   " << static_dir << "\n";
    std::cout << "Press Ctrl+C to stop.\n";
    server.start();

    return 0;
}
