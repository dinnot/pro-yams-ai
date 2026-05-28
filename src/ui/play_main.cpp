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
#include "ui/game_recorder.h"
#include "ui/server.h"
#include "ui/session_manager.h"

// ---------------------------------------------------------------------------
// pro_yams_play — public-facing "play against the NN" web app.
//
// Reuses UIServerT + SessionManagerT from the main UI, but ships a separate
// mobile-first static directory and runs on its own port. The --variant flag
// selects between 1v1 and 2v2 (default: 1v1).
// ---------------------------------------------------------------------------

namespace {

template <typename Traits>
int run_play_variant(int port, const std::string& static_dir,
                     const std::string& games_dir,
                     const std::string& checkpoint_label,
                     const PrecomputedTables& tables,
                     ProYamsNet* model, torch::Device device) {
    SessionManagerT<Traits> sessions(tables, model, device);
    GameRecorder recorder(games_dir, checkpoint_label, port);
    UIServerT<Traits> server(port, static_dir, sessions, /*log_dir=*/".",
                              /*checkpoints_dir=*/".", /*tournament=*/nullptr,
                              &recorder, games_dir);
    std::cout << "Pro Yams Play running at http://localhost:" << port << "\n";
    std::cout << "Frontend:   " << static_dir << "\n";
    std::cout << "Variant:    " << ((Traits::kNumPlayers == 4) ? "2v2" : "1v1") << "\n";
    std::cout << "Games dir:  " << games_dir << "\n";
    std::cout << "Press Ctrl+C to stop.\n";
    server.start();
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string checkpoint_path;
    std::string static_dir = "./play_static";
    std::string games_dir  = "./recorded_games";
    int         port       = 8090;
    std::string variant    = "1v1";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--static_dir" && i + 1 < argc) static_dir      = argv[++i];
        else if (arg == "--games_dir"  && i + 1 < argc) games_dir       = argv[++i];
        else if (arg == "--port"       && i + 1 < argc) port            = std::stoi(argv[++i]);
        else if (arg == "--variant"    && i + 1 < argc) variant         = argv[++i];
        else if (arg == "--game_variant" && i + 1 < argc) variant       = argv[++i];
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

    if (variant == "2v2") {
        return run_play_variant<Yams2v2>(port, static_dir, games_dir, checkpoint_path,
                                         tables, model.get(), device);
    }
    return run_play_variant<Yams1v1>(port, static_dir, games_dir, checkpoint_path,
                                     tables, model.get(), device);
}
