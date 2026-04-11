#include "training/training_loop.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include "engine/tensor.h"               // kTensorSize
#include "eval/evaluator.h"
#include "eval/eval_logging.h"
#include "self_play/training_data.h"     // extract_training_samples
#include "training/logging.h"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TrainingLoop::TrainingLoop(const TrainingConfig& config,
                            const PrecomputedTables& tables,
                            torch::Device training_device,
                            torch::Device inference_device)
    : config_(config),
      tables_(tables),
      train_device_(training_device),
      infer_device_(inference_device),
      temperature_(config.initial_temperature),
      epsilon_(config.initial_epsilon),
      heuristic_weight_(config.initial_heuristic_weight),
      sample_rng_(0xDEADBEEF12345678ULL) {
    // Replay buffer
    buffer_ = std::make_unique<ReplayBuffer>(config_.replay_capacity);

    // Trainer (creates fresh model on training device)
    ModelConfig mc = config_.model;
    mc.debug_mode = config_.debug_mode;
    trainer_ = std::make_unique<ModelTrainer>(mc, train_device_);

    // Clone model for inference engine
    auto inf_model = trainer_->clone_for_inference(infer_device_);
    inf_model->eval();
    inference_ = std::make_shared<InferenceEngine>(inf_model, infer_device_);

    // Solver config: enable exploration if temperature > 0
    solver_config_.placement_temperature = temperature_;
    solver_config_.hold_temperature      = temperature_;
    solver_config_.exploration_enabled   = (temperature_ > 0.0);
    solver_config_.debug_mode            = config_.debug_mode;
    solver_config_.heuristic_weight      = heuristic_weight_;
    solver_config_.debug_log_path        = config_.log_dir + "/debug_game_0.log";

    // Orchestrator (does NOT start threads yet — call run() for that)
    orchestrator_ = std::make_unique<SelfPlayOrchestrator>(
        config_.self_play, tables_, *inference_, solver_config_);
}

// ---------------------------------------------------------------------------
// stop (signal)
// ---------------------------------------------------------------------------

void TrainingLoop::stop() {
    stop_flag_.store(true);
}

// ---------------------------------------------------------------------------
// metrics snapshot
// ---------------------------------------------------------------------------

TrainingMetrics TrainingLoop::metrics() const {
    TrainingMetrics m;
    m.training_step     = training_step_;
    m.games_played      = games_played_;
    m.samples_in_buffer = buffer_->size();
    m.loss              = last_loss_;
    m.temperature            = temperature_;
    m.epsilon                = epsilon_;
    m.latest_eval_win_rate   = last_eval_win_rate_;
    return m;
}

// ---------------------------------------------------------------------------
// run — main training loop
// ---------------------------------------------------------------------------

void TrainingLoop::run(int num_steps) {
    // Ensure checkpoint dir exists
    std::filesystem::create_directories(config_.checkpoint_dir);

    orchestrator_->start();

    while (!stop_flag_.load(std::memory_order_relaxed) &&
           training_step_ < num_steps) {
        int collected = collect_completed_games();

        if (buffer_->size() >= config_.min_buffer_size) {
            // Throttled training: 1 gradient step per completed game
            // Guarantees a perfectly stable Replay Ratio of ~13 (2048 / 156)
            for (int i = 0; i < collected; ++i) {
                do_training_step();
                maybe_swap_model();
                maybe_checkpoint();
                maybe_evaluate();
                if (training_step_ >= num_steps || stop_flag_.load(std::memory_order_relaxed)) {
                    break;

                }
            }
        }
        
        // If we haven't reached min_buffer_size, or no games were ready this tick
        if (collected == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    orchestrator_->stop();
}

// ---------------------------------------------------------------------------
// collect_completed_games
// ---------------------------------------------------------------------------

int TrainingLoop::collect_completed_games() {
    int n = orchestrator_->collect_completed(collect_buf_, kMaxCollectBatch);
    for (int i = 0; i < n; ++i) {
        GameInstance* g = collect_buf_[i];

        int traj_len = g->trajectory_length;
        if (traj_len > 0) {
            // Use stack allocation for small trajectories; heap otherwise.
            std::vector<TrainingSample> samples(static_cast<size_t>(traj_len));
            int ns = extract_training_samples(*g, config_.td_mode,
                                              config_.td_lambda,
                                              samples.data(), traj_len);
            buffer_->add_batch(samples.data(), ns);
        }

        ++games_played_;

        // Recycle: new seed derived from games_played to stay deterministic
        uint64_t new_seed = static_cast<uint64_t>(games_played_)
                            * 6364136223846793005ULL;
        orchestrator_->recycle_game(g, new_seed);
    }
    return n;
}

// ---------------------------------------------------------------------------
// do_training_step
// ---------------------------------------------------------------------------

void TrainingLoop::do_training_step() {
    std::vector<TrainingSample> batch(static_cast<size_t>(config_.train_batch_size));
    int n = buffer_->sample_batch(batch.data(), config_.train_batch_size,
                                  sample_rng_);
    if (n <= 0) return;

    // Interleaved layout → split into flat arrays for ModelTrainer
    std::vector<float>  states(static_cast<size_t>(n) * kTensorSize);
    std::vector<double> targets(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        std::memcpy(states.data() + i * kTensorSize,
                    batch[static_cast<size_t>(i)].state,
                    kTensorSize * sizeof(float));
        targets[static_cast<size_t>(i)] = batch[static_cast<size_t>(i)].target;
    }

    last_loss_ = trainer_->train_step(states.data(), targets.data(), n);
    ++training_step_;

    // Decay temperature
    temperature_ = std::max(config_.min_temperature,
                            temperature_ * config_.temperature_decay);

    // Decay heuristic weight linearly
    if (training_step_ < config_.heuristic_decay_steps) {
        heuristic_weight_ = config_.initial_heuristic_weight * 
            (1.0 - static_cast<double>(training_step_) / config_.heuristic_decay_steps);
    } else {
        heuristic_weight_ = 0.0;
    }

    // Push updated config to self-play workers
    solver_config_.placement_temperature = temperature_;
    solver_config_.hold_temperature      = temperature_;
    solver_config_.exploration_enabled   = (temperature_ > 0.0);
    solver_config_.heuristic_weight      = heuristic_weight_;
    orchestrator_->update_solver_config(solver_config_);
}

// ---------------------------------------------------------------------------
// maybe_swap_model
// ---------------------------------------------------------------------------

void TrainingLoop::maybe_swap_model() {
    if (training_step_ % config_.model_swap_interval != 0) return;

    auto new_model = trainer_->clone_for_inference(infer_device_);
    new_model->eval();
    inference_->swap_model(new_model);
}

// ---------------------------------------------------------------------------
// maybe_checkpoint
// ---------------------------------------------------------------------------

void TrainingLoop::maybe_checkpoint() {
    if (training_step_ % config_.checkpoint_interval != 0) return;

    std::string stem = config_.checkpoint_dir + "/checkpoint_step_"
                       + std::to_string(training_step_);

    // Save model + optimizer
    trainer_->save_checkpoint(stem, training_step_, temperature_, epsilon_);

    // Save replay buffer alongside the checkpoint
    buffer_->save(stem + ".buffer");

    // Prune old checkpoints (handles .model, .optimizer, .buffer)
    prune_old_checkpoints(config_.checkpoint_dir, config_.max_checkpoints);

    // Append metrics to log
    log_metrics(config_.log_path, metrics());
}

// ---------------------------------------------------------------------------
// maybe_evaluate
// ---------------------------------------------------------------------------

void TrainingLoop::maybe_evaluate() {
    if (config_.eval_interval <= 0) return;
    if (training_step_ % config_.eval_interval != 0) return;

    EvalResult eval = run_evaluation(
        trainer_->model_mut(), trainer_->device(),
        tables_, config_.eval_games,
        static_cast<uint64_t>(training_step_));

    log_evaluation(config_.log_dir, training_step_, eval);

    last_eval_win_rate_ = eval.nn_win_rate();
}
