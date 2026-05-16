#include "training/training_loop.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <type_traits>
#include <vector>

#include "engine/tensor.h"
#include "eval/evaluator.h"
#include "eval/eval_logging.h"
#include "self_play/training_data.h"
#include "training/logging.h"

namespace {

std::vector<std::string> list_checkpoint_stems(const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<std::pair<int, std::string>> found;
    if (!fs::exists(dir)) return {};

    const std::string prefix = "checkpoint_step_";
    const std::string suffix = ".model";
    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) != 0) continue;
        if (name.size() <= prefix.size() + suffix.size()) continue;
        if (name.substr(name.size() - suffix.size()) != suffix) continue;
        try {
            int s = std::stoi(name.substr(prefix.size(),
                                          name.size() - prefix.size() - suffix.size()));
            found.emplace_back(s, dir + "/" + name.substr(0, name.size() - suffix.size()));
        } catch (...) {}
    }
    std::sort(found.begin(), found.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::string> out;
    out.reserve(found.size());
    for (auto& p : found) out.push_back(std::move(p.second));
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <typename Traits>
TrainingLoopT<Traits>::TrainingLoopT(const TrainingConfig& config,
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
      sample_rng_(0xDEADBEEF12345678ULL),
      start_time_(std::chrono::steady_clock::now()) {
    // Defensive checks: the ModelConfig must agree with the trait it's being
    // instantiated for. A mismatch here would surface much later as a tensor
    // shape error at the first inference call.
    assert(config_.model.input_size == Traits::kTensorSize &&
           "ModelConfig::input_size must match Traits::kTensorSize");
    constexpr int expected_variant =
        std::is_same_v<Traits, Yams1v1> ? kGameVariant1v1 : kGameVariant2v2;
    assert(config_.model.game_variant == expected_variant &&
           "ModelConfig::game_variant must match the trait");

    buffer_ = std::make_unique<Buffer>(config_.replay_capacity);

    ModelConfig mc = config_.model;
    mc.debug_mode = config_.debug_mode;
    if (mc.debug_mode) {
        mc.debug_log_path = config_.log_dir + "/debug_batch.log";
    }
    trainer_ = std::make_unique<ModelTrainer>(mc, train_device_);

    auto inf_model = trainer_->clone_for_inference(infer_device_);
    inf_model->eval();
    inference_ = std::make_shared<InferenceEngine>(inf_model, infer_device_);

    if (config_.past_opponent_probability > 0.0) {
        opponent_loader_ = std::make_unique<ModelTrainer>(mc, infer_device_);
        auto opp_model = opponent_loader_->clone_for_inference(infer_device_);
        opp_model->eval();
        opponent_inference_ = std::make_shared<InferenceEngine>(opp_model, infer_device_);
    }

    solver_config_.placement_temperature = temperature_;
    solver_config_.hold_temperature      = temperature_;
    solver_config_.exploration_enabled   = (temperature_ > 0.0);
    solver_config_.debug_mode            = config_.debug_mode;
    solver_config_.heuristic_weight      = heuristic_weight_;
    solver_config_.heuristic_version     = config_.heuristic_version;
    solver_config_.use_duel_margin_maximization   = config_.use_duel_margin_maximization;
    solver_config_.duel_margin_maximization_scale = config_.duel_margin_maximization_scale;
    solver_config_.use_pbrs          = config_.use_pbrs;
    solver_config_.pbrs_upper_reward = config_.pbrs_upper_reward;
    solver_config_.pbrs_clean_reward = config_.pbrs_clean_reward;
    solver_config_.debug_log_path        = config_.log_dir + "/debug_game_0.log";

    config_.self_play.debug_mode = config_.debug_mode;
    if (config_.debug_mode) {
        config_.self_play.debug_log_path = config_.log_dir + "/debug_coordinator.log";
    }

    orchestrator_ = std::make_unique<Orchestr>(
        config_.self_play, tables_, *inference_, solver_config_,
        opponent_inference_ ? opponent_inference_.get() : nullptr);
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------

template <typename Traits>
void TrainingLoopT<Traits>::stop() {
    stop_flag_.store(true);
}

// ---------------------------------------------------------------------------
// metrics
// ---------------------------------------------------------------------------

template <typename Traits>
TrainingMetrics TrainingLoopT<Traits>::metrics() const {
    TrainingMetrics m;
    m.training_step     = training_step_;
    m.games_played      = games_played_;
    m.samples_in_buffer = buffer_->size();
    m.loss              = last_loss_;
    m.temperature       = temperature_;
    m.epsilon           = epsilon_;

    auto now = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(now - start_time_).count();
    if (seconds > 0.001) {
        m.games_per_second = static_cast<double>(games_played_) / seconds;
    } else {
        m.games_per_second = 0.0;
    }
    m.total_samples_trained = total_samples_trained_;

    m.latest_eval_win_rate   = last_eval_win_rate_;
    return m;
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

template <typename Traits>
void TrainingLoopT<Traits>::run(int num_steps) {
    std::filesystem::create_directories(config_.checkpoint_dir);
    if (!config_.log_dir.empty()) {
        std::filesystem::create_directories(config_.log_dir);
    }
    if (!config_.log_path.empty()) {
        auto log_parent = std::filesystem::path(config_.log_path).parent_path();
        if (!log_parent.empty()) {
            std::filesystem::create_directories(log_parent);
        }
    }

    {
        auto initial_inf_model = trainer_->clone_for_inference(infer_device_);
        initial_inf_model->eval();
        inference_->swap_model(initial_inf_model);
    }

    refresh_opponent_model();

    orchestrator_->start();

    if (config_.logs_on_start) {
        log_metrics(config_.log_path, metrics());

        if (config_.eval_interval > 0) {
            maybe_evaluate();
        }
    }

    while (!stop_flag_.load(std::memory_order_relaxed) &&
           training_step_ < num_steps) {
        auto t_collect_start = std::chrono::steady_clock::now();
        int collected = collect_completed_games();
        auto t_collect_end = std::chrono::steady_clock::now();
        double collect_ms = std::chrono::duration<double, std::milli>(t_collect_end - t_collect_start).count();

        if (buffer_->size() >= config_.min_buffer_size) {
            double steps_per_game = config_.train_steps_per_collect > 0.0 ? config_.train_steps_per_collect : 1.0;

            pending_train_steps_ += static_cast<double>(collected) * steps_per_game;
            int steps_to_do = static_cast<int>(pending_train_steps_);
            pending_train_steps_ -= steps_to_do;

            double total_train_ms = 0;
            for (int i = 0; i < steps_to_do; ++i) {
                auto t_train_start = std::chrono::steady_clock::now();
                do_training_step();
                auto t_train_end = std::chrono::steady_clock::now();
                total_train_ms += std::chrono::duration<double, std::milli>(t_train_end - t_train_start).count();

                maybe_swap_model();
                maybe_checkpoint();
                maybe_evaluate();
                if (training_step_ >= num_steps || stop_flag_.load(std::memory_order_relaxed)) {
                    break;
                }
            }
            if (steps_to_do > 0 && config_.debug_mode) {
                std::string queue_log_path = config_.log_dir + "/debug_queues.log";
                std::ofstream f(queue_log_path, std::ios::app);
                if (f.is_open()) {
                    f << "Collected " << collected << " games in " << collect_ms << " ms\n"
                      << "Ran " << steps_to_do << " full train steps in " << total_train_ms << " ms\n\n";
                }
            }
        }

        if (collected == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    orchestrator_->stop();
}

// ---------------------------------------------------------------------------
// collect_completed_games
// ---------------------------------------------------------------------------

template <typename Traits>
int TrainingLoopT<Traits>::collect_completed_games() {
    int n = orchestrator_->collect_completed(collect_buf_, kMaxCollectBatch);
    const bool past_opp_enabled = orchestrator_->has_opponent_inference()
                                  && opponent_ready_
                                  && config_.past_opponent_probability > 0.0;

    for (int i = 0; i < n; ++i) {
        Instance* g = collect_buf_[i];

        int traj_len = g->trajectory_length;
        if (traj_len > 0) {
            int exclude = g->use_past_opponent ? g->past_opponent_player : -1;
            std::vector<Sample> samples(static_cast<size_t>(traj_len));
            int ns = extract_training_samples<Traits>(*g, config_.td_mode,
                                                     config_.td_lambda,
                                                     config_.use_duel_margin_maximization,
                                                     config_.duel_margin_maximization_scale,
                                                     config_.use_pbrs,
                                                     samples.data(), traj_len,
                                                     exclude);
            buffer_->add_batch(samples.data(), ns);
        }

        ++games_played_;

        bool use_past = false;
        int  past_player = -1;
        if (past_opp_enabled) {
            double activate = opponent_rng_.uniform_double();
            if (activate < config_.past_opponent_probability) {
                use_past = true;
                // Pick a random seat; the worker and trainer treat any seat on
                // the same team as the past opponent (see Traits::are_teammates
                // in worker.cc / training_data.cc). In 1v1 this is one player;
                // in 2v2 it covers an entire team, keeping the current network
                // playing strictly against the past one (no mixed-team self-play).
                past_player = opponent_rng_.uniform_int(0, Traits::kNumPlayers - 1);
            }
        }

        uint64_t new_seed = (static_cast<uint64_t>(games_played_) * 6364136223846793005ULL)
                            ^ (static_cast<uint64_t>(training_step_) * 1442695040888963407ULL);
        orchestrator_->recycle_game(g, new_seed, use_past, past_player);
    }
    return n;
}

// ---------------------------------------------------------------------------
// do_training_step
// ---------------------------------------------------------------------------

template <typename Traits>
void TrainingLoopT<Traits>::do_training_step() {
    std::vector<Sample> batch(static_cast<size_t>(config_.train_batch_size));
    auto t1 = std::chrono::steady_clock::now();
    int n = buffer_->sample_batch(batch.data(), config_.train_batch_size, sample_rng_);
    auto t2 = std::chrono::steady_clock::now();

    if (n <= 0) return;

    constexpr int kTSize = Traits::kTensorSize;
    std::vector<float>  states(static_cast<size_t>(n) * kTSize);
    std::vector<double> targets(static_cast<size_t>(n));

    auto t3 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        std::memcpy(states.data() + i * kTSize,
                    batch[static_cast<size_t>(i)].state,
                    kTSize * sizeof(float));
        targets[static_cast<size_t>(i)] = batch[static_cast<size_t>(i)].target;
    }
    auto t4 = std::chrono::steady_clock::now();

    auto t_start = std::chrono::steady_clock::now();
    last_loss_ = trainer_->train_step(states.data(), targets.data(), n);
    auto t_end = std::chrono::steady_clock::now();

    double sample_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double memcpy_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double train_ms  = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (config_.debug_mode && training_step_ % 200 == 0) {
        std::string queue_log_path = config_.log_dir + "/debug_queues.log";
        std::ofstream f(queue_log_path, std::ios::app);
        if (f.is_open()) {
            f << "--- Queue Sizes & Timing @ Step " << training_step_ << " ---\n"
              << "Available : " << orchestrator_->available_queue_size() << "\n"
              << "Completed : " << orchestrator_->completed_queue_size() << "\n"
              << "Trainer train_step() time: " << train_ms << " ms for batch size " << n << "\n"
              << "sample_batch() time: " << sample_ms << " ms\n"
              << "memcpy loop time: " << memcpy_ms << " ms\n\n";
        }
    }

    ++training_step_;
    total_samples_trained_ += n;

    if (training_step_ < config_.heuristic_decay_steps) {
        heuristic_weight_ = config_.initial_heuristic_weight *
            (1.0 - static_cast<double>(training_step_) / config_.heuristic_decay_steps);
    } else {
        heuristic_weight_ = 0.0;
    }

    if (training_step_ >= config_.temperature_decay_start_step) {
        if (training_step_ == config_.temperature_decay_start_step &&
            config_.temperature_decay_start_value > 0.0) {
            temperature_ = config_.temperature_decay_start_value;
        } else {
            temperature_ = std::max(config_.min_temperature,
                                    temperature_ * config_.temperature_decay);
        }
    }

    solver_config_.placement_temperature = temperature_;
    solver_config_.hold_temperature      = temperature_;
    solver_config_.exploration_enabled   = (temperature_ > 0.0);
    solver_config_.heuristic_weight      = heuristic_weight_;
    solver_config_.heuristic_version     = config_.heuristic_version;
    solver_config_.use_duel_margin_maximization   = config_.use_duel_margin_maximization;
    solver_config_.duel_margin_maximization_scale = config_.duel_margin_maximization_scale;
    solver_config_.use_pbrs          = config_.use_pbrs;
    solver_config_.pbrs_upper_reward = config_.pbrs_upper_reward;
    solver_config_.pbrs_clean_reward = config_.pbrs_clean_reward;
    orchestrator_->update_solver_config(solver_config_);
}

// ---------------------------------------------------------------------------
// maybe_swap_model
// ---------------------------------------------------------------------------

template <typename Traits>
void TrainingLoopT<Traits>::maybe_swap_model() {
    if (training_step_ % config_.model_swap_interval != 0) return;

    auto new_model = trainer_->clone_for_inference(infer_device_);
    new_model->eval();
    inference_->swap_model(new_model);
}

// ---------------------------------------------------------------------------
// maybe_checkpoint
// ---------------------------------------------------------------------------

template <typename Traits>
void TrainingLoopT<Traits>::maybe_checkpoint() {
    if (training_step_ % config_.checkpoint_interval != 0) return;

    std::string stem = config_.checkpoint_dir + "/checkpoint_step_"
                       + std::to_string(training_step_);

    trainer_->save_checkpoint(stem, training_step_, temperature_, epsilon_);
    buffer_->save(stem + ".buffer");
    prune_old_checkpoints(config_.checkpoint_dir, config_.max_checkpoints);

    refresh_opponent_model();

    log_metrics(config_.log_path, metrics());
}

// ---------------------------------------------------------------------------
// refresh_opponent_model
// ---------------------------------------------------------------------------

template <typename Traits>
void TrainingLoopT<Traits>::refresh_opponent_model() {
    if (!opponent_inference_ || !opponent_loader_) return;

    auto stems = list_checkpoint_stems(config_.checkpoint_dir);
    if (stems.empty()) return;

    int idx = opponent_rng_.uniform_int(0, static_cast<int>(stems.size()) - 1);
    try {
        opponent_loader_->load_weights(stems[static_cast<size_t>(idx)]);
        auto opp_clone = opponent_loader_->clone_for_inference(infer_device_);
        opp_clone->eval();
        opponent_inference_->swap_model(opp_clone);
        opponent_ready_ = true;
    } catch (const std::exception& e) {
        (void)e;
    }
}

// ---------------------------------------------------------------------------
// maybe_evaluate — uses the templated evaluator so both 1v1 and 2v2 produce
// in-training eval metrics against the heuristic bot.
// ---------------------------------------------------------------------------

template <typename Traits>
void TrainingLoopT<Traits>::maybe_evaluate() {
    if (config_.eval_interval <= 0) return;
    if (training_step_ % config_.eval_interval != 0) return;

    EvalResult eval = run_evaluation<Traits>(
        trainer_->model_mut(), trainer_->device(),
        tables_, config_.eval_games,
        static_cast<uint64_t>(training_step_));

    log_evaluation(config_.log_dir, training_step_, eval);

    last_eval_win_rate_ = eval.nn_win_rate();
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template class TrainingLoopT<Yams1v1>;
template class TrainingLoopT<Yams2v2>;
