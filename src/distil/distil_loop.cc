#include "distil/distil_loop.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "distil/convergence.h"
#include "distil/teacher_heuristic.h"
#include "distil/teacher_nn.h"
#include "eval/eval_logging.h"
#include "eval/evaluator.h"
#include "heuristic/heuristic_bot.h"
#include "training/logging.h"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <typename Traits>
DistilLoopT<Traits>::DistilLoopT(const DistilConfig& config,
                                 const PrecomputedTables& tables,
                                 torch::Device training_device,
                                 torch::Device inference_device)
    : config_(config),
      tables_(tables),
      train_device_(training_device),
      infer_device_(inference_device),
      start_time_(std::chrono::steady_clock::now()) {
    assert(config_.student_model.input_size == Traits::kTensorSize &&
           "DistilConfig::student_model.input_size must match Traits::kTensorSize");

    // --- Student trainer ---
    ModelConfig mc = config_.student_model;
    mc.debug_mode = config_.debug_mode;
    if (mc.debug_mode) {
        mc.debug_log_path = config_.log_dir + "/debug_batch.log";
    }
    student_trainer_ = std::make_unique<ModelTrainer>(mc, train_device_);

    // --- Teacher ---
    switch (config_.teacher_kind) {
        case TeacherKind::kHeuristic:
            teacher_ = std::make_unique<HeuristicTeacher<Traits>>(
                static_cast<HeuristicVersion>(config_.teacher_heuristic_version),
                tables_,
                config_.use_duel_margin_maximization,
                config_.duel_margin_maximization_scale);
            break;
        case TeacherKind::kNN: {
            auto nn = std::make_unique<NNTeacher<Traits>>(
                config_.teacher_checkpoint_path,
                infer_device_,
                config_.use_duel_margin_maximization);
            nn_teacher_view_ = nn.get();
            teacher_ = std::move(nn);
            break;
        }
    }

    // --- Shuffle queue ---
    queue_ = std::make_unique<ShuffleQueueT<Traits>>(
        config_.shuffle_chunk_size,
        config_.min_chunk_size_to_start,
        config_.max_buffered_samples);

    // --- Solver config (greedy; worker overrides anyway, but keep the
    // normalisation flags consistent for any softmax fall-through). ---
    SolverConfig solver_cfg;
    solver_cfg.exploration_enabled            = false;
    solver_cfg.placement_temperature          = 0.0;
    solver_cfg.hold_temperature               = 0.0;
    solver_cfg.heuristic_weight               = 0.0;
    solver_cfg.heuristic_version              = config_.teacher_heuristic_version;
    solver_cfg.use_duel_margin_maximization   = config_.use_duel_margin_maximization;
    solver_cfg.duel_margin_maximization_scale = config_.duel_margin_maximization_scale;
    solver_cfg.compute_pre_roll_ev            = false;
    solver_cfg.debug_mode                     = config_.debug_mode;

    // --- Orchestrator ---
    orchestrator_ = std::make_unique<DistilOrchestratorT<Traits>>(
        config_.self_play, tables_, *teacher_, *queue_, solver_cfg);
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------

template <typename Traits>
void DistilLoopT<Traits>::stop() {
    stop_flag_.store(true, std::memory_order_relaxed);
    if (queue_) queue_->stop();  // Unblock any blocked draw_batch().
}

// ---------------------------------------------------------------------------
// metrics
// ---------------------------------------------------------------------------

template <typename Traits>
TrainingMetrics DistilLoopT<Traits>::metrics() const {
    TrainingMetrics m;
    m.training_step     = training_step_;
    m.games_played      = orchestrator_
        ? static_cast<int>(orchestrator_->total_games_completed())
        : 0;
    m.samples_in_buffer = queue_
        ? (queue_->serving_remaining() + queue_->accumulating_size())
        : 0;
    m.loss              = last_loss_;
    m.temperature       = 0.0;
    m.epsilon           = 0.0;
    auto now = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(now - start_time_).count();
    m.games_per_second = (seconds > 0.001 && orchestrator_)
        ? (orchestrator_->total_games_completed() / seconds)
        : 0.0;
    m.total_samples_trained = total_samples_trained_;
    m.total_samples_emitted = orchestrator_
        ? orchestrator_->total_samples_emitted() : 0;
    m.latest_eval_win_rate  = last_student_win_rate_;
    m.teacher_win_rate      = teacher_win_rate_;
    m.consecutive_passes    = consecutive_passes_;
    return m;
}

// ---------------------------------------------------------------------------
// cache_teacher_baseline — run the teacher against the reference heuristic
// once at startup. Used as the convergence target the student should match.
// ---------------------------------------------------------------------------

template <typename Traits>
void DistilLoopT<Traits>::cache_teacher_baseline() {
    const auto ref = static_cast<HeuristicVersion>(
        config_.reference_heuristic_version);

    // Use final_eval_games for the baseline — it's a one-shot measurement
    // that anchors every subsequent convergence check, so spending more
    // games here pays off (a noisy anchor widens the effective convergence
    // window beyond the configured delta).
    const int N = config_.final_eval_games;
    switch (config_.teacher_kind) {
        case TeacherKind::kHeuristic: {
            const auto teacher_v = static_cast<HeuristicVersion>(
                config_.teacher_heuristic_version);
            EvalResult r = run_heuristic_vs_heuristic<Traits>(
                tables_, teacher_v, ref, N,
                /*base_seed=*/0xBA5EBA11ULL);
            teacher_win_rate_ = r.nn_win_rate();
            // Log to eval_log.csv with step=0 so the row is clearly the baseline.
            if (!config_.log_dir.empty()) {
                log_evaluation(config_.log_dir, /*training_step=*/0, r);
            }
            std::printf("Teacher baseline (heuristic V%d vs V%d, %d games): "
                        "win_rate=%.3f  avg_margin=%.1f\n",
                        config_.teacher_heuristic_version,
                        config_.reference_heuristic_version,
                        r.total_games, r.nn_win_rate(), r.avg_duel_margin);
            std::fflush(stdout);
            break;
        }
        case TeacherKind::kNN: {
            assert(nn_teacher_view_ && "NN teacher constructor did not run");
            EvalResult r = run_evaluation_vs<Traits>(
                nn_teacher_view_->model(),
                nn_teacher_view_->device(),
                tables_, ref, N,
                /*base_seed=*/0xBA5EBA11ULL);
            teacher_win_rate_ = r.nn_win_rate();
            if (!config_.log_dir.empty()) {
                log_evaluation(config_.log_dir, /*training_step=*/0, r);
            }
            std::printf("Teacher baseline (NN from %s vs V%d, %d games): "
                        "win_rate=%.3f  avg_margin=%.1f\n",
                        config_.teacher_checkpoint_path.c_str(),
                        config_.reference_heuristic_version,
                        r.total_games, r.nn_win_rate(), r.avg_duel_margin);
            std::fflush(stdout);
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// maybe_evaluate — student-vs-reference eval, gated by eval_interval.
// ---------------------------------------------------------------------------

template <typename Traits>
void DistilLoopT<Traits>::maybe_evaluate() {
    if (config_.eval_interval <= 0) return;
    if (training_step_ % config_.eval_interval != 0) return;

    // Drain the deferred train step so eval reads committed weights.
    // (No EMA update here — that's owned by do_training_step, which will
    // observe the materialised value on its next call. Folding it in here
    // too would risk double-counting if eval is called twice in a row.)
    student_trainer_->wait_until_step_complete();

    const auto ref = static_cast<HeuristicVersion>(
        config_.reference_heuristic_version);

    // Seed shifted by training_step_ so each interval gets a distinct sample
    // of games; identical re-runs at the same step give identical results.
    const uint64_t seed = 0xE7A1CEEE00000000ULL ^
                          static_cast<uint64_t>(training_step_);

    EvalResult r = run_evaluation_vs<Traits>(
        student_trainer_->model_mut(),
        student_trainer_->device(),
        tables_,
        ref,
        config_.eval_games,
        seed);

    last_student_win_rate_ = r.nn_win_rate();

    // Two parallel criteria; either passing for `convergence_patience`
    // consecutive eval checks triggers convergence. mse criterion only
    // fires when config.convergence_match_mse > 0 (see within_match_mse).
    const bool wr_pass  = distil::within_win_rate_delta(
        last_student_win_rate_, teacher_win_rate_,
        config_.convergence_win_rate_delta);
    const bool mse_pass = distil::within_match_mse(
        rolling_mse_, config_.convergence_match_mse);
    consecutive_passes_ = distil::next_patience(
        consecutive_passes_, wr_pass || mse_pass);

    if (!config_.log_dir.empty()) {
        log_evaluation(config_.log_dir, training_step_, r);
    }
}

// ---------------------------------------------------------------------------
// convergence_satisfied — pure-helper wrapper that reads the loop's state.
// ---------------------------------------------------------------------------

template <typename Traits>
bool DistilLoopT<Traits>::convergence_satisfied() const {
    return distil::convergence_satisfied(training_step_,
                                          config_.min_steps,
                                          consecutive_passes_,
                                          config_.convergence_patience);
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

template <typename Traits>
void DistilLoopT<Traits>::run() {
    std::filesystem::create_directories(config_.checkpoint_dir);
    if (!config_.log_dir.empty()) {
        std::filesystem::create_directories(config_.log_dir);
    }
    if (!config_.log_path.empty()) {
        auto parent = std::filesystem::path(config_.log_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    }

    constexpr int kT = Traits::kTensorSize;
    train_states_.assign(
        static_cast<size_t>(config_.train_batch_size) * kT, 0.0f);
    train_targets_.assign(
        static_cast<size_t>(config_.train_batch_size), 0.0);

    start_time_ = std::chrono::steady_clock::now();

    // Cache the teacher's reference win-rate BEFORE starting workers — the
    // teacher is fixed, so this measurement is one-and-done. Doing it before
    // start() also avoids workers competing for the calling thread.
    cache_teacher_baseline();

    orchestrator_->start();

    stop_reason_ = StopReason::kNotStopped;
    while (training_step_ < config_.max_steps) {
        if (stop_flag_.load(std::memory_order_relaxed)) {
            stop_reason_ = StopReason::kStopSignal;
            break;
        }
        int n = queue_->draw_batch(train_states_.data(),
                                   train_targets_.data(),
                                   config_.train_batch_size);
        if (n <= 0) {
            // draw_batch returns 0 only after stop() — distinguish whether
            // the external stop_flag was already set (above) from the
            // case where the queue was drained for some other reason.
            stop_reason_ = stop_flag_.load(std::memory_order_relaxed)
                ? StopReason::kStopSignal
                : StopReason::kQueueDrained;
            break;
        }

        do_training_step(n);
        maybe_evaluate();
        maybe_checkpoint();
        if (convergence_satisfied()) {
            stop_reason_ = StopReason::kConvergence;
            break;
        }
    }
    if (stop_reason_ == StopReason::kNotStopped) {
        // Fell out of the while because training_step_ >= max_steps.
        stop_reason_ = StopReason::kMaxSteps;
    }

    // Capture the very last train step's loss before tearing down the
    // workers (deferred .item() — the last submitted step is still pending).
    double pending = student_trainer_->wait_until_step_complete();
    if (std::isfinite(pending)) {
        last_loss_ = pending;
        distil::update_ema(rolling_mse_, last_loss_, /*alpha=*/0.05);
    }

    // Stop the queue FIRST so any producers blocked on the back-pressure cap
    // wake up and exit; only then can orchestrator_->stop() join them.
    queue_->stop();
    orchestrator_->stop();

    // Always run the final report — gives a headline number with a tighter
    // CI than the per-interval eval and a single canonical CSV row for
    // downstream tooling.
    final_report();
}

// ---------------------------------------------------------------------------
// do_training_step
// ---------------------------------------------------------------------------

template <typename Traits>
void DistilLoopT<Traits>::do_training_step(int n) {
    // train_step returns the PREVIOUS step's loss (NaN on the very first
    // call) — the current step's loss is materialised at the start of the
    // next train_step. This lets the GPU's batch-N compute overlap with the
    // CPU's batch-N+1 sampling. EMA tolerates the 1-step lag.
    double prev_loss = student_trainer_->train_step(
        train_states_.data(), train_targets_.data(), n);
    ++training_step_;
    total_samples_trained_ += n;
    if (std::isfinite(prev_loss)) {
        last_loss_ = prev_loss;
        // The per-batch loss IS the student-vs-teacher MSE on this batch
        // (each sample is one (state, teacher_ev) pair). EMA-smooth so the
        // optional match-MSE convergence criterion isn't whipsawed by
        // per-batch noise.
        distil::update_ema(rolling_mse_, last_loss_, /*alpha=*/0.05);
    }
}

// ---------------------------------------------------------------------------
// maybe_checkpoint
// ---------------------------------------------------------------------------

template <typename Traits>
void DistilLoopT<Traits>::maybe_checkpoint() {
    if (config_.checkpoint_interval <= 0) return;
    if (training_step_ % config_.checkpoint_interval != 0) return;

    std::string stem = config_.checkpoint_dir + "/checkpoint_step_"
                       + std::to_string(training_step_);
    student_trainer_->save_checkpoint(stem, training_step_, 0.0, 0.0);
    prune_old_checkpoints(config_.checkpoint_dir, config_.max_checkpoints);

    if (!config_.log_path.empty()) {
        log_metrics(config_.log_path, metrics());
    }
}

// ---------------------------------------------------------------------------
// final_report — re-eval both sides with config.final_eval_games for a tight
// CI, print a single block, and append one row to distil_final.csv.
// ---------------------------------------------------------------------------

const char* stop_reason_str(DistilStopReason r) {
    switch (r) {
        case DistilStopReason::kNotStopped:   return "not_stopped";
        case DistilStopReason::kMaxSteps:     return "max_steps";
        case DistilStopReason::kConvergence:  return "convergence";
        case DistilStopReason::kStopSignal:   return "stop_signal";
        case DistilStopReason::kQueueDrained: return "queue_drained";
    }
    return "unknown";
}

template <typename Traits>
void DistilLoopT<Traits>::final_report() {
    const auto ref = static_cast<HeuristicVersion>(
        config_.reference_heuristic_version);
    const int N = config_.final_eval_games;

    // --- Teacher: re-eval with the larger sample. ---
    switch (config_.teacher_kind) {
        case TeacherKind::kHeuristic: {
            const auto teacher_v = static_cast<HeuristicVersion>(
                config_.teacher_heuristic_version);
            EvalResult r = run_heuristic_vs_heuristic<Traits>(
                tables_, teacher_v, ref, N,
                /*base_seed=*/0xBA5EBA12ULL);  // distinct from startup seed
            final_teacher_win_rate_ = r.nn_win_rate();
            break;
        }
        case TeacherKind::kNN: {
            assert(nn_teacher_view_);
            EvalResult r = run_evaluation_vs<Traits>(
                nn_teacher_view_->model(),
                nn_teacher_view_->device(),
                tables_, ref, N,
                /*base_seed=*/0xBA5EBA12ULL);  // distinct from startup seed
            final_teacher_win_rate_ = r.nn_win_rate();
            break;
        }
    }

    // --- Student: re-eval. ---
    {
        EvalResult r = run_evaluation_vs<Traits>(
            student_trainer_->model_mut(),
            student_trainer_->device(),
            tables_, ref, N,
            /*base_seed=*/0xF1A1C0DEULL);
        final_student_win_rate_ = r.nn_win_rate();
    }

    // --- Console block. ---
    const double gap = final_student_win_rate_ - final_teacher_win_rate_;
    const char*  reason = stop_reason_str(stop_reason_);

    std::printf("\n=== Distillation final report ===\n");
    if (config_.teacher_kind == TeacherKind::kHeuristic) {
        std::printf("Teacher (heuristic V%d) vs V%d: win_rate = %.4f  (%d games)\n",
                    config_.teacher_heuristic_version,
                    config_.reference_heuristic_version,
                    final_teacher_win_rate_, N);
    } else {
        std::printf("Teacher (NN) vs V%d:           win_rate = %.4f  (%d games)\n",
                    config_.reference_heuristic_version,
                    final_teacher_win_rate_, N);
    }
    std::printf("Student vs V%d:                 win_rate = %.4f  (%d games)\n",
                config_.reference_heuristic_version,
                final_student_win_rate_, N);
    std::printf("Win-rate gap: %+.4f  (convergence target was |gap| < %.4f)\n",
                gap, config_.convergence_win_rate_delta);
    std::printf("Stop reason: %s at step %d / max %d\n",
                reason, training_step_, config_.max_steps);
    std::fflush(stdout);

    // --- CSV row. ---
    if (!config_.log_dir.empty()) {
        namespace fs = std::filesystem;
        fs::create_directories(config_.log_dir);
        const std::string path = config_.log_dir + "/distil_final.csv";
        const bool write_header = !fs::exists(path) ||
                                  fs::file_size(path) == 0;
        std::ofstream f(path, std::ios::app);
        if (f) {
            if (write_header) {
                f << "stop_reason,training_step,max_steps,"
                     "teacher_win_rate,student_win_rate,gap,"
                     "convergence_delta,final_eval_games\n";
            }
            f << reason << ','
              << training_step_   << ','
              << config_.max_steps << ','
              << final_teacher_win_rate_ << ','
              << final_student_win_rate_ << ','
              << gap << ','
              << config_.convergence_win_rate_delta << ','
              << N << '\n';
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template class DistilLoopT<Yams1v1>;
template class DistilLoopT<Yams2v2>;
