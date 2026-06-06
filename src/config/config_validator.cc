#include "config/config_validator.h"

#include <filesystem>

#include "distil/distil_config.h"
#include "engine/game_traits.h"
#include "engine/tensor.h"   // kTensorVersion*, tensor_size_for_version
#include "model/model_config.h"
#include "self_play/game_instance.h"  // GameInstance::kMaxAfterstates

static void validate_model_common(const ModelConfig& m, ValidationResult& r,
                                  const char* prefix) {
    auto fail = [&](const std::string& msg) {
        r.fail(std::string(prefix) + msg);
    };

    if (m.input_size <= 0)    fail("input_size must be positive");
    if (m.hidden_layers <= 0) fail("hidden_layers must be positive");
    if (m.hidden_width <= 0)  fail("hidden_width must be positive");
    if (m.learning_rate <= 0.0) fail("learning_rate must be positive");
    if (m.output_activation != "tanh" && m.output_activation != "sigmoid")
        fail("output_activation must be \"tanh\" or \"sigmoid\"");
    if (m.loss_function != "mse" && m.loss_function != "bce")
        fail("loss_function must be \"mse\" or \"bce\"");
    if (m.architecture != "mlp" && m.architecture != "resnet")
        fail("architecture must be \"mlp\" or \"resnet\"");

    if (m.game_variant != kGameVariant1v1 && m.game_variant != kGameVariant2v2) {
        fail("game_variant must be 1v1 or 2v2");
    } else if (m.tensor_version < kTensorVersionV1 ||
               m.tensor_version > kTensorVersionLatest) {
        fail("tensor_version must be in [" + std::to_string(kTensorVersionV1) +
             ", " + std::to_string(kTensorVersionLatest) + "]");
    } else {
        // input_size is fully determined by (game_variant, tensor_version).
        const int expected = (m.game_variant == kGameVariant2v2)
            ? tensor_size_for_version<Yams2v2>(m.tensor_version)
            : tensor_size_for_version<Yams1v1>(m.tensor_version);
        if (m.input_size != expected) {
            fail("input_size must equal " + std::to_string(expected) +
                 " for game_variant=" +
                 (m.game_variant == kGameVariant2v2 ? std::string("2v2")
                                                    : std::string("1v1")) +
                 " tensor_version=" + std::to_string(m.tensor_version));
        }
    }
}

static void validate_self_play_common(const SelfPlayConfig& sp,
                                      ValidationResult& r,
                                      const char* prefix) {
    auto fail = [&](const std::string& msg) {
        r.fail(std::string(prefix) + msg);
    };
    if (sp.num_workers <= 0) fail("num_workers must be positive");
    if (sp.num_games   <= 0) fail("num_games must be positive");
    if (sp.max_inference_batch < kMaxAfterstateRequests)
        fail("max_inference_batch must be >= kMaxAfterstateRequests (" +
             std::to_string(kMaxAfterstateRequests) + ")");
    if (sp.min_games_per_batch <= 0) fail("min_games_per_batch must be positive");
    if (sp.batch_timeout_ms    <= 0) fail("batch_timeout_ms must be positive");
}

static ValidationResult validate_training(const AppConfig& cfg) {
    ValidationResult r;
    const TrainingConfig& tc  = cfg.training;
    const SelfPlayConfig& sp  = tc.self_play;
    const ModelConfig&    m   = tc.model;

    if (cfg.num_steps <= 0)
        r.fail("num_steps must be positive");

    // Replay buffer
    if (tc.replay_capacity <= 0)
        r.fail("training.replay_capacity must be positive");
    if (tc.min_buffer_size <= 0)
        r.fail("training.min_buffer_size must be positive");
    if (tc.min_buffer_size > tc.replay_capacity)
        r.fail("training.min_buffer_size must not exceed replay_capacity");
    if (tc.train_batch_size <= 0)
        r.fail("training.train_batch_size must be positive");
    if (tc.train_batch_size > tc.min_buffer_size)
        r.fail("training.train_batch_size must not exceed min_buffer_size");

    // Intervals
    if (tc.model_swap_interval <= 0)
        r.fail("training.model_swap_interval must be positive");
    if (tc.checkpoint_interval <= 0)
        r.fail("training.checkpoint_interval must be positive");
    if (tc.max_checkpoints <= 0)
        r.fail("training.max_checkpoints must be positive");

    // Temperature
    if (tc.initial_temperature <= 0.0)
        r.fail("training.initial_temperature must be positive");
    if (tc.min_temperature <= 0.0)
        r.fail("training.min_temperature must be positive");
    if (tc.min_temperature > tc.initial_temperature)
        r.fail("training.min_temperature must not exceed initial_temperature");
    if (tc.temperature_decay <= 0.0 || tc.temperature_decay > 1.0)
        r.fail("training.temperature_decay must be in (0, 1]");
    if (tc.temperature_decay_start_step < 0)
        r.fail("training.temperature_decay_start_step must be non-negative");
    if (tc.temperature_decay_start_value < 0.0)
        r.fail("training.temperature_decay_start_value must be non-negative");
    if (tc.temperature_decay_start_value > 0.0 &&
        tc.temperature_decay_start_value < tc.min_temperature)
        r.fail("training.temperature_decay_start_value must not be below min_temperature");

    // Duel margin maximization
    if (tc.duel_margin_maximization_scale <= 0.0)
        r.fail("training.duel_margin_maximization_scale must be positive");

    // Evaluation (eval_interval == 0 means disabled, which is valid)
    if (tc.eval_interval < 0)
        r.fail("training.eval_interval must be non-negative (0 = disabled)");
    if (tc.eval_games <= 0)
        r.fail("training.eval_games must be positive");

    // Learning-rate back-off
    if (tc.lr_backoff_factor <= 0.0 || tc.lr_backoff_factor > 1.0)
        r.fail("training.lr_backoff_factor must be in (0, 1]");
    if (tc.lr_backoff_min_lr <= 0.0)
        r.fail("training.lr_backoff_min_lr must be positive");
    if (tc.lr_backoff_enabled && tc.eval_interval <= 0)
        r.fail("training.lr_backoff_enabled requires eval_interval > 0 "
               "(back-off is driven by eval win rate)");

    validate_self_play_common(sp, r, "training.self_play.");
    validate_model_common(m, r, "training.model.");
    return r;
}

static ValidationResult validate_distil(const AppConfig& cfg) {
    ValidationResult r;
    const DistilConfig&   dc = cfg.distil;
    const SelfPlayConfig& sp = dc.self_play;
    const ModelConfig&    m  = dc.student_model;

    // Teacher selection
    if (dc.teacher_kind == TeacherKind::kHeuristic) {
        if (dc.teacher_heuristic_version < 1 || dc.teacher_heuristic_version > 17) {
            r.fail("distil.teacher_heuristic_version must be in [1, 17]");
        }
    } else {  // kNN
        if (dc.teacher_checkpoint_path.empty()) {
            r.fail("distil.teacher_checkpoint is required when teacher_kind=nn");
        } else {
            // The .model file is the canonical artefact; the .optimizer sibling
            // is optional (and unused for an inference-only teacher).
            const std::string model_path = dc.teacher_checkpoint_path + ".model";
            if (!std::filesystem::exists(model_path)) {
                r.fail("distil.teacher_checkpoint: model file not found: " + model_path);
            }
        }
    }

    if (dc.reference_heuristic_version < 1 || dc.reference_heuristic_version > 17) {
        r.fail("distil.reference_heuristic_version must be in [1, 17]");
    }

    // Replay buffer
    if (dc.train_batch_size <= 0)
        r.fail("distil.train_batch_size must be positive");
    if (dc.replay_buffer_capacity <= 0)
        r.fail("distil.replay_buffer_capacity must be positive");
    if (dc.min_samples_to_start <= 0)
        r.fail("distil.min_samples_to_start must be positive");
    if (dc.min_samples_to_start < dc.train_batch_size)
        r.fail("distil.min_samples_to_start must be >= train_batch_size "
               "(the first draw needs at least one full batch of samples)");
    if (dc.replay_buffer_capacity < dc.min_samples_to_start)
        r.fail("distil.replay_buffer_capacity must be >= min_samples_to_start "
               "(otherwise the buffer can never accumulate enough to start training)");
    if (!(dc.samples_per_train >= 1.0))
        r.fail("distil.samples_per_train must be >= 1.0 "
               "(values < 1 would drop samples without training on them)");
    if (!(dc.samples_per_games_rate > 0.0 && dc.samples_per_games_rate <= 1.0))
        r.fail("distil.samples_per_games_rate must be in (0, 1]");

    // Checkpointing
    if (dc.checkpoint_interval <= 0)
        r.fail("distil.checkpoint_interval must be positive");
    if (dc.max_checkpoints <= 0)
        r.fail("distil.max_checkpoints must be positive");

    // Convergence / stopping
    if (dc.max_steps <= 0)
        r.fail("distil.max_steps must be positive");
    if (dc.min_steps < 0)
        r.fail("distil.min_steps must be non-negative");
    if (dc.min_steps > dc.max_steps)
        r.fail("distil.min_steps must not exceed max_steps");
    if (dc.eval_interval <= 0)
        r.fail("distil.eval_interval must be positive (distillation needs eval to detect convergence)");
    if (dc.eval_games <= 0)
        r.fail("distil.eval_games must be positive");
    if (dc.convergence_patience <= 0)
        r.fail("distil.convergence_patience must be positive");
    if (dc.final_eval_games <= 0)
        r.fail("distil.final_eval_games must be positive");

    // Target normalisation
    if (dc.duel_margin_maximization_scale <= 0.0)
        r.fail("distil.duel_margin_maximization_scale must be positive");
    // Student's output_activation must agree with the normalisation form so
    // teacher labels land in-range for the student's loss.
    if (dc.use_duel_margin_maximization && m.output_activation != "tanh") {
        r.fail("distil.student.output_activation must be \"tanh\" when "
               "use_duel_margin_maximization=true (targets in [-1, 1])");
    }
    if (!dc.use_duel_margin_maximization && m.output_activation != "sigmoid") {
        r.fail("distil.student.output_activation must be \"sigmoid\" when "
               "use_duel_margin_maximization=false (targets in [0, 1])");
    }

    validate_self_play_common(sp, r, "distil.self_play.");
    validate_model_common(m, r, "distil.student.");
    return r;
}

ValidationResult validate_config(const AppConfig& cfg) {
    if (cfg.mode == "distil") return validate_distil(cfg);
    return validate_training(cfg);
}
