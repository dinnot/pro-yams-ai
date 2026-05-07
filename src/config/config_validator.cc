#include "config/config_validator.h"

#include "self_play/game_instance.h"  // GameInstance::kMaxAfterstates

ValidationResult validate_config(const AppConfig& cfg) {
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

    // Self-play
    if (sp.num_workers <= 0)
        r.fail("training.self_play.num_workers must be positive");
    if (sp.num_games <= 0)
        r.fail("training.self_play.num_games must be positive");
    if (sp.max_inference_batch < kMaxAfterstateRequests)
        r.fail("training.self_play.max_inference_batch must be >= kMaxAfterstateRequests (" +
               std::to_string(kMaxAfterstateRequests) + ")");
    if (sp.min_games_per_batch <= 0)
        r.fail("training.self_play.min_games_per_batch must be positive");
    if (sp.batch_timeout_ms <= 0)
        r.fail("training.self_play.batch_timeout_ms must be positive");

    // Model
    if (m.input_size <= 0)
        r.fail("training.model.input_size must be positive");
    if (m.hidden_layers <= 0)
        r.fail("training.model.hidden_layers must be positive");
    if (m.hidden_width <= 0)
        r.fail("training.model.hidden_width must be positive");
    if (m.learning_rate <= 0.0)
        r.fail("training.model.learning_rate must be positive");
    if (m.output_activation != "tanh" && m.output_activation != "sigmoid")
        r.fail("training.model.output_activation must be \"tanh\" or \"sigmoid\"");
    if (m.loss_function != "mse" && m.loss_function != "bce")
        r.fail("training.model.loss_function must be \"mse\" or \"bce\"");
    if (m.architecture != "mlp" && m.architecture != "resnet")
        r.fail("training.model.architecture must be \"mlp\" or \"resnet\"");

    return r;
}
