#!/usr/bin/env python3
"""Write a training YAML for hp10k sweep.

Usage:  python make_config.py <run_dir> <overrides as k=v...>

Baseline reflects test_tdl_100k_new1 (87% @ 100k), adapted for 10k steps.
"""
import sys, pathlib

BASELINE = {
    "num_steps": 10000,
    "replay_capacity": 30000,
    "min_buffer_size": 10000,
    "train_batch_size": 4096,
    "train_steps_per_collect": 0.125,
    "model_swap_interval": 10,
    "checkpoint_interval": 10000,
    "max_checkpoints": 1,
    "td_mode": "tdlambda",
    "td_lambda": 0.95,
    "use_duel_margin_maximization": "true",
    "duel_margin_maximization_scale": 4000.0,
    "initial_temperature": 1.0,
    "min_temperature": 0.1,
    # reach min_temp (~0.1) around step 8000
    "temperature_decay": 0.999712,
    "initial_epsilon": 0.0,
    "eval_interval": 2000,
    "eval_games": 500,
    "initial_heuristic_weight": 1.0,
    "heuristic_decay_steps": 50000,
    # self_play
    "max_inference_batch": 4096,
    "min_games_per_batch": 2,
    "batch_timeout_ms": 5,
    "num_workers": 24,
    "num_games": 2048,
    "num_coordinators": 2,
    # model
    "input_size": 809,
    "hidden_layers": 3,
    "hidden_width": 256,
    "learning_rate": 0.00032,
    "output_activation": "tanh",
    "loss_function": "mse",
    "architecture": "resnet",
}

SELF_PLAY_KEYS = {"max_inference_batch","min_games_per_batch","batch_timeout_ms",
                  "num_workers","num_games","num_coordinators"}
MODEL_KEYS = {"input_size","hidden_layers","hidden_width","learning_rate",
              "output_activation","loss_function","architecture"}

def render(params, run_dir):
    run_dir = str(run_dir)
    # booleans as YAML lowercase literals
    def y(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        return v
    p = params
    lines = []
    lines.append(f"num_steps: {p['num_steps']}")
    lines.append("")
    lines.append("training:")
    for k in ["replay_capacity","min_buffer_size","train_batch_size",
              "train_steps_per_collect","model_swap_interval",
              "checkpoint_interval","max_checkpoints","td_mode","td_lambda",
              "use_duel_margin_maximization","duel_margin_maximization_scale",
              "initial_temperature","min_temperature","temperature_decay",
              "initial_epsilon","eval_interval","eval_games",
              "initial_heuristic_weight","heuristic_decay_steps"]:
        lines.append(f"  {k}: {y(p[k])}")
    lines.append(f"  checkpoint_dir: {run_dir}/checkpoints")
    lines.append(f"  log_dir: {run_dir}")
    lines.append(f"  log_path: {run_dir}/training_log.csv")
    lines.append("  debug_mode: true")
    lines.append("  self_play:")
    for k in SELF_PLAY_KEYS:
        lines.append(f"    {k}: {y(p[k])}")
    lines.append(f'    debug_log_path: "{run_dir}/debug_coordinator.log"')
    lines.append("  model:")
    for k in MODEL_KEYS:
        lines.append(f"    {k}: {y(p[k])}")
    lines.append("    debug_mode: true")
    lines.append(f'    debug_log_path: "{run_dir}/debug_batch.log"')
    return "\n".join(lines) + "\n"

def parse_val(v):
    if v in ("true","false"):
        return v
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: pass
    return v

def main():
    run_dir = pathlib.Path(sys.argv[1]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    params = dict(BASELINE)
    for tok in sys.argv[2:]:
        k, v = tok.split("=", 1)
        params[k] = parse_val(v)
    (run_dir / "config.yaml").write_text(render(params, run_dir))
    print(f"wrote {run_dir / 'config.yaml'}")

if __name__ == "__main__":
    main()
