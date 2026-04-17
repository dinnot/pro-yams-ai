#!/bin/bash
# Variations sweep. Baseline = tdlambda/0.95/lr3.2e-4/tanh+mse+resnet/duel_margin_max.
set -uo pipefail
cd /home/sorin/pro_yams_ai

run_one() {
    local label="$1"; shift
    local overrides="$*"
    local run_dir="logs/hp10k/${label}"
    echo "===================================================="
    echo " $(date +%H:%M:%S)  ${label}  <${overrides}>"
    echo "===================================================="
    rm -rf "${run_dir}"
    python3 scratch/hp10k/make_config.py "${run_dir}" ${overrides} > /dev/null
    local t0=$(date +%s)
    ./build/pro_yams_ai --mode train --debug_mode 1 --config "${run_dir}/config.yaml" \
        > "${run_dir}/stdout.log" 2> "${run_dir}/stderr.log"
    local rc=$?
    local t1=$(date +%s)
    local dt=$((t1-t0))
    if [ -f "${run_dir}/eval_log.csv" ]; then
        local eval_line=$(tail -1 ${run_dir}/eval_log.csv)
    else
        local eval_line="NO_EVAL"
    fi
    if [ -f "${run_dir}/training_log.csv" ]; then
        local train_line=$(tail -1 ${run_dir}/training_log.csv)
    else
        local train_line="NO_TRAIN"
    fi
    echo "  ${label} rc=${rc}  time=${dt}s"
    echo "  eval:  ${eval_line}"
    echo "  train: ${train_line}"
    echo "${label},${dt},${rc},${eval_line},${train_line}" >> logs/hp10k/sweep_summary.csv
}

# Header (idempotent: if file exists, leave it).
if [ ! -f logs/hp10k/sweep_summary.csv ]; then
    echo "label,seconds,rc,eval_ts,eval_step,eval_games,eval_nn_wins,eval_heur_wins,eval_draws,win_rate,wr_p0,wr_p1,avg_margin,train_step,games_played,buffer_size,loss,temp,eps,gps,total_samples" > logs/hp10k/sweep_summary.csv
fi

# ── Runs to execute (pass label as $1, rest of positional args are overrides) ──
runs=(
    "run02_lr_hi       learning_rate=0.00064"
    "run03_lr_lo       learning_rate=0.00016"
    "run04_mlp         architecture=mlp"
    "run05_mc          td_mode=mc td_lambda=0.0"
    "run06_td0         td_mode=td0 td_lambda=0.0"
    "run07_tspc_hi     train_steps_per_collect=0.25"
    "run08_tspc_lo     train_steps_per_collect=0.0625"
    "run09_hw512       hidden_width=512"
    "run10_bce_mlp     architecture=mlp output_activation=sigmoid loss_function=bce use_duel_margin_maximization=false"
    "run11_lowtemp     temperature_decay_start_value=0.3 temperature_decay_start_step=500 temperature_decay=0.999712"
)

for entry in "${runs[@]}"; do
    # shellcheck disable=SC2206
    args=($entry)
    run_one "${args[@]}"
done

echo ""
echo "=== SWEEP COMPLETE ==="
