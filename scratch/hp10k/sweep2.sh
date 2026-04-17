#!/bin/bash
# Phase 2: combine top-3 winners from phase 1 (mlp, tspc_lo, hw512).
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
    local eval_line="NO_EVAL"
    local train_line="NO_TRAIN"
    [ -f "${run_dir}/eval_log.csv" ] && eval_line=$(tail -1 ${run_dir}/eval_log.csv)
    [ -f "${run_dir}/training_log.csv" ] && train_line=$(tail -1 ${run_dir}/training_log.csv)
    echo "  ${label} rc=${rc}  time=${dt}s"
    echo "  eval:  ${eval_line}"
    echo "  train: ${train_line}"
}

runs=(
    "combo1_mlp_tspc_lo     architecture=mlp train_steps_per_collect=0.0625"
    "combo2_mlp_hw512       architecture=mlp hidden_width=512"
    "combo3_mlp_mc          architecture=mlp td_mode=mc td_lambda=0.0"
    "combo4_mlp_hw512_tspc  architecture=mlp hidden_width=512 train_steps_per_collect=0.0625"
    "combo5_mlp_lr_hi       architecture=mlp learning_rate=0.00064"
)

for entry in "${runs[@]}"; do
    args=($entry)
    run_one "${args[@]}"
done

echo ""
echo "=== SWEEP2 COMPLETE ==="
