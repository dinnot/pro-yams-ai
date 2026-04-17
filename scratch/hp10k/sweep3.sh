#!/bin/bash
# Phase 3: try to push past the 0.77 ceiling.
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
    # Stack top two resnet winners
    "final1_resnet_hw512_tspc  hidden_width=512 train_steps_per_collect=0.0625"
    # Deeper MLP (hl=5 instead of 3)
    "final2_mlp_hl5            architecture=mlp hidden_layers=5"
    # MLP + tdl=0.99 (longer traces to compensate for Watkins cutting)
    "final3_mlp_tdl99          architecture=mlp td_lambda=0.99"
)

for entry in "${runs[@]}"; do
    args=($entry)
    run_one "${args[@]}"
done

echo ""
echo "=== SWEEP3 COMPLETE ==="
