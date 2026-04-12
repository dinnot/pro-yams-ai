#!/bin/bash
# =============================================================================
# Pro Yams AI — Learning Rate & TD Mode Sweep
#
# Investigates selected configurations of TD modes and learning rates.
# Format: "td_mode td_lambda lr tbs replay_cap min_buffer label"
# =============================================================================

set -uo pipefail

BINARY="./build/pro_yams_ai"
BASE_LOG_DIR="./logs/grid_lr"
SUMMARY_FILE="${BASE_LOG_DIR}/grid_lr_summary.csv"
START_RUN="${1:-1}"

# --- FIXED PARAMETERS (from ab_control_sprint.yaml or user request) ---
NUM_STEPS=6000
TRAIN_STEPS_PER_COLLECT=0.125
MODEL_SWAP_INTERVAL=10
CHECKPOINT_INTERVAL=2000
MAX_CHECKPOINTS=2
EVAL_INTERVAL=2000
EVAL_GAMES=500
INITIAL_TEMP=1.0
MIN_TEMP=0.1
TEMP_DECAY=0.999712

# Self-play params
MIB=4096
MGPB=2
BATCH_TIMEOUT_MS=5
NW=24
NG=2048
NC=2

# ─────────────────────────────────────────────────────────────────────────────
# CURATED CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS=(
    # Mode     Lambda   LR        Batch   Cap      MinBuf  Label
    "mc        0.8      0.00001    4096    1000000  200000  mc_lr1e-5"
    "td0       0.8      0.00001    4096    50000    20000   td0_lr1e-5"
    "tdlambda  0.8      0.00001    4096    50000    20000   tdl0.8_lr1e-5"
    "tdlambda  0.95     0.00001    4096    50000    20000   tdl0.95_lr1e-5"
    "tdlambda  0.95     0.00001    4096    30000    10000   tdl0.95_lr1e-5"
    "tdlambda  0.95     0.00001    4096    70000    30000   tdl0.95_lr1e-5"
    "tdlambda  0.95     0.00002    4096    50000    20000   tdl0.95_lr1e-5"
    "tdlambda  0.95     0.000005   4096    50000    20000   tdl0.95_lr1e-5"
    "tdlambda  0.95     0.00004    4096    50000    20000   tdl0.95_lr1e-5"
    "tdlambda  0.95     0.00008    4096    50000    20000   tdl0.95_lr1e-5"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG REGENERATION
# ─────────────────────────────────────────────────────────────────────────────

generate_config() {
    local run_dir="$1"
    local td_mode="$2"
    local td_lambda="$3"
    local lr="$4"
    local tbs="$5"
    local replay_cap="$6"
    local min_buf="$7"

    cat > "${run_dir}/config.yaml" <<EOF
num_steps: ${NUM_STEPS}

training:
  replay_capacity: ${replay_cap}
  min_buffer_size: ${min_buf}
  train_batch_size: ${tbs}
  train_steps_per_collect: ${TRAIN_STEPS_PER_COLLECT}
  model_swap_interval: ${MODEL_SWAP_INTERVAL}
  checkpoint_interval: ${CHECKPOINT_INTERVAL}
  max_checkpoints: ${MAX_CHECKPOINTS}

  td_mode: ${td_mode}
  td_lambda: ${td_lambda}

  initial_temperature: ${INITIAL_TEMP}
  min_temperature: ${MIN_TEMP}
  temperature_decay: ${TEMP_DECAY}

  eval_interval: ${EVAL_INTERVAL}
  eval_games: ${EVAL_GAMES}

  checkpoint_dir: ${run_dir}/checkpoints
  log_dir: ${run_dir}
  log_path: ${run_dir}/training_log.csv

  self_play:
    max_inference_batch: ${MIB}
    min_games_per_batch: ${MGPB}
    batch_timeout_ms: ${BATCH_TIMEOUT_MS}
    num_workers: ${NW}
    num_games: ${NG}
    num_coordinators: ${NC}
    debug_mode: true
    debug_log_path: "${run_dir}/debug_coordinator.log"

  model:
    input_size: 809
    hidden_layers: 3
    hidden_width: 256
    learning_rate: ${lr}
    debug_mode: true
    debug_log_path: "${run_dir}/debug_batch.log"
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION UTILS
# ─────────────────────────────────────────────────────────────────────────────

extract_win_rate() {
    local csv="$1"
    if [[ -f "$csv" ]] && [[ $(wc -l < "$csv") -ge 2 ]]; then
        tail -1 "$csv" | awk -F',' '{print $7}'
    else
        echo "N/A"
    fi
}

extract_avg_margin() {
    local csv="$1"
    if [[ -f "$csv" ]] && [[ $(wc -l < "$csv") -ge 2 ]]; then
        tail -1 "$csv" | awk -F',' '{print $10}'
    else
        echo "N/A"
    fi
}

extract_loss() {
    local csv="$1"
    if [[ -f "$csv" ]] && [[ $(wc -l < "$csv") -ge 2 ]]; then
        tail -1 "$csv" | awk -F',' '{print $4}'
    else
        echo "N/A"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found at $BINARY. Run 'make -j\$(nproc)' first."
    exit 1
fi

mkdir -p "${BASE_LOG_DIR}"

# Header for summary
if [[ "$START_RUN" -le 1 ]]; then
    echo "run_id,label,td_mode,td_lambda,lr,batch_size,replay_cap,min_buffer,win_rate,avg_margin,loss" > "$SUMMARY_FILE"
fi

total_configs=${#CONFIGS[@]}

for (( i=0; i<total_configs; i++ )); do
    run_num=$((i + 1))
    
    if [[ $run_num -lt $START_RUN ]]; then
        continue
    fi

    read -r td_mode td_lambda lr tbs cap mib label <<< "${CONFIGS[$i]}"
    
    run_id="run_$(printf '%02d' $run_num)"
    run_dir="${BASE_LOG_DIR}/${run_id}"
    mkdir -p "${run_dir}"

    echo "────────────────────────────────────────────────────────────"
    echo "[${run_num}/${total_configs}] ${run_id}: ${label}"
    echo "  ${td_mode}(λ=${td_lambda}) LR=${lr} BS=${tbs} CAP=${cap} MIB=${mib}"
    echo "────────────────────────────────────────────────────────────"

    generate_config "$run_dir" "$td_mode" "$td_lambda" "$lr" "$tbs" "$cap" "$mib"

    start_time=$(date +%s)
    if "${BINARY}" --mode train --debug_mode 1 --config "${run_dir}/config.yaml" 2>"${run_dir}/stderr.log"; then
        end_time=$(date +%s)
        elapsed=$(( end_time - start_time ))
        
        wr=$(extract_win_rate "${run_dir}/eval_log.csv")
        am=$(extract_avg_margin "${run_dir}/eval_log.csv")
        loss=$(extract_loss "${run_dir}/training_log.csv")
        
        echo "  ✓ Win Rate: ${wr}  Margin: ${am}  Loss: ${loss}  Time: ${elapsed}s"
        echo "${run_id},${label},${td_mode},${td_lambda},${lr},${tbs},${cap},${mib},${wr},${am},${loss}" >> "$SUMMARY_FILE"
    else
        end_time=$(date +%s)
        elapsed=$(( end_time - start_time ))
        echo "  ✗ FAILED after ${elapsed}s. Check ${run_dir}/stderr.log"
        echo "${run_id},${label},${td_mode},${td_lambda},${lr},${tbs},${cap},${mib},FAILED,FAILED,FAILED" >> "$SUMMARY_FILE"
    fi
    echo ""
done

echo "============================================================"
echo "  Grid Search Complete. Results: ${SUMMARY_FILE}"
echo "============================================================"
echo ""
echo "Top 10 by Win Rate:"
printf "%-10s %-15s %-10s %-8s %-8s %-10s %-10s\n" "RUN" "LABEL" "MODE" "LAMBDA" "LR" "WIN_RATE" "MARGIN"
echo "────────────────────────────────────────────────────────────────────────────────────────────────"
tail -n +2 "$SUMMARY_FILE" | grep -v "FAILED" | sort -t',' -k9 -rn | head -n 10 | while IFS=',' read -r rid lbl tdm tdl lr bs cap mib wr am ls; do
    printf "%-10s %-15s %-10s %-8s %-8s %-10s %-10s\n" "$rid" "$lbl" "$tdm" "$tdl" "$lr" "$wr" "$am"
done
