#!/bin/bash
# =============================================================================
# Pro Yams AI — Learning Rate & TD Mode Grid Search
#
# Investigates interaction between:
#   - TD Mode (mc, td0, tdlambda 0.8, tdlambda 0.95)
#   - Learning Rate (0.0001, 0.0005, 0.001)
#   - Train Batch Size (4096, 8192, 16384)
#
# Metrics: Win Rate, Avg Margin, Loss
# =============================================================================

set -uo pipefail

BINARY="./build/pro_yams_ai"
BASE_LOG_DIR="./logs/grid_lr"
SUMMARY_FILE="${BASE_LOG_DIR}/grid_lr_summary.csv"
START_RUN="${1:-1}"

# --- FIXED PARAMETERS (from ab_control_sprint.yaml or user request) ---
NUM_STEPS=6000
REPLAY_CAPACITY=1000000
MIN_BUFFER_SIZE=200000
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
# CONFIG REGENERATION
# ─────────────────────────────────────────────────────────────────────────────

generate_config() {
    local run_dir="$1"
    local td_mode="$2"
    local td_lambda="$3"
    local lr="$4"
    local tbs="$5"

    cat > "${run_dir}/config.yaml" <<EOF
num_steps: ${NUM_STEPS}

training:
  replay_capacity: ${REPLAY_CAPACITY}
  min_buffer_size: ${MIN_BUFFER_SIZE}
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
# GRID DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

TD_MODES=("mc" "td0" "tdlambda" "tdlambda")
TD_LAMBDAS=(0.8 0.8 0.8 0.95)
LRS=(0.00001)
BS=(4096)

TOTAL_RUNS=$(( ${#TD_MODES[@]} * ${#LRS[@]} * ${#BS[@]} ))

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
    echo "run_id,td_mode,td_lambda,lr,batch_size,win_rate,avg_margin,loss" > "$SUMMARY_FILE"
fi

run_count=0
for (( i=0; i<${#TD_MODES[@]}; i++ )); do
    for lr in "${LRS[@]}"; do
        for bs in "${BS[@]}"; do
            run_count=$((run_count + 1))
            
            if [[ $run_count -lt $START_RUN ]]; then
                continue
            fi

            td_mode="${TD_MODES[$i]}"
            td_lambda="${TD_LAMBDAS[$i]}"
            
            run_id="run_$(printf '%02d' $run_count)"
            run_dir="${BASE_LOG_DIR}/${run_id}"
            mkdir -p "${run_dir}"

            label="${td_mode}_L${td_lambda}_LR${lr}_BS${bs}"
            
            echo "────────────────────────────────────────────────────────────"
            echo "[${run_count}/${TOTAL_RUNS}] ${run_id}: ${label}"
            echo "────────────────────────────────────────────────────────────"

            generate_config "$run_dir" "$td_mode" "$td_lambda" "$lr" "$bs"

            start_time=$(date +%s)
            if "${BINARY}" --mode train --debug_mode 1 --config "${run_dir}/config.yaml" 2>"${run_dir}/stderr.log"; then
                end_time=$(date +%s)
                elapsed=$(( end_time - start_time ))
                
                wr=$(extract_win_rate "${run_dir}/eval_log.csv")
                am=$(extract_avg_margin "${run_dir}/eval_log.csv")
                loss=$(extract_loss "${run_dir}/training_log.csv")
                
                echo "  ✓ Win Rate: ${wr}  Margin: ${am}  Loss: ${loss}  Time: ${elapsed}s"
                echo "${run_id},${td_mode},${td_lambda},${lr},${bs},${wr},${am},${loss}" >> "$SUMMARY_FILE"
            else
                end_time=$(date +%s)
                elapsed=$(( end_time - start_time ))
                echo "  ✗ FAILED after ${elapsed}s. Check ${run_dir}/stderr.log"
                echo "${run_id},${td_mode},${td_lambda},${lr},${bs},FAILED,FAILED,FAILED" >> "$SUMMARY_FILE"
            fi
            echo ""
        done
    done
done

echo "============================================================"
echo "  Grid Search Complete. Results: ${SUMMARY_FILE}"
echo "============================================================"
echo ""
echo "Top 10 by Win Rate:"
printf "%-10s %-10s %-8s %-8s %-10s %-10s %-10s\n" "RUN" "MODE" "LAMBDA" "LR" "BATCH" "WIN_RATE" "MARGIN"
echo "──────────────────────────────────────────────────────────────────────────────────────────────"
tail -n +2 "$SUMMARY_FILE" | grep -v "FAILED" | sort -t',' -k6 -rn | head -n 10 | while IFS=',' read -r rid tdm tdl lr bs wr am ls; do
    printf "%-10s %-10s %-8s %-8s %-10s %-10s %-10s\n" "$rid" "$tdm" "$tdl" "$lr" "$bs" "$wr" "$am"
done
