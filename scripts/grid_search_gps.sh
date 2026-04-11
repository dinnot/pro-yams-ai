#!/bin/bash
# =============================================================================
# Pro Yams AI — Curated GPS Benchmark Sweep
#
# 40 pre-selected configs organized in 5 phases:
#   Phase 1: Coordinator count scaling (1-4 coordinators)
#   Phase 2: Game pool sizing
#   Phase 3: Worker scaling
#   Phase 4: Inference batch tuning
#   Phase 5: Training cadence sweep
#
# Usage:
#   ./scripts/grid_search_gps.sh [start_run]
#
# Pass a start_run number to resume from a specific run (1-indexed).
# =============================================================================

set -euo pipefail

BINARY="./build/pro_yams_ai"
BASE_LOG_DIR="./logs"
SUMMARY_FILE="${BASE_LOG_DIR}/grid_summary.csv"
START_RUN="${1:-1}"

# ─────────────────────────────────────────────────────────────────────────────
# FIXED PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

NUM_STEPS=2000
REPLAY_CAPACITY=500000
MIN_BUFFER_SIZE=5000
MODEL_SWAP_INTERVAL=99999
CHECKPOINT_INTERVAL=99999
MAX_CHECKPOINTS=1
TD_MODE="mc"
TD_LAMBDA=0.8
INITIAL_TEMP=1.0
MIN_TEMP=0.1
TEMP_DECAY=0.99994245
EVAL_INTERVAL=2000
EVAL_GAMES=1
BATCH_TIMEOUT_MS=5
INPUT_SIZE=809
HIDDEN_LAYERS=3
HIDDEN_WIDTH=256
LEARNING_RATE=0.0005

# ─────────────────────────────────────────────────────────────────────────────
# CURATED CONFIGS
# Format: "tbs tspc mib mgpb nw nc ng label"
#
# tbs  = train_batch_size
# tspc = train_steps_per_collect
# mib  = max_inference_batch
# mgpb = min_games_per_batch
# nw   = num_workers
# nc   = num_coordinators
# ng   = num_games
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS=(
    # ── Phase 1: Coordinator count scaling ────────────────────────────────
    # Fix: tbs=2048, tspc=4, mib=4096, nw=20, ng=2048
    # Vary: num_coordinators 1-4
    "2048 4 4096  2 20 1 2048  P1_1coord"
    "2048 4 4096  2 20 2 2048  P1_2coord"
    "2048 4 4096  2 20 3 2048  P1_3coord"
    "2048 4 4096  2 20 4 2048  P1_4coord"

    # ── Phase 2: Game pool sizing ─────────────────────────────────────────
    # Fix: tbs=2048, tspc=4, mib=4096, nw=20
    # Vary: num_games with 1 and 2 coordinators
    
    # 1 coordinator
    "2048 4 4096  2 20 1  512  P2_1c_512g"
    "2048 4 4096  2 20 1 1024  P2_1c_1kg"
    "2048 4 4096  2 20 1 4096  P2_1c_4kg"
    
    # 2 coordinators
    "2048 4 4096  2 20 2  512  P2_2c_512g"
    "2048 4 4096  2 20 2 1024  P2_2c_1kg"
    "2048 4 4096  2 20 2 3072  P2_2c_3kg"
    "2048 4 4096  2 20 2 4096  P2_2c_4kg"
    "2048 4 4096  2 20 2 6144  P2_2c_6kg"

    # ── Phase 3: Worker scaling ───────────────────────────────────────────
    # Fix: tbs=2048, tspc=4, mib=4096, nc=2, ng=2048
    # Vary: num_workers
    "2048 4 4096  2 12 2 2048  P3_12w"
    "2048 4 4096  2 16 2 2048  P3_16w"
    "2048 4 4096  2 24 2 2048  P3_24w"
    "2048 4 4096  2 28 2 2048  P3_28w"
    "2048 4 4096  2 32 2 2048  P3_32w"

    # ── Phase 4: Inference batch tuning ───────────────────────────────────
    # Fix: tbs=2048, tspc=4, nw=20, nc=2, ng=2048
    # Vary: max_inference_batch and min_games_per_batch
    "2048 4 1024  2 20 2 2048  P4_mib1k"
    "2048 4 2048  2 20 2 2048  P4_mib2k"
    "2048 4 8192  2 20 2 2048  P4_mib8k"
    "2048 4 16384 2 20 2 2048  P4_mib16k"
    "2048 4 4096  1 20 2 2048  P4_mgpb1"
    "2048 4 4096  3 20 2 2048  P4_mgpb3"
    "2048 4 4096  5 20 2 2048  P4_mgpb5"

    # Larger mib with more games to feed it
    "2048 4 8192  2 20 2 4096  P4_mib8k_4kg"
    "2048 4 16384 2 20 3 4096  P4_mib16k_3c"

    # ── Phase 5: Training cadence sweep ───────────────────────────────────
    # Fix: mib=4096, mgpb=2, nw=20, nc=2, ng=2048
    # Vary: train_batch_size × train_steps_per_collect
    
    # Sweep tspc with tbs=2048
    "2048  1 4096  2 20 2 2048  P5_tspc1"
    "2048  2 4096  2 20 2 2048  P5_tspc2"
    "2048  8 4096  2 20 2 2048  P5_tspc8"
    "2048 16 4096  2 20 2 2048  P5_tspc16"
    "2048 32 4096  2 20 2 2048  P5_tspc32"

    # Sweep tbs with tspc=4
    "256   4 4096  2 20 2 2048  P5_tbs256"
    "512   4 4096  2 20 2 2048  P5_tbs512"
    "1024  4 4096  2 20 2 2048  P5_tbs1k"
    "4096  4 4096  2 20 2 2048  P5_tbs4k"
    "8192  4 4096  2 20 2 2048  P5_tbs8k"

    # Interesting combos: big batch + few steps vs small batch + many steps
    "4096  1 4096  2 20 2 2048  P5_4k_x1"
    "4096  2 4096  2 20 2 2048  P5_4k_x2"
    "512   8 4096  2 20 2 2048  P5_512_x8"
    "512  16 4096  2 20 2 2048  P5_512_x16"
    "1024  8 4096  2 20 2 2048  P5_1k_x8"
    "256  16 4096  2 20 2 2048  P5_256_x16"
)

# ─────────────────────────────────────────────────────────────────────────────

generate_config() {
    local run_dir="$1"
    local tbs="$2" tspc="$3" mib="$4" mgpb="$5" nw="$6" nc="$7" ng="$8"

    cat > "${run_dir}/config.yaml" <<EOF
num_steps: ${NUM_STEPS}

training:
  replay_capacity: ${REPLAY_CAPACITY}
  min_buffer_size: ${MIN_BUFFER_SIZE}
  train_batch_size: ${tbs}
  train_steps_per_collect: ${tspc}
  model_swap_interval: ${MODEL_SWAP_INTERVAL}
  checkpoint_interval: ${CHECKPOINT_INTERVAL}
  max_checkpoints: ${MAX_CHECKPOINTS}

  td_mode: ${TD_MODE}
  td_lambda: ${TD_LAMBDA}

  initial_temperature: ${INITIAL_TEMP}
  min_temperature: ${MIN_TEMP}
  temperature_decay: ${TEMP_DECAY}

  eval_interval: ${EVAL_INTERVAL}
  eval_games: ${EVAL_GAMES}

  checkpoint_dir: ${run_dir}/checkpoints
  log_dir: ${run_dir}
  log_path: ${run_dir}/training_log.csv

  self_play:
    max_inference_batch: ${mib}
    min_games_per_batch: ${mgpb}
    batch_timeout_ms: ${BATCH_TIMEOUT_MS}
    num_workers: ${nw}
    num_games: ${ng}
    num_coordinators: ${nc}
    debug_mode: true
    debug_log_path: "${run_dir}/debug_coordinator.log"

  model:
    input_size: ${INPUT_SIZE}
    hidden_layers: ${HIDDEN_LAYERS}
    hidden_width: ${HIDDEN_WIDTH}
    learning_rate: ${LEARNING_RATE}
    debug_mode: true
    debug_log_path: "${run_dir}/debug_batch.log"
EOF
}

extract_gps() {
    local csv="$1"
    if [[ -f "$csv" ]] && [[ $(wc -l < "$csv") -ge 2 ]]; then
        tail -1 "$csv" | awk -F',' '{print $7}'
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

total=${#CONFIGS[@]}

echo "============================================================"
echo "  Pro Yams AI — Curated GPS Benchmark (${total} configs)"
echo "  Starting from run ${START_RUN}"
echo "============================================================"
echo ""

mkdir -p "${BASE_LOG_DIR}"

# Write header only if starting fresh
if [[ "$START_RUN" -le 1 ]]; then
    echo "run_id,label,train_batch_size,train_steps_per_collect,max_inference_batch,min_games_per_batch,num_workers,num_coordinators,num_games,gps,games_played" \
        > "$SUMMARY_FILE"
fi

for (( i=0; i<total; i++ )); do
    run_num=$((i + 1))
    
    if [[ $run_num -lt $START_RUN ]]; then
        continue
    fi

    read -r tbs tspc mib mgpb nw nc ng label <<< "${CONFIGS[$i]}"

    run_id="grid_$(printf '%02d' $run_num)"
    run_dir="${BASE_LOG_DIR}/${run_id}"
    mkdir -p "${run_dir}"

    generate_config "$run_dir" "$tbs" "$tspc" "$mib" "$mgpb" "$nw" "$nc" "$ng"

    echo "────────────────────────────────────────────────────────────"
    echo "[${run_num}/${total}] ${run_id} — ${label}"
    echo "  tbs=${tbs} tspc=${tspc} mib=${mib} mgpb=${mgpb} nw=${nw} nc=${nc} ng=${ng}"
    echo "────────────────────────────────────────────────────────────"

    start_time=$(date +%s)

    if "${BINARY}" --mode train --config "${run_dir}/config.yaml" 2>"${run_dir}/stderr.log"; then
        end_time=$(date +%s)
        elapsed=$(( end_time - start_time ))
        gps=$(extract_gps "${run_dir}/training_log.csv")
        games=$(tail -1 "${run_dir}/training_log.csv" 2>/dev/null | awk -F',' '{print $2}')
        echo "  ✓ GPS: ${gps}  Games: ${games:-N/A}  Time: ${elapsed}s"
    else
        end_time=$(date +%s)
        elapsed=$(( end_time - start_time ))
        gps="FAILED"
        games="FAILED"
        echo "  ✗ Run failed after ${elapsed}s! Check ${run_dir}/stderr.log"
    fi

    echo "${run_id},${label},${tbs},${tspc},${mib},${mgpb},${nw},${nc},${ng},${gps},${games:-N/A}" \
        >> "$SUMMARY_FILE"

    echo ""
done

echo ""
echo "============================================================"
echo "  Results: ${SUMMARY_FILE}"
echo "============================================================"
echo ""
echo "All results (sorted by GPS, best first):"
echo ""
printf "%-10s %-18s %5s %5s %6s %4s %3s %3s %5s %8s\n" \
       "RUN" "LABEL" "TBS" "TSPC" "MIB" "MGPB" "NW" "NC" "NG" "GPS"
echo "───────────────────────────────────────────────────────────────────────────────"
tail -n +2 "$SUMMARY_FILE" | grep -v "FAILED" | sort -t',' -k10 -rn | while IFS=',' read -r rid lbl tbs tspc mib mgpb nw nc ng gps gp; do
    printf "%-10s %-18s %5s %5s %6s %4s %3s %3s %5s %8s\n" \
           "$rid" "$lbl" "$tbs" "$tspc" "$mib" "$mgpb" "$nw" "$nc" "$ng" "$gps"
done
