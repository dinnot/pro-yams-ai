#!/bin/bash
# Minimal verification of grid_search_lr.sh logic

# 1. Create a dummy grid_search_lr_test.sh with NUM_STEPS=100 and only 1 config
cat > scripts/grid_search_lr_test.sh <<EOF
#!/bin/bash
set -uo pipefail
BINARY="./build/pro_yams_ai"
BASE_LOG_DIR="./logs/test_lr"
SUMMARY_FILE="\${BASE_LOG_DIR}/grid_lr_summary.csv"

NUM_STEPS=100
REPLAY_CAPACITY=10000
MIN_BUFFER_SIZE=1000
TRAIN_STEPS_PER_COLLECT=1.0
MODEL_SWAP_INTERVAL=10
CHECKPOINT_INTERVAL=100
MAX_CHECKPOINTS=1
EVAL_INTERVAL=100
EVAL_GAMES=2
INITIAL_TEMP=1.0
MIN_TEMP=0.1
TEMP_DECAY=0.999

# Self-play params
MIB=4096
MGPB=2
BATCH_TIMEOUT_MS=5
NW=2
NG=64
NC=1

generate_config() {
    local run_dir="\$1"
    local td_mode="\$2"
    local td_lambda="\$3"
    local lr="\$4"
    local tbs="\$5"

    cat > "\${run_dir}/config.yaml" <<EOC
num_steps: \${NUM_STEPS}
training:
  replay_capacity: \${REPLAY_CAPACITY}
  min_buffer_size: \${MIN_BUFFER_SIZE}
  train_batch_size: \${tbs}
  train_steps_per_collect: \${TRAIN_STEPS_PER_COLLECT}
  model_swap_interval: \${MODEL_SWAP_INTERVAL}
  checkpoint_interval: \${CHECKPOINT_INTERVAL}
  max_checkpoints: \${MAX_CHECKPOINTS}
  td_mode: \${td_mode}
  td_lambda: \${td_lambda}
  initial_temperature: \${INITIAL_TEMP}
  min_temperature: \${MIN_TEMP}
  temperature_decay: \${TEMP_DECAY}
  eval_interval: \${EVAL_INTERVAL}
  eval_games: \${EVAL_GAMES}
  checkpoint_dir: \${run_dir}/checkpoints
  log_dir: \${run_dir}
  log_path: \${run_dir}/training_log.csv
  self_play:
    max_inference_batch: \${MIB}
    min_games_per_batch: \${MGPB}
    batch_timeout_ms: \${BATCH_TIMEOUT_MS}
    num_workers: \${NW}
    num_games: \${NG}
    num_coordinators: \${NC}
  model:
    input_size: 809
    hidden_layers: 2
    hidden_width: 64
    learning_rate: \${lr}
EOC
}

extract_win_rate() { tail -1 "\$1" | awk -F',' '{print \$7}'; }
extract_avg_margin() { tail -1 "\$1" | awk -F',' '{print \$10}'; }
extract_loss() { tail -1 "\$1" | awk -F',' '{print \$4}'; }

mkdir -p "\${BASE_LOG_DIR}"
echo "run_id,td_mode,td_lambda,lr,batch_size,win_rate,avg_margin,loss" > "\$SUMMARY_FILE"

# Test 1 config: mc, lambda 0.8, lr 0.0005, bs 256
run_id="test_run"
run_dir="\${BASE_LOG_DIR}/\${run_id}"
mkdir -p "\${run_dir}"

generate_config "\${run_dir}" "mc" "0.8" "0.0005" "256"

if "\${BINARY}" --mode train --config "\${run_dir}/config.yaml" > "\${run_dir}/stdout.log" 2>"\${run_dir}/stderr.log"; then
    wr=\$(extract_win_rate "\${run_dir}/eval_log.csv")
    am=\$(extract_avg_margin "\${run_dir}/eval_log.csv")
    loss=\$(extract_loss "\${run_dir}/training_log.csv")
    echo "\${run_id},mc,0.8,0.0005,256,\${wr:-N/A},\${am:-N/A},\${ls:-N/A}" >> "\$SUMMARY_FILE"
    echo "SUCCESS: Win Rate \${wr}, Margin \${am}, Loss \${loss}"
else
    echo "FAILED: Check \${run_dir}/stderr.log"
    exit 1
fi
EOF

chmod +x scripts/grid_search_lr_test.sh
./scripts/grid_search_lr_test.sh
