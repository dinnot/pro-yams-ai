#!/bin/bash
# Run hp10k experiments serially. Baseline = tdlambda/0.95/lr3.2e-4/tanh+mse+resnet.
set -uo pipefail
cd /home/sorin/pro_yams_ai

# Usage: ./runs.sh <label> <override1> <override2> ...
# Single-run wrapper; caller loops over it.
label="$1"; shift
overrides="$*"
run_dir="logs/hp10k/${label}"

echo "===================================================="
echo " ${label}  <${overrides}>"
echo "===================================================="
rm -rf "${run_dir}"
python3 scratch/hp10k/make_config.py "${run_dir}" ${overrides}

t0=$(date +%s)
./build/pro_yams_ai --mode train --debug_mode 1 --config "${run_dir}/config.yaml" \
    > "${run_dir}/stdout.log" 2> "${run_dir}/stderr.log"
rc=$?
t1=$(date +%s)
echo "  ${label} rc=${rc}  time=$((t1-t0))s"
if [ -f "${run_dir}/eval_log.csv" ]; then
    echo "  final eval: $(tail -1 ${run_dir}/eval_log.csv)"
fi
if [ -f "${run_dir}/training_log.csv" ]; then
    echo "  final train: $(tail -1 ${run_dir}/training_log.csv)"
fi
