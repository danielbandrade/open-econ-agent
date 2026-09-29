#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

for seed in 123 2024; do
  for model in llama3.1:8b qwen2.5:7b mistral:7b gemma2:9b qwen2.5:14b; do
    tag=${model/:/-}
    log="logs/run-${tag}-seed${seed}.log"
    echo "[$(date -Is)] starting $model seed $seed" | tee -a logs/replications.log
    OLLAMA_CONCURRENCY=3 python3 -u simulate.py \
      --policy_model llama --ollama_model "$model" \
      --num_agents 100 --episode_length 240 --seed "$seed" \
      > "$log" 2>&1
    echo "[$(date -Is)] finished $model seed $seed" | tee -a logs/replications.log
  done
done
