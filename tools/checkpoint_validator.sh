#!/usr/bin/env bash
set -e

for d in /mnt/checkpoints/grpo_run_v1/checkpoint-* ; do
  printf "%s  " "$d"

  MODEL_FILE=""
  # accept either safetensors or bin shards
  [[ -f "$d/model.safetensors"          ]] && MODEL_FILE=model.safetensors
  [[ -f "$d/pytorch_model.bin"          ]] && MODEL_FILE=pytorch_model.bin
  [[ -f "$d/pytorch_model-00001-of-"*   ]] && MODEL_FILE=$(ls "$d"/pytorch_model-00001-of-* | head -1)

  if [[ -n $MODEL_FILE ]]               \
     && [[ -f "$d/optimizer.pt"       ]] \
     && [[ -f "$d/scheduler.pt"       ]] \
     && [[ -f "$d/trainer_state.json" ]]; then
        echo "OK"
  else
        echo "BROKEN"
  fi
done
