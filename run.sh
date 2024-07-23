#!/bin/bash

source .venv/bin/activate

set -x  # for printing commands

current_datetime=$(date '+%Y-%m-%d-%H-%M')

for seed in {1..1}
do
  # Save a video of the agent on the first run
  python3 train.py \
      --wandb \
      --project="safelife-debug" \
      --n_envs=1 \
      --name="test-1-env"
done
