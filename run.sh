#!/bin/bash

source .venv/bin/activate

set -x  # for printing commands

current_datetime=$(date '+%Y-%m-%d-%H-%M')

for seed in {2..2}
do
  # Save a video of the agent on the first run
  python3 train.py \
      --wandb \
      --project="safelife-debug" \
      --n_envs=8 \
      --name="test-1-env" \
      --seed=$seed \
      --log_dir=logs/seed_test_$current_datetime
done
