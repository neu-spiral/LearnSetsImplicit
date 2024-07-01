#!/bin/bash

# Directory
dir="EquiVSet"

# Possible values for data_name and mode
data_names=("gaussian" "moons") #"amazon" "moons"
modes=("diffMF" "ind" "copula")

# Possible values for amazon_cat
amazon_cats=("toys" "furniture" "gear" "carseats" "bath" "health" "diaper" "bedding" "safety" "feeding" "apparel" "media")

# Number of repetitions
repetitions=5
RNN_steps=6

# Loop over each data_name
for data_name in "${data_names[@]}"; do
  if [ "$data_name" == "amazon" ]; then
    # Loop over each amazon_cat
    for amazon_cat in "${amazon_cats[@]}"; do
      # Loop over each mode
      for mode in "${modes[@]}"; do
          for ((i=1; i<=repetitions; i++)); do
            # Generate a random seed
            seed=$i
            iters=$K
            echo "Running $dir/main.py with --data_name=$data_name --amazon_cat=$amazon_cat --mode=$mode --seed=$i (Run $i)"
            (cd "$dir" && python main.py equivset --train --cuda --data_name="$data_name" --amazon_cat="$amazon_cat" --mode="$mode" --seed=$i)
          done
      done
    done
  else
    # Loop over each mode
    for mode in "${modes[@]}"; do
      for ((K=1; K<=RNN_steps; K++)); do
        for ((i=1; i<=repetitions; i++)); do
          # Generate a random seed
          echo "Running $dir/main.py with --data_name=$data_name --mode=$mode --seed=$i (Run $i) --RNN_steps=$K"
          (cd "$dir" && CUDA_VISIBLE_DEVICES=1 python main.py equivset --train --cuda --data_name="$data_name" --mode="$mode" --seed=$i --RNN_steps=$K)
        done
      done
    done
  fi
done
