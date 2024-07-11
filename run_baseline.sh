#!/bin/bash

# Directory
dir="EquiVSet"

# Possible values for data_name and mode

data_names=("celeba") #"moons" "gaussian" "amazon" "bindingdb"
modes=("diffMF" "ind" "copula")



# Possible values for amazon_cat
amazon_cats=("toys" "furniture" "gear" "carseats" "bath" "health" "diaper" "bedding" "safety" "feeding" "apparel" "media")

# Learning rates
learning_rates=("0.001" "0.0001" "0.00001")

# Number of repetitions
folds=5
num_layers=3

# Loop over each data_name
for data_name in "${data_names[@]}"; do
  if [ "$data_name" == "amazon" ]; then
    # Loop over each amazon_cat
    for amazon_cat in "${amazon_cats[@]}"; do
      # Loop over each mode
      for mode in "${modes[@]}"; do
        for lr in "${learning_rates[@]}"; do
          for ((i=1; i<=folds; i++)); do
            fold=$i
            echo "Running $dir/main.py with --data_name=$data_name --amazon_cat=$amazon_cat --mode=$mode --lr=$lr --fold=$fold (Run $i) --num_layers=$num_layers"
            (cd "$dir" && CUDA_VISIBLE_DEVICES=2 python main.py equivset --train --cuda --data_name="$data_name" --amazon_cat="$amazon_cat" --mode="$mode" --lr="$lr" --fold=$fold --num_layers=$num_layers)
          done
        done
      done
    done
  else
    # Loop over each mode
    for mode in "${modes[@]}"; do
      for lr in "${learning_rates[@]}"; do
        for ((i=1; i<=folds; i++)); do
          fold=$i
          echo "Running $dir/main.py with --data_name=$data_name --mode=$mode --lr=$lr --fold=$fold (Run $i) --num_layers=$num_layers"
          (cd "$dir" && CUDA_VISIBLE_DEVICES=2 python main.py equivset --train --cuda --data_name="$data_name" --mode="$mode" --lr="$lr" --fold=$fold --num_layers=$num_layers)
        done
      done
    done
  fi
done