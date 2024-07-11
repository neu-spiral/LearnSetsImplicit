#!/bin/bash

# Directory
dir="ImplicitEquiVSetFlax"

# Possible values for data_name
data_names=("amazon") # "celeba") #"moons" "gaussian" )

# Possible values for amazon_cat
amazon_cats=("bedding" "carseats" "diaper" "feeding" "gear" "media" "bath" "health" "toys" "furniture" "safety" ) #"apparel"

# Learning rates
learning_rates=("0.01" "0.001" "0.0001" "0.00001") #

# Number of repetitions
folds=5
fwd_tol=1e-6
fwd_maxiter=20
num_layers=2


# Loop over each data_name
for data_name in "${data_names[@]}"; do
  if [ "$data_name" == "amazon" ]; then
    # Loop over each amazon_cat
    for amazon_cat in "${amazon_cats[@]}"; do
      for lr in "${learning_rates[@]}"; do
        for ((i=1; i<=folds; i++)); do
          fold=$i
          echo "Running $dir/main_flax.py with --data_name=$data_name --amazon_cat=$amazon_cat --lr=$lr --fold=$fold (Run $i) --fwd_tol=$fwd_tol --fwd_maxiter=$fwd_maxiter --num_layers=$num_layers"
          (cd "$dir" && CUDA_VISIBLE_DEVICES=0 python main_flax.py --data_name="$data_name" --amazon_cat="$amazon_cat" --lr="$lr" --fold=$fold --fwd_tol=$fwd_tol --fwd_maxiter=$fwd_maxiter --num_layers=$num_layers)
        done
      done
    done
  else
    for lr in "${learning_rates[@]}"; do
      for ((i=1; i<=folds; i++)); do
        fold=$i
        echo "Running $dir/main_flax.py with --data_name=$data_name --lr=$lr --fold=$fold (Run $i) --fwd_tol=$fwd_tol --fwd_maxiter=$fwd_maxiter --num_layers=$num_layers"
        (cd "$dir" && CUDA_VISIBLE_DEVICES=0 python main_flax.py --data_name="$data_name" --lr="$lr" --fold=$fold --fwd_tol=$fwd_tol --fwd_maxiter=$fwd_maxiter --num_layers=$num_layers)
      done
    done
  fi
done