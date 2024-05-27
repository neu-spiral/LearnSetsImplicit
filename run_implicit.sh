#!/bin/bash

# Directory
dir="ImplicitEquiVSetFlax"

# Possible values for data_name
data_names=("amazon" "moons" "gaussian" "celeba" )

# Possible values for amazon_cat
amazon_cats=("toys" "furniture" "gear" "carseats" "bath" "health" "diaper" "bedding" "safety" "feeding" "apparel" "media")

# Number of repetitions
repetitions=5
fwd_tol=1e-6
fwd_maxiter=20

# Loop over each data_name
for data_name in "${data_names[@]}"; do
  if [ "$data_name" == "amazon" ]; then
    # Loop over each amazon_cat
    for amazon_cat in "${amazon_cats[@]}"; do
      for ((i=1; i<=repetitions; i++)); do
        # Generate a random seed
        seed=i
        echo "Running $dir/main_flax.py with --data_name=$data_name --amazon_cat=$amazon_cat --seed=$seed (Run $i) --fwd_tol=$fwd_tol --fwd_maxiter=$fwd_maxiter"
        (cd "$dir" && python main_flax.py --data_name="$data_name" --amazon_cat="$amazon_cat" --seed=$seed --fwd_tol=$fwd_tol --fwd_maxiter=$fwd_maxiter)
      done
    done
  else
    for ((i=1; i<=repetitions; i++)); do
      # Generate a random seed
      seed=i
      echo "Running $dir/main_flax.py with --data_name=$data_name --seed=$seed (Run $i) --fwd_tol=$fwd_tol --fwd_maxiter=$fwd_maxiter"
      (cd "$dir" && python main_flax.py --data_name="$data_name" --seed=$seed --fwd_tol=$fwd_tol --fwd_maxiter=$fwd_maxiter)
    done
  fi
done
