#!/bin/bash

# Define the parameters
learning_rates=("0.00001" "0.0001" "0.001" "0.01")
data_names=("celeba") #"moons"  "amazon" "gaussian")
modes=("ind") # "diffMF" "copula" )

cd ./EquiVSet
# Loop over each combination of parameters
for data_name in "${data_names[@]}"; do
    for mode in "${modes[@]}"; do
        for lr in "${learning_rates[@]}"; do
            command="python main.py equivset --train --cuda --data_name ${data_name} --mode ${mode} --lr ${lr}"
            echo "Executing: $command"
            $command
        done
    done
done