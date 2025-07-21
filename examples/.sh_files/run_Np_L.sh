#!/bin/bash

# Set the value for N_steps
N_steps=400

# Define the arrays of particle numbers, corresponding L values, and N values
Np_array=(2 16 54 128 250 432 686 1024)
L_array=(4.18 8.36 12.54 16.72 19.6590 25.08 29.26 33.44)
N_array=(20 40 60 80 100 120 140 160)

# Ensure all arrays have the same length
if [ ${#Np_array[@]} -ne ${#L_array[@]} ] || [ ${#Np_array[@]} -ne ${#N_array[@]} ]; then
    echo "Error: Np_array, L_array, and N_array must have the same length"
    exit 1
fi

# Iterate over the arrays
for index in "${!Np_array[@]}"; do
    Np=${Np_array[$index]}
    L=${L_array[$index]}
    N=${N_array[$index]}
    
    echo "Running python main_MaZe.py for Np = $Np, N = $N, N_steps = $N_steps, and L = $L"
    python main_Maze_md.py $Np $N $N_steps $L > ../data/dati_paper/plot_Np/output_Np$Np.out
    
    echo "Iteration for Np = $Np finished"
done

echo "All iterations completed"
