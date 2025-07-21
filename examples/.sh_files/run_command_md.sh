#!/bin/bash

# Set the values for N and stride (s)
N_max=201
N_min=50
N_steps=5000
N_p=2
s=50

for ((i=N_min; i<=N_max; i+=s)); do
    echo "Running python main_MaZe_md.py for N = $i and N_steps = $N_steps"
    python main_Maze_md.py  $N_p $i $N_steps > new_data/test_asymptotic/output_N$i.out
    
    echo "Iteration $i finished"
done

echo "All iterations completed"
