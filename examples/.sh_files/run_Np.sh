#!/bin/bash

# Set the values for N and stride (s)
Np_max=64
Np_min=2
N_steps=10
N=100
s=2

for ((i=Np_min; i<=Np_max; i*=s)); do
    echo "Running python main_MaZe.py for Np = $i and N_steps = $N_steps"
    python main_Maze.py  $i $N $N_steps > data/scaling_N_p_random/N_p_$i/output_Np$i.out
    
    echo "Iteration $i finished"
done

echo "All iterations completed"
