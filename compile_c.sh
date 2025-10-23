#!/bin/bash

echo "Building the shared library..."

# Check if source file exists
if [ ! -f "poisson_solver.c" ]; then
    echo "Error: poisson_solver.c not found in current directory"
    echo "Available C files:"
    ls *.c 2>/dev/null || echo "No C files found"
    exit 1
fi

# Use system GCC with correct syntax
gcc -O3 -ffast-math -fopenmp -fPIC -shared poisson_solver.c -o libmaze_poisson.so

if [ $? -eq 0 ]; then
    echo "Build successful! The shared library has been created as 'libmaze_poisson.so'."
else
    echo "Build failed. Please check the command and your environment."
    exit 1
fi