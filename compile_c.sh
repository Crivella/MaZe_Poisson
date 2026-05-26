#!/bin/bash

# how to run this: ./compile_c.sh   
# Print a message indicating the start of the build process
echo "Building the shared library..."

# Execute the GCC command to build the shared library
/opt/homebrew/Cellar/gcc/14.2.0_1/bin/gcc-14 -O3 --fast-math -shared laplace.c -o maze_poisson/libmaze_poisson.so -fopenmp -fPIC

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! The shared library has been created at 'maze_poisson/libmaze_poisson.so'."
else
    echo "Build failed. Please check the command and your environment."
fi
