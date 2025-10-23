#!/bin/bash
echo "Building the shared library..."

# Check if laplace.c exists (this is your actual file)
if [ -f "laplace.c" ]; then
    echo "Found laplace.c - using this file to build the library"
    gcc -O3 -ffast-math -fopenmp -fPIC -shared laplace.c -o libmaze_poisson.so
else
    echo "Error: laplace.c not found"
    echo "Available C files:"
    find . -name "*.c" -type f
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "Build successful! The shared library has been created as 'libmaze_poisson.so'."
else
    echo "Build failed. Please check the command and your environment."
    exit 1
fi
