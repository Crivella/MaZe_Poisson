#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=2:00:00
#SBATCH --mem=490GB
#SBATCH --account=cecam_premium
#SBATCH --output="log_temp_%j.out"  # Temporary log file
#SBATCH --mail-user=jayashree.narayan@epfl.ch
#SBATCH --mail-type=END

# Print job information
echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"

# Dynamically set the log file name based on threads
export OMP_NUM_THREADS=32
LOG_FILE="log_threads${OMP_NUM_THREADS}_job${SLURM_JOB_ID}.out"
exec >"$LOG_FILE" 2>&1

echo "Output is being redirected to $LOG_FILE"
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"

# Load necessary modules
module load gcc libxc openmpi fftw openblas netlib-scalapack || { echo "Failed to load modules"; exit 1; }

# Compile the shared library
echo "Compiling the shared library..."
gcc -O3 --fast-math -shared ../laplace.c -o ../maze_poisson/libmaze_poisson.so -fopenmp -fPIC || { echo "Compilation failed"; exit 1; }
echo "Compilation completed successfully."

# Activate Python virtual environment
echo "Activating Python virtual environment..."
source ../venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Run the code
echo "Running the maze application..."
maze run main-maze-md test_input_density_1.379.yml || { echo "Maze execution failed"; exit 1; }
echo "Maze application completed successfully."

# Deactivate the virtual environment
deactivate

echo "Job ended at $(date)"