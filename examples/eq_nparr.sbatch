#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=2:00:00
#SBATCH --mem=490GB
#SBATCH --account=cecam_premium
#SBATCH --output="log_temp_%A_%a.out"  # Log output for each array task
#SBATCH --mail-user=jayashree.narayan@epfl.ch
#SBATCH --mail-type=END
#SBATCH --array=0-5  # Array index from 0 to 6 (7 different jobs)

# Define arrays for OMP_NUM_THREADS and N values
NP_ARRAY=(128 250 432 686 1024 1458)

# Use the job array index to select the corresponding values
OMP_NUM_THREADS=32
NP=${NP_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Print job information
echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "SLURM job array task ID: $SLURM_ARRAY_TASK_ID"
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"
echo "Np set to: $NP"

# Dynamically set the log file name based on threads
LOG_FILE="log_Np${NP}_job${SLURM_JOB_ID}.out"
exec >"$LOG_FILE" 2>&1

echo "Output is being redirected to $LOG_FILE"

# Load necessary modules
module load gcc libxc openmpi fftw openblas netlib-scalapack || { echo "Failed to load modules"; exit 1; }

# Compile the shared library
echo "Compiling the shared library..."
gcc -O3 --fast-math -shared ../laplace.c -o ../maze_poisson/libmaze_poisson.so -fopenmp -fPIC || { echo "Compilation failed"; exit 1; }
echo "Compilation completed successfully."

# Activate Python virtual environment
echo "Activating Python virtual environment..."
source ../venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Export the variables to Python
export NP

# Run the code
echo "Running the maze application..."
maze run main-maze-md test_input_density_1.379.yml || { echo "Maze execution failed"; exit 1; }
echo "Maze application completed successfully."

# Deactivate the virtual environment
deactivate

echo "Job ended at $(date)"
