#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=30G
#SBATCH --account=cecam_premium
#SBATCH --output="log_temp_%j.out"  # Temporary log file
#SBATCH --mail-user=jayashree.narayan@epfl.ch
#SBATCH --mail-type=END

# Print job information
echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"

echo "Output is being redirected to $LOG_FILE"
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"

# Load necessary modules
module load gcc libxc openmpi fftw openblas netlib-scalapack || { echo "Failed to load modules"; exit 1; }

# Activate Python virtual environment
echo "Activating Python virtual environment..."
source ../venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Run the code
echo "Running the maze application..."
maze plot weak-scaling-vs-thread || { echo "Plot execution failed"; exit 1; }
echo "Plot application completed successfully."

# Deactivate the virtual environment
deactivate

echo "Job ended at $(date)"
