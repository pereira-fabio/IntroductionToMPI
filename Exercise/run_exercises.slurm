#!/bin/bash -l
#SBATCH --job-name=MPI_exercises
#SBATCH --output=mpi_exercises_%j.out
#SBATCH --error=mpi_exercises_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=batch

# Load required modules for AION cluster (2023b release)
module purge
module load toolchain/foss/2023b
module load lib/FlexiBLAS/3.3.1-GCC-13.2.0

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "=========================================="

# Set working directory
cd $SLURM_SUBMIT_DIR
BUILD_DIR="../build/bin"

echo ""
echo "=========================================="
echo "Exercise 1.1: DAXPY and DDOT"
echo "=========================================="
srun --ntasks=1 ${BUILD_DIR}/exercise_1_1

echo ""
echo "=========================================="
echo "Exercise 1.2: DGEMV (column-wise vs row-wise)"
echo "=========================================="
srun --ntasks=1 ${BUILD_DIR}/exercise_1_2

echo ""
echo "=========================================="
echo "Exercise 1.3: DGEMM (column-wise vs row-wise)"
echo "=========================================="
srun --ntasks=1 ${BUILD_DIR}/exercise_1_3

echo ""
echo "=========================================="
echo "Exercise 2.1: Parallel GEMM (Static Scattering)"
echo "Running with 1, 2, 4, 8, 16, 32, 64, 128 processes"
echo "=========================================="

for np in 1 2 4 8 16 32 64 128; do
    echo ""
    echo "--- Running with $np MPI processes ---"
    srun --ntasks=$np ${BUILD_DIR}/exercise_2_1
done

echo ""
echo "=========================================="
echo "Exercise 2.2: Parallel GEMM (Dynamic Distribution)"
echo "Running with 1, 2, 4, 8, 16, 32, 64, 128 processes"
echo "=========================================="

for np in 1 2 4 8 16 32 64 128; do
    echo ""
    echo "--- Running with $np MPI processes ---"
    srun --ntasks=$np ${BUILD_DIR}/exercise_2_2
done

echo ""
echo "=========================================="
echo "All exercises completed"
echo "End time: $(date)"
echo "=========================================="
