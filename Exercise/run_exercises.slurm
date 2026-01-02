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

# Build the exercises first
echo ""
echo "=========================================="
echo "Building exercises..."
echo "=========================================="
cd ../Examples

# Clean and build libraries (this also installs headers)
make clean
make libraries

# Manually install headers since make libraries doesn't do it
mkdir -p build/include
cp utils/array.h build/include/
cp utils/multiply.h build/include/

# Verify libraries and headers exist
if [ ! -f "build/lib/libarray.a" ] || [ ! -f "build/include/array.h" ]; then
    echo "ERROR: Libraries or headers not built correctly"
    ls -la build/lib/ 2>/dev/null || echo "lib dir missing"
    ls -la build/include/ 2>/dev/null || echo "include dir missing"
    exit 1
fi

# Build exercise files manually with proper includes
EXERCISES_SRC="../Exercise"
INCLUDE_DIR="build/include"
LIB_DIR="build/lib"
BIN_DIR="build/bin"
OBJ_DIR="build/obj/exercises"

mkdir -p ${OBJ_DIR}
mkdir -p ${BIN_DIR}

# Get BLAS flags
BLAS_CFLAGS=$(pkg-config --cflags flexiblas)
BLAS_LIBS=$(pkg-config --libs flexiblas)

echo "BLAS_CFLAGS: ${BLAS_CFLAGS}"
echo "BLAS_LIBS: ${BLAS_LIBS}"
echo "INCLUDE_DIR: ${INCLUDE_DIR}"

# Compile BLAS exercises (1.1, 1.2, 1.3)
for ex in exercise_1_1 exercise_1_2 exercise_1_3; do
    echo "Compiling ${ex}..."
    gcc -c -O2 -Wall -I${INCLUDE_DIR} ${BLAS_CFLAGS} ${EXERCISES_SRC}/${ex}.c -o ${OBJ_DIR}/${ex}.o
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to compile ${ex}.c"
        exit 1
    fi
    gcc -O2 -Wall ${OBJ_DIR}/${ex}.o ${LIB_DIR}/libarray.a ${BLAS_LIBS} -lm -o ${BIN_DIR}/${ex}
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to link ${ex}"
        exit 1
    fi
done

# Compile MPI exercises (2.1, 2.2)
for ex in exercise_2_1 exercise_2_2; do
    echo "Compiling ${ex}..."
    mpicc -c -O2 -Wall -I${INCLUDE_DIR} ${EXERCISES_SRC}/${ex}.c -o ${OBJ_DIR}/${ex}.o
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to compile ${ex}.c"
        exit 1
    fi
    mpicc -O2 -Wall ${OBJ_DIR}/${ex}.o ${LIB_DIR}/libarray.a ${LIB_DIR}/libmultiply.a -lm -o ${BIN_DIR}/${ex}
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to link ${ex}"
        exit 1
    fi
done

echo "Build completed successfully."
cd ../Exercise

# Check if build was successful
if [ ! -f "${BUILD_DIR}/exercise_1_1" ]; then
    echo "ERROR: Build failed. Executables not found."
    exit 1
fi

echo "Build completed successfully."

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
