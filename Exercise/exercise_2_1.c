#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <mpi.h>

#include "array.h"
#include "multiply.h"

static unsigned int const seed = 1234;
static int const dimensions[] = {128*1, 128*2, 128*4, 128*8};
static int const n_dimensions = sizeof(dimensions)/sizeof(int);
static double const epsilon = 1e-10;

typedef void (*GEMM)(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C
);

static void populate_compatible_random_matrix_pairs(
    int const m, int const k, int const n,
    int const seed,
    double* const A, double* const B)
{
    set_initilize_rand_seed(seed);

    initialize_2d_double_blocked_rand(A, m, k);
    initialize_2d_double_blocked_rand(B, k, n);
}

static void initialize_problem_matrices(
    int const m, int const k, int const n,
    double** const A, double** const B, double** const C)
{
    *A = allocate_2d_double_blocked(m, k);
    *B = allocate_2d_double_blocked(k, n);
    *C = allocate_2d_double_blocked(m, n);
}

static void destroy_problem_matrices(double** const A, double** const B, double** const C)
{
    *A = free_2d_double_blocked(*A);
    *B = free_2d_double_blocked(*B);
    *C = free_2d_double_blocked(*C);
}

static bool test_muptiply(int const m, int const k, int const n, GEMM gemm, double const epsilon, unsigned int const seed)
{
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    initialize_problem_matrices(m, k, n, &A, &B, &C);
    populate_compatible_random_matrix_pairs(m, k, n, seed, A, B);

    gemm(m, k, n, A, B, C);
    bool result_is_correct = is_product(m, k, n, A, B, C, epsilon);

    destroy_problem_matrices(&A, &B, &C);

    return result_is_correct;
}

// Implementation of parallel matrix multiplication using static scattering
// of a single operand (matrix B columns are distributed across processes).
// C = A * B where B is scattered column-wise and C is gathered.
void parallel_gemm(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate local columns for each process
    int const cols_per_proc = n / size;
    int const local_n = (rank < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
    
    // Allocate local storage for B columns and C columns
    double* local_B = allocate_2d_double_blocked(k, local_n);
    double* local_C = allocate_2d_double_blocked(m, local_n);
    
    // Scatter columns of B to all processes
    // Each process receives cols_per_proc columns of B
    if (rank == 0) {
        // Root process: distribute B columns
        for (int p = 1; p < size; p++) {
            int const p_local_n = (p < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
            int const start_col = p * cols_per_proc;
            // Send columns of B (contiguous in column-major order)
            MPI_Send(&B[start_col * k], k * p_local_n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
        // Copy local B columns for rank 0
        for (int j = 0; j < local_n; j++) {
            for (int i = 0; i < k; i++) {
                local_B[i + j * k] = B[i + j * k];
            }
        }
    } else {
        // Non-root processes: receive B columns
        MPI_Recv(local_B, k * local_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Broadcast matrix A to all processes (all processes need full A)
    double* local_A = allocate_2d_double_blocked(m, k);
    if (rank == 0) {
        for (int i = 0; i < m * k; i++) {
            local_A[i] = A[i];
        }
    }
    MPI_Bcast(local_A, m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Each process computes its portion: local_C = local_A * local_B
    // Initialize local_C to zero
    for (int i = 0; i < m * local_n; i++) {
        local_C[i] = 0.0;
    }
    
    // Matrix multiplication: C = A * B (column-wise access pattern)
    for (int j = 0; j < local_n; j++) {
        for (int l = 0; l < k; l++) {
            double const b_lj = local_B[l + j * k];
            for (int i = 0; i < m; i++) {
                local_C[i + j * m] += local_A[i + l * m] * b_lj;
            }
        }
    }
    
    // Gather results back to root process
    if (rank == 0) {
        // Copy local C columns for rank 0
        for (int j = 0; j < local_n; j++) {
            for (int i = 0; i < m; i++) {
                C[i + j * m] = local_C[i + j * m];
            }
        }
        // Receive C columns from other processes
        for (int p = 1; p < size; p++) {
            int const p_local_n = (p < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
            int const start_col = p * cols_per_proc;
            MPI_Recv(&C[start_col * m], m * p_local_n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // Non-root processes: send local C columns
        MPI_Send(local_C, m * local_n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    // Free local storage
    local_A = free_2d_double_blocked(local_A);
    local_B = free_2d_double_blocked(local_B);
    local_C = free_2d_double_blocked(local_C);
}

// Then set "tested_gemm" to the address of your funtion
// GEMM const tested_gemm = &multiply_matrices;
GEMM const tested_gemm = &parallel_gemm;

static bool generate_square_matrix_dimension(int* const m, int* const k, int* const n)
{
    int const max_dim = n_dimensions;
    static int dim = 0;

    if (dim >= max_dim) {
        return false;
    }

    *m = dimensions[dim];
    *k = dimensions[dim];
    *n = dimensions[dim];
    
    dim++;

    return true;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    bool all_test_pass = true;

    int m = 0;
    int k = 0;
    int n = 0;

    while (generate_square_matrix_dimension(&m, &k, &n)) {
        bool const test_pass = test_muptiply(m, k, n, tested_gemm, epsilon, seed);
        if (!test_pass && rank == 0) {
            printf("Multiplication failed for: m=%d, k=%d, n=%d\n", m, k, n);
            all_test_pass = false;
        }
    }

    MPI_Finalize();
    
    if (!all_test_pass) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
