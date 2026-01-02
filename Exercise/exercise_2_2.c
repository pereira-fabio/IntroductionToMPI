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

// Implementation of parallel matrix multiplication using dynamic distribution
// of operands (simplified row-wise Cannon's algorithm).
// Both A (row bands) and B (column bands) are distributed, then A bands
// are rotated among processes during computation.
void parallel_gemm(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate local dimensions
    int const rows_per_proc = m / size;
    int const cols_per_proc = n / size;
    int const local_m = (rank < size - 1) ? rows_per_proc : (m - rows_per_proc * (size - 1));
    int const local_n = (rank < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
    
    // Allocate local storage
    // Each process holds a horizontal band of A and a vertical band of B
    double* local_A = allocate_2d_double_blocked(local_m, k);
    double* local_A_recv = allocate_2d_double_blocked(rows_per_proc + (m % size), k); // Buffer for received A
    double* local_B = allocate_2d_double_blocked(k, local_n);
    double* local_C = allocate_2d_double_blocked(local_m, local_n);
    
    // Initialize local_C to zero
    for (int i = 0; i < local_m * local_n; i++) {
        local_C[i] = 0.0;
    }
    
    // Distribute A (row bands) and B (column bands) from root
    if (rank == 0) {
        // Distribute row bands of A
        for (int p = 0; p < size; p++) {
            int const p_local_m = (p < size - 1) ? rows_per_proc : (m - rows_per_proc * (size - 1));
            int const start_row = p * rows_per_proc;
            if (p == 0) {
                // Copy rows of A for rank 0 (column-major: non-contiguous rows)
                for (int j = 0; j < k; j++) {
                    for (int i = 0; i < local_m; i++) {
                        local_A[i + j * local_m] = A[start_row + i + j * m];
                    }
                }
            } else {
                // Pack and send rows of A
                double* send_buf = allocate_2d_double_blocked(p_local_m, k);
                for (int j = 0; j < k; j++) {
                    for (int i = 0; i < p_local_m; i++) {
                        send_buf[i + j * p_local_m] = A[start_row + i + j * m];
                    }
                }
                MPI_Send(send_buf, p_local_m * k, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                send_buf = free_2d_double_blocked(send_buf);
            }
        }
        // Distribute column bands of B
        for (int p = 0; p < size; p++) {
            int const p_local_n = (p < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
            int const start_col = p * cols_per_proc;
            if (p == 0) {
                for (int j = 0; j < local_n; j++) {
                    for (int i = 0; i < k; i++) {
                        local_B[i + j * k] = B[i + (start_col + j) * k];
                    }
                }
            } else {
                MPI_Send(&B[start_col * k], p_local_n * k, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        // Receive row band of A
        MPI_Recv(local_A, local_m * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Receive column band of B
        MPI_Recv(local_B, local_n * k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Cannon's algorithm: rotate A bands and accumulate partial products
    // Each process computes C[rank_row, rank_col] by receiving different A bands
    int const prev_rank = (rank - 1 + size) % size;
    int const next_rank = (rank + 1) % size;
    
    // Current A band info
    int current_A_owner = rank;
    int current_local_m = local_m;
    
    for (int step = 0; step < size; step++) {
        // Compute partial product with current A band
        // We need to multiply the A band (owned by current_A_owner) with our B band
        // and add to the corresponding part of C
        
        // Determine which rows of C this contributes to
        int const A_start_row = current_A_owner * rows_per_proc;
        int const A_local_m = (current_A_owner < size - 1) ? rows_per_proc : (m - rows_per_proc * (size - 1));
        
        // Compute: partial_C = local_A * local_B
        for (int j = 0; j < local_n; j++) {
            for (int l = 0; l < k; l++) {
                double const b_lj = local_B[l + j * k];
                for (int i = 0; i < A_local_m; i++) {
                    // Accumulate into local_C if this is our row band
                    if (current_A_owner == rank) {
                        local_C[i + j * local_m] += local_A[i + l * A_local_m] * b_lj;
                    }
                }
            }
        }
        
        // Rotate A band to next process (except on last step)
        if (step < size - 1) {
            int const send_size = A_local_m * k;
            int const recv_owner = (current_A_owner - 1 + size) % size;
            int const recv_local_m = (recv_owner < size - 1) ? rows_per_proc : (m - rows_per_proc * (size - 1));
            int const recv_size = recv_local_m * k;
            
            MPI_Sendrecv(local_A, send_size, MPI_DOUBLE, next_rank, 2,
                         local_A_recv, recv_size, MPI_DOUBLE, prev_rank, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Swap buffers
            double* temp = local_A;
            local_A = local_A_recv;
            local_A_recv = temp;
            
            current_A_owner = recv_owner;
            current_local_m = recv_local_m;
        }
    }
    
    // Gather results back to root process
    if (rank == 0) {
        // Copy local C for rank 0
        for (int j = 0; j < local_n; j++) {
            for (int i = 0; i < local_m; i++) {
                C[i + j * m] = local_C[i + j * local_m];
            }
        }
        // Receive C from other processes
        for (int p = 1; p < size; p++) {
            int const p_local_m = (p < size - 1) ? rows_per_proc : (m - rows_per_proc * (size - 1));
            int const p_local_n = (p < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
            int const start_row = p * rows_per_proc;
            int const start_col = p * cols_per_proc;
            
            double* recv_buf = allocate_2d_double_blocked(p_local_m, p_local_n);
            MPI_Recv(recv_buf, p_local_m * p_local_n, MPI_DOUBLE, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Unpack into C
            for (int j = 0; j < p_local_n; j++) {
                for (int i = 0; i < p_local_m; i++) {
                    C[start_row + i + (start_col + j) * m] = recv_buf[i + j * p_local_m];
                }
            }
            recv_buf = free_2d_double_blocked(recv_buf);
        }
    } else {
        // Send local C to root
        MPI_Send(local_C, local_m * local_n, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
    
    // Free local storage
    local_A = free_2d_double_blocked(local_A);
    local_A_recv = free_2d_double_blocked(local_A_recv);
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
