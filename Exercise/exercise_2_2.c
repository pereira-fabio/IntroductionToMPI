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
// A is distributed in row bands, B is distributed in column bands.
// Each process computes a full row of C blocks by rotating its A band
// and receiving B bands from other processes.
void parallel_gemm(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // For simplicity, we use a 1D decomposition where each process gets
    // a band of rows from A and computes the corresponding rows of C
    int const rows_per_proc = m / size;
    int const local_m = (rank < size - 1) ? rows_per_proc : (m - rows_per_proc * (size - 1));
    int const start_row = rank * rows_per_proc;
    
    // Each process needs: local rows of A, and will receive bands of B
    // B is divided into column bands for rotation
    int const cols_per_proc = n / size;
    
    // Allocate storage
    double* local_A = allocate_2d_double_blocked(local_m, k);  // My rows of A
    double* local_B = allocate_2d_double_blocked(k, cols_per_proc + (n % size));  // Current B band
    double* local_B_recv = allocate_2d_double_blocked(k, cols_per_proc + (n % size));  // Buffer for receiving B
    double* local_C = allocate_2d_double_blocked(local_m, n);  // My rows of C (full width)
    
    // Initialize local_C to zero
    for (int i = 0; i < local_m * n; i++) {
        local_C[i] = 0.0;
    }
    
    // Distribute rows of A from root
    if (rank == 0) {
        // Copy my rows of A
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < local_m; i++) {
                local_A[i + j * local_m] = A[i + j * m];
            }
        }
        // Send rows to other processes
        for (int p = 1; p < size; p++) {
            int const p_local_m = (p < size - 1) ? rows_per_proc : (m - rows_per_proc * (size - 1));
            int const p_start_row = p * rows_per_proc;
            double* send_buf = allocate_2d_double_blocked(p_local_m, k);
            for (int j = 0; j < k; j++) {
                for (int i = 0; i < p_local_m; i++) {
                    send_buf[i + j * p_local_m] = A[p_start_row + i + j * m];
                }
            }
            MPI_Send(send_buf, p_local_m * k, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            send_buf = free_2d_double_blocked(send_buf);
        }
    } else {
        MPI_Recv(local_A, local_m * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Distribute initial B bands - each process p gets column band p
    if (rank == 0) {
        // Copy my B band (columns 0 to cols_per_proc-1)
        int const my_local_n = cols_per_proc;
        for (int j = 0; j < my_local_n; j++) {
            for (int i = 0; i < k; i++) {
                local_B[i + j * k] = B[i + j * k];
            }
        }
        // Send B bands to other processes
        for (int p = 1; p < size; p++) {
            int const p_local_n = (p < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
            int const p_start_col = p * cols_per_proc;
            MPI_Send(&B[p_start_col * k], k * p_local_n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
    } else {
        int const my_local_n = (rank < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
        MPI_Recv(local_B, k * my_local_n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Ring communication for rotating B bands
    int const prev_rank = (rank - 1 + size) % size;
    int const next_rank = (rank + 1) % size;
    
    // Current B band owner (determines which columns of C we're computing)
    int current_B_owner = rank;
    
    for (int step = 0; step < size; step++) {
        // Compute partial product: local_C[:, B_cols] += local_A * local_B
        int const B_start_col = current_B_owner * cols_per_proc;
        int const B_local_n = (current_B_owner < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
        
        // Matrix multiplication: C[my_rows, B_cols] = A[my_rows, :] * B[:, B_cols]
        for (int j = 0; j < B_local_n; j++) {
            for (int l = 0; l < k; l++) {
                double const b_lj = local_B[l + j * k];
                for (int i = 0; i < local_m; i++) {
                    local_C[i + (B_start_col + j) * local_m] += local_A[i + l * local_m] * b_lj;
                }
            }
        }
        
        // Rotate B band to next process (except on last step)
        if (step < size - 1) {
            int const send_local_n = B_local_n;
            int const next_B_owner = (current_B_owner - 1 + size) % size;
            int const recv_local_n = (next_B_owner < size - 1) ? cols_per_proc : (n - cols_per_proc * (size - 1));
            
            MPI_Sendrecv(local_B, k * send_local_n, MPI_DOUBLE, next_rank, 2,
                         local_B_recv, k * recv_local_n, MPI_DOUBLE, prev_rank, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Swap buffers
            double* temp = local_B;
            local_B = local_B_recv;
            local_B_recv = temp;
            
            current_B_owner = next_B_owner;
        }
    }
    
    // Gather results back to root process
    if (rank == 0) {
        // Copy my rows of C
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < local_m; i++) {
                C[i + j * m] = local_C[i + j * local_m];
            }
        }
        // Receive rows from other processes
        for (int p = 1; p < size; p++) {
            int const p_local_m = (p < size - 1) ? rows_per_proc : (m - rows_per_proc * (size - 1));
            int const p_start_row = p * rows_per_proc;
            double* recv_buf = allocate_2d_double_blocked(p_local_m, n);
            MPI_Recv(recv_buf, p_local_m * n, MPI_DOUBLE, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < p_local_m; i++) {
                    C[p_start_row + i + j * m] = recv_buf[i + j * p_local_m];
                }
            }
            recv_buf = free_2d_double_blocked(recv_buf);
        }
    } else {
        MPI_Send(local_C, local_m * n, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
    
    // Free local storage
    local_A = free_2d_double_blocked(local_A);
    local_B = free_2d_double_blocked(local_B);
    local_B_recv = free_2d_double_blocked(local_B_recv);
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
