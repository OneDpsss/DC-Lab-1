#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double sequential_multiply(int N, double *A, double *B, double *C) {
    // Initialize C to zero
    for (int i = 0; i < N * N; ++i)
        C[i] = 0.0;

    double t0 = MPI_Wtime();
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
    double t1 = MPI_Wtime();
    return t1 - t0;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if number of processes is a perfect square
    int p = (int)sqrt((double)size);
    if (p * p != size) {
        if (rank == 0)
            fprintf(stderr, "Error: number of processes (%d) must be a perfect square.\n", size);
        MPI_Finalize();
        return 1;
    }

    // Parse matrix size N
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    if (N % p != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N=%d is not divisible by p=%d (block size must be integer).\n", N, p);
        MPI_Finalize();
        return 1;
    }

    int n = N / p; // local block dimension

    // Allocate local blocks
    double *A_block = (double *)malloc(n * n * sizeof(double));
    double *B_block = (double *)malloc(n * n * sizeof(double));
    double *C_block = (double *)calloc(n * n, sizeof(double));

    if (!A_block || !B_block || !C_block) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize full matrices on rank 0
    double *A_full = NULL, *B_full = NULL;
    double T_seq = 0.0;

    if (rank == 0) {
        A_full = (double *)malloc(N * N * sizeof(double));
        B_full = (double *)malloc(N * N * sizeof(double));
        if (!A_full || !B_full) {
            fprintf(stderr, "Rank 0: failed to allocate full matrices.\n");
            free(A_full);
            free(B_full);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N * N; ++i) {
            A_full[i] = (double)(i % 10); // small values to avoid overflow
            B_full[i] = (double)((i + 1) % 10);
        }
        double *C_seq = (double *)malloc(N * N * sizeof(double));
        T_seq = sequential_multiply(N, A_full, B_full, C_seq);
        free(C_seq);
    }

    // Distribute blocks
    MPI_Scatter(A_full, n * n, MPI_DOUBLE, A_block, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B_full, n * n, MPI_DOUBLE, B_block, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(A_full);
        free(B_full);
    }

    // 2D grid coordinates
    int row = rank / p;
    int col = rank % p;

    // Initial shift for A: left by 'row' positions
    int shift_a = (-row + p) % p;
    for (int s = 0; s < shift_a; ++s) {
        int left_rank = row * p + (col - 1 + p) % p;
        int right_rank = row * p + (col + 1) % p;
        MPI_Sendrecv_replace(A_block, n * n, MPI_DOUBLE,
                             right_rank, 0,
                             left_rank, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Initial shift for B: up by 'col' positions
    int shift_b = (-col + p) % p;
    for (int s = 0; s < shift_b; ++s) {
        int up_rank = ((row - 1 + p) % p) * p + col;
        int down_rank = ((row + 1) % p) * p + col;
        MPI_Sendrecv_replace(B_block, n * n, MPI_DOUBLE,
                             down_rank, 1,
                             up_rank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Main Cannon loop
    double t_start = MPI_Wtime();

    for (int step = 0; step < p; ++step) {
        // Local block multiplication: C += A * B
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    C_block[i * n + j] += A_block[i * n + k] * B_block[k * n + j];

        // Shift A left
        int left_rank = row * p + (col - 1 + p) % p;
        int right_rank = row * p + (col + 1) % p;
        MPI_Sendrecv_replace(A_block, n * n, MPI_DOUBLE,
                             right_rank, 2,
                             left_rank, 2,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Shift B up
        int up_rank = ((row - 1 + p) % p) * p + col;
        int down_rank = ((row + 1) % p) * p + col;
        MPI_Sendrecv_replace(B_block, n * n, MPI_DOUBLE,
                             down_rank, 3,
                             up_rank, 3,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    // Get max execution time across all processes
    double T_par;
    MPI_Reduce(&local_time, &T_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double speedup = T_seq / T_par;
        double efficiency = speedup / size;
        // CSV format: N, P, T_par, T_seq, Speedup, Efficiency
        // printf("N, P, T_par, T_seq, Speedup, Efficiency\n");
        printf("%d,%d,%.6f,%.6f,%.4f,%.4f\n", N, size, T_par, T_seq, speedup, efficiency);
    }

    // Cleanup
    free(A_block);
    free(B_block);
    free(C_block);

    MPI_Finalize();
    return 0;
}