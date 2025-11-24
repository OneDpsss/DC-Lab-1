#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double sequential_multiply(int N, double *A, double *x, double *y) {
    for (int i = 0; i < N; ++i)
        y[i] = 0.0;

    double t0 = MPI_Wtime();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            y[i] += A[i * N + j] * x[j];
    double t1 = MPI_Wtime();
    return t1 - t0;
}

double multiply_rowwise(int N, double *A_full, double *x_full, double *y_full, int rank, int size) {
    int rows_per_proc = N / size;
    int remainder = N % size;

    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    double *A_local = (double *)malloc(local_rows * N * sizeof(double));
    double *y_local = (double *)calloc(local_rows, sizeof(double));

    if (!A_local || !y_local) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N;
            displs[i] = (i * rows_per_proc + (i < remainder ? i : remainder)) * N;
        }
    }

    MPI_Bcast(sendcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(A_full, sendcounts, displs, MPI_DOUBLE,
                 A_local, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(x_full, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t_start = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < N; ++j)
            y_local[i] += A_local[i * N + j] * x_full[j];

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *recvdispls = (int *)malloc(size * sizeof(int));

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = rows_per_proc + (i < remainder ? 1 : 0);
            recvdispls[i] = i * rows_per_proc + (i < remainder ? i : remainder);
        }
    }

    MPI_Bcast(recvcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(recvdispls, size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(y_local, local_rows, MPI_DOUBLE,
                y_full, recvcounts, recvdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);

    free(A_local);
    free(y_local);

    double T_par;
    MPI_Reduce(&local_time, &T_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return T_par;
}

double multiply_columnwise(int N, double *A_full, double *x_full, double *y_full, int rank, int size) {
    int cols_per_proc = N / size;
    int remainder = N % size;

    int local_cols = cols_per_proc + (rank < remainder ? 1 : 0);

    double *A_local = (double *)malloc(N * local_cols * sizeof(double));
    double *x_local = (double *)malloc(local_cols * sizeof(double));
    double *y_local = (double *)calloc(N, sizeof(double));

    if (!A_local || !x_local || !y_local) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        for (int proc = 0; proc < size; ++proc) {
            int local_cols_proc = cols_per_proc + (proc < remainder ? 1 : 0);
            int col_off = proc * cols_per_proc + (proc < remainder ? proc : remainder);
            if (proc == 0) {
                for (int j = 0; j < local_cols_proc; ++j) {
                    for (int i = 0; i < N; ++i)
                        A_local[j * N + i] = A_full[i * N + (col_off + j)];
                    x_local[j] = x_full[col_off + j];
                }
            } else {
                double *temp_A = (double *)malloc(N * local_cols_proc * sizeof(double));
                double *temp_x = (double *)malloc(local_cols_proc * sizeof(double));
                for (int j = 0; j < local_cols_proc; ++j) {
                    for (int i = 0; i < N; ++i)
                        temp_A[j * N + i] = A_full[i * N + (col_off + j)];
                    temp_x[j] = x_full[col_off + j];
                }
                MPI_Send(temp_A, N * local_cols_proc, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
                MPI_Send(temp_x, local_cols_proc, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
                free(temp_A);
                free(temp_x);
            }
        }
    } else {
        MPI_Recv(A_local, N * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(x_local, local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double t_start = MPI_Wtime();

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < local_cols; ++j)
            y_local[i] += A_local[j * N + i] * x_local[j];

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    MPI_Reduce(y_local, y_full, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    free(A_local);
    free(x_local);
    free(y_local);

    double T_par;
    MPI_Reduce(&local_time, &T_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return T_par;
}

double multiply_blockwise(int N, double *A_full, double *x_full, double *y_full, int rank, int size) {
    int p = (int)sqrt((double)size);
    if (p * p != size) {
        if (rank == 0)
            fprintf(stderr, "Error: number of processes (%d) must be a perfect square for block distribution.\n", size);
        return -1.0;
    }

    if (N % p != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N=%d is not divisible by p=%d.\n", N, p);
        return -1.0;
    }

    int n = N / p;
    int row = rank / p;
    int col = rank % p;

    int my_rank = rank;

    double *A_block = (double *)malloc(n * n * sizeof(double));
    double *x_block = (double *)malloc(n * sizeof(double));
    double *y_block = (double *)calloc(n, sizeof(double));

    if (!A_block || !x_block || !y_block) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        for (int proc = 0; proc < size; ++proc) {
            int proc_row = proc / p;
            int proc_col = proc % p;
            if (proc == 0) {
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        A_block[i * n + j] = A_full[(proc_row * n + i) * N + (proc_col * n + j)];
            } else {
                double *temp_block = (double *)malloc(n * n * sizeof(double));
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        temp_block[i * n + j] = A_full[(proc_row * n + i) * N + (proc_col * n + j)];
                MPI_Send(temp_block, n * n, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
                free(temp_block);
            }
        }

        for (int proc = 0; proc < size; ++proc) {
            int proc_col = proc % p;
            if (proc == 0) {
                for (int j = 0; j < n; ++j)
                    x_block[j] = x_full[proc_col * n + j];
            } else {
                double *temp_x = (double *)malloc(n * sizeof(double));
                for (int j = 0; j < n; ++j)
                    temp_x[j] = x_full[proc_col * n + j];
                MPI_Send(temp_x, n, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
                free(temp_x);
            }
        }
    } else {
        MPI_Recv(A_block, n * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(x_block, n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double t_start = MPI_Wtime();

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            y_block[i] += A_block[i * n + j] * x_block[j];

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    double *y_row = (double *)calloc(n, sizeof(double));
    MPI_Comm row_comm;
    int color = row;
    MPI_Comm_split(MPI_COMM_WORLD, color, my_rank, &row_comm);

    int row_rank_in_comm;
    MPI_Comm_rank(row_comm, &row_rank_in_comm);
    if (col == 0) {
        MPI_Reduce(y_block, y_row, n, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    } else {
        MPI_Reduce(y_block, NULL, n, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    }

    if (col == 0) {
        if (my_rank == 0) {
            for (int i = 0; i < n; ++i)
                y_full[i] = y_row[i];
            for (int r = 1; r < p; ++r) {
                MPI_Recv(y_full + r * n, n, MPI_DOUBLE, r * p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            MPI_Send(y_row, n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }
    }

    free(y_row);
    MPI_Comm_free(&row_comm);

    free(A_block);
    free(x_block);
    free(y_block);

    double T_par;
    MPI_Reduce(&local_time, &T_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return T_par;
}

int run_experiment(int N, int algo, int rank, int size) {
    double *A_full = NULL, *x_full = NULL, *y_full = NULL;
    double T_seq = 0.0;

    if (rank == 0) {
        A_full = (double *)malloc(N * N * sizeof(double));
        x_full = (double *)malloc(N * sizeof(double));
        y_full = (double *)malloc(N * sizeof(double));

        if (!A_full || !x_full || !y_full) {
            fprintf(stderr, "Rank 0: failed to allocate memory for N=%d.\n", N);
            free(A_full);
            free(x_full);
            free(y_full);
            return 1;
        }

        for (int i = 0; i < N * N; ++i)
            A_full[i] = (double)(i % 10);
        for (int i = 0; i < N; ++i)
            x_full[i] = (double)((i + 1) % 10);

        double *y_seq = (double *)malloc(N * sizeof(double));
        T_seq = sequential_multiply(N, A_full, x_full, y_seq);
        free(y_seq);
    }

    if (rank != 0) {
        x_full = (double *)malloc(N * sizeof(double));
    }

    double T_par = 0.0;
    const char *algo_name = "";

    if (rank != 0) {
        y_full = (double *)malloc(N * sizeof(double));
    }

    int valid = 1;
    if (algo == 3) {
        int p = (int)sqrt((double)size);
        if (p * p != size) {
            if (rank == 0)
                fprintf(stderr, "Skipping N=%d: P=%d is not a perfect square for block algorithm.\n", N, size);
            valid = 0;
        } else if (N % p != 0) {
            if (rank == 0)
                fprintf(stderr, "Skipping N=%d: not divisible by sqrt(P)=%d for block algorithm.\n", N, p);
            valid = 0;
        }
    } else {
        if (N % size != 0) {
            if (rank == 0)
                fprintf(stderr, "Skipping N=%d: not divisible by P=%d.\n", N, size);
            valid = 0;
        }
    }

    if (!valid) {
        if (rank == 0) {
            free(A_full);
        }
        free(x_full);
        free(y_full);
        return 0;
    }

    if (algo == 1) {
        algo_name = "row";
        T_par = multiply_rowwise(N, A_full, x_full, y_full, rank, size);
    } else if (algo == 2) {
        algo_name = "column";
        T_par = multiply_columnwise(N, A_full, x_full, y_full, rank, size);
    } else if (algo == 3) {
        algo_name = "block";
        T_par = multiply_blockwise(N, A_full, x_full, y_full, rank, size);
        if (T_par < 0) {
            if (rank != 0) {
                free(y_full);
            }
            free(x_full);
            if (rank == 0) {
                free(A_full);
            }
            return 1;
        }
    } else {
        if (rank == 0)
            fprintf(stderr, "Error: invalid algorithm. Use 1=row, 2=column, 3=block\n");
        if (rank != 0) {
            free(y_full);
        }
        free(x_full);
        if (rank == 0) {
            free(A_full);
        }
        return 1;
    }

    if (rank == 0) {
        double speedup = T_seq / T_par;
        double efficiency = speedup / size;
        printf("%d,%d,%s,%.6f,%.6f,%.4f,%.4f\n", N, size, algo_name, T_par, T_seq, speedup, efficiency);
    }

    if (rank == 0) {
        free(A_full);
    }
    free(x_full);
    free(y_full);

    return 0;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int algo_specified = (argc > 1) ? atoi(argv[1]) : 0;

    int N_specified = (argc > 2) ? atoi(argv[2]) : 0;

    int sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int num_sizes = 8;

    int algorithms[] = {1, 2, 3};
    int num_algorithms = 3;

    if (algo_specified < 0 || algo_specified > 3) {
        if (rank == 0)
            fprintf(stderr, "Error: invalid algorithm. Use 0=all, 1=row, 2=column, 3=block\n");
        MPI_Finalize();
        return 1;
    }

    int start_algo, end_algo;
    if (algo_specified == 0) {
        start_algo = 0;
        end_algo = num_algorithms - 1;
    } else {
        start_algo = algo_specified - 1;
        end_algo = algo_specified - 1;
    }
    printf("N, size, algo_name, T_par, T_seq, speedup, efficiency\n");
    if (N_specified > 0) {
        for (int a = start_algo; a <= end_algo; ++a) {
            if (run_experiment(N_specified, algorithms[a], rank, size) != 0) {
                MPI_Finalize();
                return 1;
            }
        }
    } else {
        for (int a = start_algo; a <= end_algo; ++a) {
            for (int i = 0; i < num_sizes; ++i) {
                run_experiment(sizes[i], algorithms[a], rank, size);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
