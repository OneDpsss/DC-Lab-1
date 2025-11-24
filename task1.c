#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments
    long long total_points = 0;
    
    if (rank == 0) {
        if (argc < 2) {
            printf("Usage: mpirun -n <num_processes> %s <num_points>\n", argv[0]);
            printf("Example: mpirun -n 4 %s 1000000\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        total_points = atoll(argv[1]);
        if (total_points <= 0) {
            printf("Error: number of points must be a positive number\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast number of points to all processes
    MPI_Bcast(&total_points, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Processes: %d, Points: %lld\n", size, total_points);
    }

    // Initialize random number generator
    unsigned int seed = time(NULL) + rank * 1000;
    srand(seed);

    // === SEQUENTIAL VERSION ===
    double sequential_time = 0.0;
    long long sequential_hits = 0;
    double pi_estimate_seq = 0.0;
    double error_seq = 0.0;

    if (rank == 0) {
        double seq_start = MPI_Wtime();

        // Save current generator state
        unsigned int seq_seed = seed;
        srand(seq_seed);

        for (long long i = 0; i < total_points; i++) {
            double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand() / RAND_MAX * 2.0 - 1.0;

            if (x * x + y * y <= 1.0) {
                sequential_hits++;
            }
        }

        double seq_end = MPI_Wtime();
        sequential_time = seq_end - seq_start;
        pi_estimate_seq = 4.0 * (double)sequential_hits / (double)total_points;
        error_seq = fabs(pi_estimate_seq - 3.14159265358979323846);

        // Restore generator for parallel version
        srand(seed);
    }

    // === PARALLEL VERSION ===
    double start_time = MPI_Wtime();

    long long points_per_process = total_points / size;
    long long local_hits = 0;
    long long local_points = points_per_process;

    // Adjust for last process
    if (rank == size - 1) {
        local_points = total_points - (size - 1) * points_per_process;
    }

    // Local computations
    for (long long i = 0; i < local_points; i++) {
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;

        if (x * x + y * y <= 1.0) {
            local_hits++;
        }
    }

    long long total_hits;
    MPI_Reduce(&local_hits, &total_hits, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double parallel_time = end_time - start_time;

    // Print results
    if (rank == 0) {
        double pi_estimate_par = 4.0 * (double)total_hits / (double)total_points;
        double speedup = sequential_time / parallel_time;
        double efficiency = speedup / size;
        double error_par = fabs(pi_estimate_par - 3.14159265358979323846);
        double time_saved = sequential_time - parallel_time;
        double time_saved_percent = (time_saved / sequential_time) * 100.0;
        double hit_rate_seq = (double)sequential_hits / (double)total_points * 100.0;
        double hit_rate_par = (double)total_hits / (double)total_points * 100.0;

        printf("\n========================================\n");
        printf("RESULTS\n");
        printf("========================================\n\n");
        
        printf("SEQUENTIAL VERSION:\n");
        printf("  Hits in circle:     %lld / %lld (%.2f%%)\n", sequential_hits, total_points, hit_rate_seq);
        printf("  Pi estimate:        %.10f\n", pi_estimate_seq);
        printf("  Error:              %.10f\n", error_seq);
        printf("  Relative error:     %.6f%%\n", (error_seq / 3.14159265358979323846) * 100.0);
        printf("  Execution time:     %.6f sec\n", sequential_time);
        printf("\n");
        
        printf("PARALLEL VERSION:\n");
        printf("  Hits in circle:     %lld / %lld (%.2f%%)\n", total_hits, total_points, hit_rate_par);
        printf("  Pi estimate:        %.10f\n", pi_estimate_par);
        printf("  Error:              %.10f\n", error_par);
        printf("  Relative error:     %.6f%%\n", (error_par / 3.14159265358979323846) * 100.0);
        printf("  Execution time:     %.6f sec\n", parallel_time);
        printf("\n");
        
        printf("PERFORMANCE METRICS:\n");
        printf("  Speedup:            %.6fx\n", speedup);
        printf("  Efficiency:         %.6f (%.2f%%)\n", efficiency, efficiency * 100.0);
        printf("  Time saved:         %.6f sec\n", time_saved);
        printf("  Time saved percent: %.2f%%\n", time_saved_percent);
        printf("========================================\n");
    }

    MPI_Finalize();
    return 0;
}