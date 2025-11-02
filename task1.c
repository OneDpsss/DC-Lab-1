#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size;
    long long total_points = 1000000;

    if (argc > 1) {
        total_points = atoll(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    double start_time = MPI_Wtime();

    long long points_per_process = total_points / size;
    long long local_hits = 0;
    long long local_points = points_per_process;

    if (rank == size - 1) {
        local_points = total_points - (size - 1) * points_per_process;
    }

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

    if (rank == 0) {
        long long sequential_points = total_points;
        long long sequential_hits = 0;
        
        double seq_start = MPI_Wtime();
        srand(time(NULL));
        
        for (long long i = 0; i < sequential_points; i++) {
            double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
            
            if (x * x + y * y <= 1.0) {
                sequential_hits++;
            }
        }
        
        double seq_end = MPI_Wtime();
        double sequential_time = seq_end - seq_start;
        
        double pi_estimate = 4.0 * (double)total_hits / (double)total_points;
        double speedup = sequential_time / parallel_time;
        double efficiency = speedup / size;
        
        printf("Total points: %lld\n", total_points);
        printf("Total hits in circle: %lld\n", total_hits);
        printf("Pi estimate: %.10f\n", pi_estimate);
        printf("Hit ratio: %.10f\n", (double)total_hits / total_points);
        printf("Expected ratio (π/4): %.10f\n", 3.14159265358979323846 / 4.0);
        printf("Sequential time: %.6f seconds\n", sequential_time);
        printf("Parallel time: %.6f seconds\n", parallel_time);
        printf("Speedup: %.6f\n", speedup);
        printf("Efficiency: %.6f\n", efficiency);
    }

    MPI_Finalize();
    return 0;
}