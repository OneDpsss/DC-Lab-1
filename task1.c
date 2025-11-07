#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char** argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FILE *fp = NULL;
    if (rank == 0) {
        fp = fopen("task1.txt", "w");
        if (fp == NULL) {
            printf("Error: Cannot open file task1.txt\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp, "=== Тесты для чисел 2^n (n от 1 до 16) ===\n\n");
        fprintf(fp, "Количество процессов: %d\n\n", size);
    }

    for (int n = 1; n <= 16; n++) {
        long long total_points = (long long)pow(2, n);
        
        if (rank == 0) {
            fprintf(fp, "----------------------------------------\n");
            fprintf(fp, "n = %d, Total points = 2^%d = %lld\n", n, n, total_points);
            fprintf(fp, "----------------------------------------\n");
        }

        // === БЕЗ БУСТА (последовательная версия) ===
        if (rank == 0) {
            long long sequential_hits = 0;
            double seq_start = MPI_Wtime();
            srand(time(NULL));
            
            for (long long i = 0; i < total_points; i++) {
                double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
                double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
                
                if (x * x + y * y <= 1.0) {
                    sequential_hits++;
                }
            }
            
            double seq_end = MPI_Wtime();
            double sequential_time = seq_end - seq_start;
            double pi_estimate_seq = 4.0 * (double)sequential_hits / (double)total_points;
            
            fprintf(fp, "\nБЕЗ БУСТА (последовательная версия):\n");
            fprintf(fp, "  Total hits in circle: %lld\n", sequential_hits);
            fprintf(fp, "  Pi estimate: %.10f\n", pi_estimate_seq);
            fprintf(fp, "  Hit ratio: %.10f\n", (double)sequential_hits / total_points);
            fprintf(fp, "  Expected ratio (π/4): %.10f\n", 3.14159265358979323846 / 4.0);
            fprintf(fp, "  Time: %.6f seconds\n", sequential_time);
        }

        // === С БУСТОМ (параллельная версия) ===
        MPI_Barrier(MPI_COMM_WORLD);
        
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
            double pi_estimate_par = 4.0 * (double)total_hits / (double)total_points;
            
            // Вычисляем sequential time для сравнения
            long long sequential_hits_for_speedup = 0;
            double seq_start_speedup = MPI_Wtime();
            srand(time(NULL) + 999);
            
            for (long long i = 0; i < total_points; i++) {
                double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
                double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
                
                if (x * x + y * y <= 1.0) {
                    sequential_hits_for_speedup++;
                }
            }
            
            double seq_end_speedup = MPI_Wtime();
            double sequential_time_for_speedup = seq_end_speedup - seq_start_speedup;
            double speedup = sequential_time_for_speedup / parallel_time;
            double efficiency = speedup / size;
            
            fprintf(fp, "\nС БУСТОМ (параллельная версия):\n");
            fprintf(fp, "  Total hits in circle: %lld\n", total_hits);
            fprintf(fp, "  Pi estimate: %.10f\n", pi_estimate_par);
            fprintf(fp, "  Hit ratio: %.10f\n", (double)total_hits / total_points);
            fprintf(fp, "  Expected ratio (π/4): %.10f\n", 3.14159265358979323846 / 4.0);
            fprintf(fp, "  Time: %.6f seconds\n", parallel_time);
            fprintf(fp, "  Speedup: %.6f\n", speedup);
            fprintf(fp, "  Efficiency: %.6f\n", efficiency);
            fprintf(fp, "\n");
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        fclose(fp);
        printf("Результаты сохранены в файл task1.txt\n");
    }

    MPI_Finalize();
    return 0;
}