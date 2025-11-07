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
        fp = fopen("task1_fixed.txt", "w");
        if (fp == NULL) {
            printf("Error: Cannot open file task1_fixed.txt\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // CSV заголовок
        fprintf(fp, "n,Total_Points,Sequential_Hits,Sequential_Pi,Sequential_Error,Sequential_Time,Parallel_Hits,Parallel_Pi,Parallel_Error,Parallel_Time,Speedup,Efficiency,Time_Saved,Time_Saved_Percent\n");
    }

    // Инициализация генератора случайных чисел
    unsigned int seed = time(NULL) + rank * 1000;
    srand(seed);

    for (int n = 1; n <= 16; n++) {
        long long total_points = (long long)pow(2, n);
        

        // === БЕЗ БУСТА (последовательная версия) ===
        double sequential_time = 0.0;
        long long sequential_hits = 0;
        double pi_estimate_seq = 0.0;
        double error_seq = 0.0;
        
        if (rank == 0) {
            double seq_start = MPI_Wtime();
            
            // Сохраняем текущее состояние генератора
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
            
            // Восстанавливаем генератор для параллельной версии
            srand(seed);
        }

        // === С БУСТОМ (параллельная версия) ===
        double start_time = MPI_Wtime();

        long long points_per_process = total_points / size;
        long long local_hits = 0;
        long long local_points = points_per_process;

        // Корректировка для последнего процесса
        if (rank == size - 1) {
            local_points = total_points - (size - 1) * points_per_process;
        }

        // Локальные вычисления
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
            
            // ПРАВИЛЬНОЕ вычисление ускорения
            double speedup = sequential_time / parallel_time;
            double efficiency = speedup / size;
            
            double error_par = fabs(pi_estimate_par - 3.14159265358979323846);
            double time_saved = sequential_time - parallel_time;
            double time_saved_percent = (time_saved / sequential_time) * 100.0;
            
            // CSV строка с данными
            fprintf(fp, "%d,%lld,%lld,%.10f,%.10f,%.6f,%lld,%.10f,%.10f,%.6f,%.6f,%.6f,%.6f,%.2f\n",
                    n, total_points,
                    sequential_hits, pi_estimate_seq, error_seq, sequential_time,
                    total_hits, pi_estimate_par, error_par, parallel_time,
                    speedup, efficiency, time_saved, time_saved_percent);
        }
    }

    if (rank == 0) {
        fclose(fp);
        printf("Results saved to file task1_fixed.txt\n");
    }

    MPI_Finalize();
    return 0;
}