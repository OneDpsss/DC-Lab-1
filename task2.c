#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define MAX_SIZE 10000
#define NUM_EXPERIMENTS 5

// Структура для хранения результатов измерений
typedef struct {
    double time_sequential;
    double time_row;
    double time_column;
    double time_block;
    double speedup_row;
    double speedup_column;
    double speedup_block;
    double efficiency_row;
    double efficiency_column;
    double efficiency_block;
    int size;
    int num_procs;
} ExperimentResult;

// Функция для выделения памяти под матрицу
double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

// Функция для освобождения памяти матрицы
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Инициализация матрицы случайными значениями
void init_matrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (double)(rand() % 100) / 10.0;
        }
    }
}

// Инициализация вектора случайными значениями
void init_vector(double* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = (double)(rand() % 100) / 10.0;
    }
}

// Умножение матрицы на вектор - разбиение по строкам
double matrix_vector_multiply_rowwise(
    double** matrix, double* vector, double* result,
    int matrix_rows, int matrix_cols, int rank, int num_procs) {
    
    double start_time = MPI_Wtime();
    
    // Каждый процесс получает свою часть строк
    int rows_per_proc = matrix_rows / num_procs;
    int remainder = matrix_rows % num_procs;
    int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int my_start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    
    // Распределение матрицы по процессам
    double** local_matrix = allocate_matrix(my_rows, matrix_cols);
    
    // Подготовка данных для Scatterv: преобразуем матрицу в одномерный массив
    double* matrix_flat = NULL;
    int* sendcounts = (int*)malloc(num_procs * sizeof(int));
    int* displs = (int*)malloc(num_procs * sizeof(int));
    
    if (rank == 0) {
        matrix_flat = (double*)malloc(matrix_rows * matrix_cols * sizeof(double));
        // Преобразуем матрицу в одномерный массив
        for (int i = 0; i < matrix_rows; i++) {
            for (int j = 0; j < matrix_cols; j++) {
                matrix_flat[i * matrix_cols + j] = matrix[i][j];
            }
        }
        
        // Вычисляем размеры для Scatterv
        for (int i = 0; i < num_procs; i++) {
            int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = proc_rows * matrix_cols;
            displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
        }
    } else {
        for (int i = 0; i < num_procs; i++) {
            int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = proc_rows * matrix_cols;
            displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
        }
    }
    
    double* local_flat = (double*)malloc(my_rows * matrix_cols * sizeof(double));
    MPI_Scatterv(matrix_flat, sendcounts, displs, MPI_DOUBLE,
                 local_flat, my_rows * matrix_cols, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Преобразуем локальный одномерный массив обратно в матрицу
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < matrix_cols; j++) {
            local_matrix[i][j] = local_flat[i * matrix_cols + j];
        }
    }
    
    // Все процессы получают полный вектор
    double* local_vector = (double*)malloc(matrix_cols * sizeof(double));
    MPI_Bcast(vector, matrix_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    memcpy(local_vector, vector, matrix_cols * sizeof(double));
    
    // Освобождаем временную память
    if (rank == 0) {
        free(matrix_flat);
    }
    free(local_flat);
    free(sendcounts);
    free(displs);
    
    // Локальное вычисление
    double* local_result = (double*)calloc(my_rows, sizeof(double));
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < matrix_cols; j++) {
            local_result[i] += local_matrix[i][j] * local_vector[j];
        }
    }
    
    // Сбор результатов
    int* recvcounts = (int*)malloc(num_procs * sizeof(int));
    int* recv_displs = (int*)malloc(num_procs * sizeof(int));
    
    for (int i = 0; i < num_procs; i++) {
        int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
        recvcounts[i] = proc_rows;
        recv_displs[i] = (i == 0) ? 0 : recv_displs[i-1] + recvcounts[i-1];
    }
    
    MPI_Gatherv(local_result, my_rows, MPI_DOUBLE,
                result, recvcounts, recv_displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    
    // Очистка памяти
    free_matrix(local_matrix, my_rows);
    free(local_vector);
    free(local_result);
    free(recvcounts);
    free(recv_displs);
    
    return elapsed;
}

// Умножение матрицы на вектор - разбиение по столбцам
double matrix_vector_multiply_columnwise(
    double** matrix, double* vector, double* result,
    int matrix_rows, int matrix_cols, int rank, int num_procs) {
    
    double start_time = MPI_Wtime();
    
    // Каждый процесс получает свою часть столбцов
    int cols_per_proc = matrix_cols / num_procs;
    int remainder = matrix_cols % num_procs;
    int my_cols = cols_per_proc + (rank < remainder ? 1 : 0);
    int my_start_col = rank * cols_per_proc + (rank < remainder ? rank : remainder);
    
    // Распределение матрицы по процессам (каждый процесс получает все строки, но свои столбцы)
    // Преобразуем матрицу в одномерный массив для передачи
    double* matrix_flat = NULL;
    if (rank == 0) {
        matrix_flat = (double*)malloc(matrix_rows * matrix_cols * sizeof(double));
        for (int i = 0; i < matrix_rows; i++) {
            for (int j = 0; j < matrix_cols; j++) {
                matrix_flat[i * matrix_cols + j] = matrix[i][j];
            }
        }
    } else {
        matrix_flat = (double*)malloc(matrix_rows * matrix_cols * sizeof(double));
    }
    
    // Передаем всю матрицу всем процессам (можно оптимизировать, но для простоты используем Bcast)
    MPI_Bcast(matrix_flat, matrix_rows * matrix_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Каждый процесс выбирает свои столбцы
    double** local_matrix = allocate_matrix(matrix_rows, my_cols);
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < my_cols; j++) {
            int global_col = my_start_col + j;
            local_matrix[i][j] = matrix_flat[i * matrix_cols + global_col];
        }
    }
    
    // Распределение вектора
    double* full_vector = NULL;
    if (rank == 0) {
        full_vector = vector;
    } else {
        full_vector = (double*)malloc(matrix_cols * sizeof(double));
    }
    MPI_Bcast(full_vector, matrix_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double* local_vector = (double*)malloc(my_cols * sizeof(double));
    for (int j = 0; j < my_cols; j++) {
        int global_col = my_start_col + j;
        local_vector[j] = full_vector[global_col];
    }
    
    // Освобождаем временную память
    free(matrix_flat);
    if (rank != 0) {
        free(full_vector);
    }
    
    // Локальное вычисление частичных результатов
    double* local_result = (double*)calloc(matrix_rows, sizeof(double));
    for (int i = 0; i < matrix_rows; i++) {
        for (int j = 0; j < my_cols; j++) {
            local_result[i] += local_matrix[i][j] * local_vector[j];
        }
    }
    
    // Суммирование результатов от всех процессов
    MPI_Reduce(local_result, result, matrix_rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    
    // Очистка памяти
    free_matrix(local_matrix, matrix_rows);
    free(local_vector);
    free(local_result);
    
    return elapsed;
}

// Умножение матрицы на вектор - разбиение по блокам (2D разбиение)
double matrix_vector_multiply_blockwise(
    double** matrix, double* vector, double* result,
    int matrix_rows, int matrix_cols, int rank, int num_procs) {
    
    double start_time = MPI_Wtime();
    
    // Определяем размеры сетки процессов
    // Пробуем найти оптимальное разбиение (близкое к квадратному)
    int grid_rows = (int)sqrt(num_procs);
    while (grid_rows > 0 && num_procs % grid_rows != 0) {
        grid_rows--;
    }
    if (grid_rows == 0) grid_rows = 1;
    
    int grid_cols = num_procs / grid_rows;
    
    // Находим позицию процесса в сетке
    int proc_row = rank / grid_cols;
    int proc_col = rank % grid_cols;
    
    // Размеры локального блока для каждого процесса
    int rows_per_proc = matrix_rows / grid_rows;
    int cols_per_proc = matrix_cols / grid_cols;
    int remainder_rows = matrix_rows % grid_rows;
    int remainder_cols = matrix_cols % grid_cols;
    
    int my_rows = rows_per_proc + (proc_row < remainder_rows ? 1 : 0);
    int my_cols = cols_per_proc + (proc_col < remainder_cols ? 1 : 0);
    
    // Вычисляем начальные индексы для этого процесса
    int my_start_row = 0;
    for (int r = 0; r < proc_row; r++) {
        my_start_row += rows_per_proc + (r < remainder_rows ? 1 : 0);
    }
    
    int my_start_col = 0;
    for (int c = 0; c < proc_col; c++) {
        my_start_col += cols_per_proc + (c < remainder_cols ? 1 : 0);
    }
    
    // Распределение локального блока матрицы
    // Преобразуем матрицу в одномерный массив для передачи
    double* matrix_flat = NULL;
    if (rank == 0) {
        matrix_flat = (double*)malloc(matrix_rows * matrix_cols * sizeof(double));
        for (int i = 0; i < matrix_rows; i++) {
            for (int j = 0; j < matrix_cols; j++) {
                matrix_flat[i * matrix_cols + j] = matrix[i][j];
            }
        }
    } else {
        matrix_flat = (double*)malloc(matrix_rows * matrix_cols * sizeof(double));
    }
    
    // Передаем всю матрицу всем процессам
    MPI_Bcast(matrix_flat, matrix_rows * matrix_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Каждый процесс выбирает свой блок
    double** local_matrix = allocate_matrix(my_rows, my_cols);
    for (int i = 0; i < my_rows; i++) {
        int global_row = my_start_row + i;
        for (int j = 0; j < my_cols; j++) {
            int global_col = my_start_col + j;
            local_matrix[i][j] = matrix_flat[global_row * matrix_cols + global_col];
        }
    }
    
    // Распределение соответствующей части вектора
    double* full_vector = NULL;
    if (rank == 0) {
        full_vector = vector;
    } else {
        full_vector = (double*)malloc(matrix_cols * sizeof(double));
    }
    MPI_Bcast(full_vector, matrix_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double* local_vector = (double*)malloc(my_cols * sizeof(double));
    for (int j = 0; j < my_cols; j++) {
        int global_col = my_start_col + j;
        local_vector[j] = full_vector[global_col];
    }
    
    // Освобождаем временную память
    free(matrix_flat);
    if (rank != 0) {
        free(full_vector);
    }
    
    // Локальное вычисление частичного результата
    double* local_result = (double*)calloc(my_rows, sizeof(double));
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < my_cols; j++) {
            local_result[i] += local_matrix[i][j] * local_vector[j];
        }
    }
    
    // Сбор результатов: сначала суммируем по столбцам (в процессах с одинаковым proc_row)
    // Затем собираем результаты по строкам
    MPI_Comm row_comm;
    
    // Коммуникатор для строк процессов (процессы с одинаковым proc_row)
    int row_color = proc_row;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
    
    // Суммируем частичные результаты по столбцам в каждой строке процессов
    double* row_result = (double*)calloc(my_rows, sizeof(double));
    MPI_Allreduce(local_result, row_result, my_rows, MPI_DOUBLE, MPI_SUM, row_comm);
    
    // Теперь собираем результаты со всех строк процессов на процесс 0
    if (proc_col == 0) {
        // Вычисляем размеры для Gatherv
        int* recvcounts = (int*)malloc(grid_rows * sizeof(int));
        int* displs = (int*)malloc(grid_rows * sizeof(int));
        
        int offset = 0;
        for (int r = 0; r < grid_rows; r++) {
            int rows = rows_per_proc + (r < remainder_rows ? 1 : 0);
            recvcounts[r] = rows;
            displs[r] = offset;
            offset += rows;
        }
        
        MPI_Gatherv(row_result, my_rows, MPI_DOUBLE,
                    result, recvcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        free(recvcounts);
        free(displs);
    } else {
        // Отправляем пустые данные
        MPI_Gatherv(NULL, 0, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    
    MPI_Comm_free(&row_comm);
    
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    
    // Очистка памяти
    free_matrix(local_matrix, my_rows);
    free(local_vector);
    free(local_result);
    free(row_result);
    
    return elapsed;
}

// Последовательная версия (для baseline и проверки)
double sequential_multiply(double** matrix, double* vector, double* result,
                          int rows, int cols) {
    double start_time = MPI_Wtime();
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    double end_time = MPI_Wtime();
    return end_time - start_time;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (argc < 4) {
        if (rank == 0) {
            printf("Usage: %s <min_size> <max_size> <step> [output_file]\n", argv[0]);
            printf("Example: %s 100 2000 100 results.txt\n", argv[0]);
            printf("\nNote: Run with 1 process first to get sequential baseline\n");
            printf("      Then run with multiple processes for parallel results\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    int min_size = atoi(argv[1]);
    int max_size = atoi(argv[2]);
    int step = atoi(argv[3]);
    char* output_file = (argc >= 5) ? argv[4] : "results.txt";
    
    if (rank == 0) {
        printf("=== Matrix-Vector Multiplication Performance Analysis ===\n");
        printf("Processes: %d\n", num_procs);
        printf("Size range: %d to %d (step: %d)\n", min_size, max_size, step);
        printf("Output file: %s\n\n", output_file);
        
        FILE* fp = fopen(output_file, "w");
        if (fp == NULL) {
            printf("Error: Cannot open output file %s\n", output_file);
            MPI_Finalize();
            return 1;
        }
        fprintf(fp, "Size,Processes,Time_Sequential,Time_Row,Time_Column,Time_Block,"
                    "Speedup_Row,Speedup_Column,Speedup_Block,"
                    "Efficiency_Row,Efficiency_Column,Efficiency_Block\n");
        fclose(fp);
    }
    
    srand(time(NULL) + rank);
    
    // Для последовательного выполнения (1 процесс)
    int is_sequential = (num_procs == 1);
    
    for (int size = min_size; size <= max_size; size += step) {
        if (rank == 0) {
            printf("Testing size: %dx%d\n", size, size);
        }
        
        // Выделение памяти (только на процессе 0 для начальной инициализации)
        double** matrix = NULL;
        double* vector = NULL;
        double* result_row = NULL;
        double* result_col = NULL;
        double* result_block = NULL;
        double* result_seq = NULL;
        
        if (rank == 0) {
            matrix = allocate_matrix(size, size);
            vector = (double*)malloc(size * sizeof(double));
            result_row = (double*)malloc(size * sizeof(double));
            result_col = (double*)malloc(size * sizeof(double));
            result_block = (double*)malloc(size * sizeof(double));
            result_seq = (double*)malloc(size * sizeof(double));
            
            init_matrix(matrix, size, size);
            init_vector(vector, size);
        }
        
        double time_sequential = 0.0;
        double time_row = 0.0;
        double time_column = 0.0;
        double time_block = 0.0;
        
        if (is_sequential) {
            // Последовательное выполнение для baseline
            if (rank == 0) {
                time_sequential = sequential_multiply(matrix, vector, result_seq, size, size);
                // Для последовательной версии все методы одинаковы
                time_row = time_sequential;
                time_column = time_sequential;
                time_block = time_sequential;
                memcpy(result_row, result_seq, size * sizeof(double));
                memcpy(result_col, result_seq, size * sizeof(double));
                memcpy(result_block, result_seq, size * sizeof(double));
            }
        } else {
            // Параллельное выполнение
            // Умножение по строкам
            MPI_Barrier(MPI_COMM_WORLD);
            time_row = matrix_vector_multiply_rowwise(
                matrix, vector, result_row, size, size, rank, num_procs);
            
            // Умножение по столбцам
            MPI_Barrier(MPI_COMM_WORLD);
            time_column = matrix_vector_multiply_columnwise(
                matrix, vector, result_col, size, size, rank, num_procs);
            
            // Умножение по блокам
            MPI_Barrier(MPI_COMM_WORLD);
            time_block = matrix_vector_multiply_blockwise(
                matrix, vector, result_block, size, size, rank, num_procs);
            
            // Проверка результатов (на процессе 0)
            if (rank == 0) {
                time_sequential = sequential_multiply(matrix, vector, result_seq, size, size);
            }
        }
        
        // Вывод результатов и вычисление метрик (на процессе 0)
        if (rank == 0) {
            // Вычисляем ускорение и эффективность (только для параллельного выполнения)
            double speedup_row = 0.0, speedup_column = 0.0, speedup_block = 0.0;
            double efficiency_row = 0.0, efficiency_column = 0.0, efficiency_block = 0.0;
            
            if (!is_sequential && time_sequential > 0) {
                speedup_row = time_sequential / time_row;
                speedup_column = time_sequential / time_column;
                speedup_block = time_sequential / time_block;
                
                efficiency_row = speedup_row / num_procs;
                efficiency_column = speedup_column / num_procs;
                efficiency_block = speedup_block / num_procs;
            } else if (is_sequential) {
                // Для последовательной версии ускорение = 1, эффективность = 1
                speedup_row = speedup_column = speedup_block = 1.0;
                efficiency_row = efficiency_column = efficiency_block = 1.0;
            }
            
            // Проверка корректности (только для параллельного)
            if (!is_sequential) {
                int row_match = 1, col_match = 1, block_match = 1;
                double tolerance = 1e-6;
                
                for (int i = 0; i < size; i++) {
                    if (fabs(result_row[i] - result_seq[i]) > tolerance) {
                        row_match = 0;
                        if (i < 5) printf("Row diff[%d] = %.10f\n", i, result_row[i] - result_seq[i]);
                    }
                    if (fabs(result_col[i] - result_seq[i]) > tolerance) {
                        col_match = 0;
                        if (i < 5) printf("Col diff[%d] = %.10f\n", i, result_col[i] - result_seq[i]);
                    }
                    if (fabs(result_block[i] - result_seq[i]) > tolerance) {
                        block_match = 0;
                        if (i < 5) printf("Block diff[%d] = %.10f\n", i, result_block[i] - result_seq[i]);
                    }
                }
                
                printf("  Sequential:  %.6f sec\n", time_sequential);
                printf("  Row-wise:     %.6f sec (speedup: %.2f, efficiency: %.2f%%, correct: %s)\n", 
                       time_row, speedup_row, efficiency_row * 100, row_match ? "yes" : "no");
                printf("  Column-wise:  %.6f sec (speedup: %.2f, efficiency: %.2f%%, correct: %s)\n", 
                       time_column, speedup_column, efficiency_column * 100, col_match ? "yes" : "no");
                printf("  Block-wise:   %.6f sec (speedup: %.2f, efficiency: %.2f%%, correct: %s)\n", 
                       time_block, speedup_block, efficiency_block * 100, block_match ? "yes" : "no");
            } else {
                printf("  Sequential:   %.6f sec\n", time_sequential);
            }
            printf("\n");
            
            // Запись результатов
            FILE* fp = fopen(output_file, "a");
            fprintf(fp, "%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f\n",
                    size, num_procs, time_sequential, time_row, time_column, time_block,
                    speedup_row, speedup_column, speedup_block,
                    efficiency_row, efficiency_column, efficiency_block);
            fclose(fp);
            
            // Освобождение памяти
            free_matrix(matrix, size);
            free(vector);
            free(result_row);
            free(result_col);
            free(result_block);
            free(result_seq);
        }
    }
    
    MPI_Finalize();
    return 0;
}

