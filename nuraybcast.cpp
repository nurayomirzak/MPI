#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

const int AES_BLOCK_SIZE = 16; // Размер блока AES (16 байт)

void processMatrix(std::vector<uint8_t>& matrix) {
    // Простая обработка для примера
    for (auto& byte : matrix) {
        byte ^= 0xFF; // Инверсия байтов
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int numMatrices = 1024; // Количество матриц
    const int matrixSize = AES_BLOCK_SIZE; // Размер одной матрицы

    std::vector<uint8_t> allData;
    if (rank == 0) {
        // Генерация данных на процессе 0
        allData.resize(numMatrices * matrixSize);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        for (auto& byte : allData) {
            byte = dis(gen);
        }
    }

    // Распространение данных на все процессы
    if (rank == 0) {
        std::cout << "Broadcasting data..." << std::endl;
    }
    allData.resize(numMatrices * matrixSize);
    MPI_Bcast(allData.data(), numMatrices * matrixSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Определяем границы работы для текущего процесса
    int matricesPerProcess = numMatrices / size;
    int startIdx = rank * matricesPerProcess * matrixSize;
    int endIdx = (rank == size - 1) ? numMatrices * matrixSize : startIdx + matricesPerProcess * matrixSize;

    // Обработка данных
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = startIdx; i < endIdx; i += matrixSize) {
        std::vector<uint8_t> matrix(allData.begin() + i, allData.begin() + i + matrixSize);
        processMatrix(matrix);
        std::copy(matrix.begin(), matrix.end(), allData.begin() + i);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Сбор обработанных данных
    std::vector<uint8_t> resultData;
    if (rank == 0) {
        resultData.resize(numMatrices * matrixSize);
    }
    MPI_Gather(allData.data() + startIdx, (endIdx - startIdx), MPI_UNSIGNED_CHAR,
               resultData.data(), (endIdx - startIdx), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Processing complete in " 
                  << std::chrono::duration<double>(end - start).count() << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
