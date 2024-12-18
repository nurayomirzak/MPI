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

    const int numMatrices = 1024; // Общее количество матриц
    const int matrixSize = AES_BLOCK_SIZE; // Размер одной матрицы
    const int matricesPerProcess = numMatrices / size; // Число матриц на процесс

    std::vector<uint8_t> allData; // Все данные (только на процессе 0)
    std::vector<uint8_t> localData(matricesPerProcess * matrixSize); // Локальный буфер для данных

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

    // Рассылка данных с помощью MPI_Scatter
    MPI_Scatter(allData.data(), matricesPerProcess * matrixSize, MPI_UNSIGNED_CHAR,
                localData.data(), matricesPerProcess * matrixSize, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // Обработка данных
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < matricesPerProcess; ++i) {
        std::vector<uint8_t> matrix(localData.begin() + i * matrixSize, localData.begin() + (i + 1) * matrixSize);
        processMatrix(matrix);
        std::copy(matrix.begin(), matrix.end(), localData.begin() + i * matrixSize);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Сбор обработанных данных с помощью MPI_Gather
    MPI_Gather(localData.data(), matricesPerProcess * matrixSize, MPI_UNSIGNED_CHAR,
               allData.data(), matricesPerProcess * matrixSize, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // Вывод результата на процессе 0
    if (rank == 0) {
        std::cout << "Processing complete in " 
                  << std::chrono::duration<double>(end - start).count() << " seconds." << std::endl;

        // Сохранение данных в файл
        std::ofstream resultFile("results_mpi.txt");
        for (int i = 0; i < numMatrices; ++i) {
            resultFile << "Matrix " << i + 1 << ":\n";
            for (int j = 0; j < matrixSize; ++j) {
                resultFile << static_cast<int>(allData[i * matrixSize + j]) << " ";
                if ((j + 1) % 4 == 0) resultFile << "\n"; // Форматирование 4x4
            }
            resultFile << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
