#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mpi.h>

const int AES_BLOCK_SIZE = 16; // Размер блока AES (16 байт)

// Умножение в поле Галуа
uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    while (b) {
        if (b & 1) p ^= a;
        a = (a << 1) ^ (a & 0x80 ? 0x1b : 0);
        b >>= 1;
    }
    return p;
}

// gmixColumn для одного столбца
void gmixColumn(unsigned char* r) {
    unsigned char a[4];
    unsigned char b[4];
    unsigned char h;

    for (unsigned char c = 0; c < 4; c++) {
        a[c] = r[c];
        h = r[c] & 0x80;
        b[c] = r[c] << 1;
        if (h) b[c] ^= 0x1b;
    }

    r[0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
    r[1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
    r[2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
    r[3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0];
}

// Генерация случайных матриц
std::vector<std::vector<uint8_t>> generateMatrix() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

    std::vector<std::vector<uint8_t>> matrix(4, std::vector<uint8_t>(4));
    for (auto& row : matrix)
        for (auto& byte : row)
            byte = dis(gen);
    return matrix;
}

// Обработка матрицы
void processMatrix(std::vector<std::vector<uint8_t>>& matrix) {
    for (int col = 0; col < 4; ++col) {
        unsigned char column[4];
        for (int row = 0; row < 4; ++row)
            column[row] = matrix[row][col];

        gmixColumn(column);

        for (int row = 0; row < 4; ++row)
            matrix[row][col] = column[row];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int dataSize = 10 * 1024 * 1024; // Размер данных (например, 10 МБ)
    const int numMatrices = dataSize / AES_BLOCK_SIZE;

    std::vector<std::vector<std::vector<uint8_t>>> localMatrices;
    if (rank == 0) {
        // Генерация матриц только на главном процессе
        std::vector<std::vector<std::vector<uint8_t>>> matrices(numMatrices);
        for (int i = 0; i < numMatrices; ++i)
            matrices[i] = generateMatrix();

        // Распределение матриц между процессами
        int matricesPerProcess = numMatrices / size;
        for (int i = 1; i < size; ++i) {
            int startIdx = i * matricesPerProcess;
            int count = (i == size - 1) ? numMatrices - startIdx : matricesPerProcess;

            MPI_Send(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            for (int j = 0; j < count; ++j) {
                MPI_Send(matrices[startIdx + j].data()->data(), AES_BLOCK_SIZE, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
        // Сохраняем матрицы для обработки главным процессом
        localMatrices.insert(localMatrices.end(), matrices.begin(), matrices.begin() + matricesPerProcess);
    } else {
        // Получение данных от главного процесса
        int localCount;
        MPI_Recv(&localCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localMatrices.resize(localCount, std::vector<std::vector<uint8_t>>(4, std::vector<uint8_t>(4)));

        for (int i = 0; i < localCount; ++i) {
            MPI_Recv(localMatrices[i].data()->data(), AES_BLOCK_SIZE, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Обработка матриц
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& matrix : localMatrices) {
        processMatrix(matrix);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    if (rank == 0) {
        std::cout << "Processing time on process " << rank << ": " << duration.count() << " seconds" << std::endl;
    }

    // Сбор результатов на главном процессе
    if (rank != 0) {
        for (const auto& matrix : localMatrices) {
            MPI_Send(matrix.data()->data(), AES_BLOCK_SIZE, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        std::vector<std::vector<std::vector<uint8_t>>> resultMatrices(numMatrices);
        for (size_t i = 0; i < localMatrices.size(); ++i) {
            resultMatrices[i] = localMatrices[i];
        }

        for (int i = 1; i < size; ++i) {
            int startIdx = i * (numMatrices / size);
            int count = (i == size - 1) ? numMatrices - startIdx : (numMatrices / size);

            for (int j = 0; j < count; ++j) {
                std::vector<std::vector<uint8_t>> matrix(4, std::vector<uint8_t>(4));
                MPI_Recv(matrix.data()->data(), AES_BLOCK_SIZE, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                resultMatrices[startIdx + j] = matrix;
            }
        }

        std::ofstream resultFile("results_mpi.txt");
        for (size_t i = 0; i < resultMatrices.size(); ++i) {
            resultFile << "Matrix " << i + 1 << ":\n";
            for (const auto& row : resultMatrices[i]) {
                for (const auto& byte : row)
                    resultFile << static_cast<int>(byte) << " ";
                resultFile << "\n";
            }
            resultFile << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
