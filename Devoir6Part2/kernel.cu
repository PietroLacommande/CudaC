
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>
#include <cstdlib> // For rand()
#include <ctime>   // For seeding rand()

#define tileLength (int)2


cudaError_t multiplyWithCuda(float* c, const float* a, const float* b, unsigned int size);

void printMatrix(const float* matrix, int rows, int columns, const char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; ++i) {
        printf("{ "); // Start of row delimiter
        for (int j = 0; j < columns; ++j) {
            printf("%.2f", matrix[i * columns + j]); // Print with 2 decimal place
            if (j < columns - 1) {
                printf(" "); // Add space between elements
            }
        }
        printf(" }"); // End of row delimiter
        printf("\n"); // Newline after each row
    }
    printf("\n");
}



float* createArray(int rows, int columns) {
    int totalSize = rows * columns;

    float* array = new float[totalSize];

    for (int i = 0; i < totalSize; ++i) {
        array[i] = static_cast<float>(std::rand()) / RAND_MAX * 4.0f; // Random floats [0, 4]    
    }
    return array;
}


__global__ void multiplyKernel(float* c, const float* a, const float* b, int length) {
    int threadY = threadIdx.y;
    int threadX = threadIdx.x;

    int row = (tileLength * blockIdx.y) + threadY;
    int column = (tileLength * blockIdx.x) + threadX;

    __shared__ float tileA[tileLength][tileLength]; 
    __shared__ float tileB[tileLength][tileLength];

    float sum = 0;

    int numTiles = (length + tileLength - 1) / tileLength; 

    for (int t = 0; t < numTiles; t++) {
        // Data loading
        if (row < length && (t * tileLength + threadX) < length) {
            tileA[threadY][threadX] = a[row * length + (t * tileLength + threadX)];
        }
        else {
            tileA[threadY][threadX] = 0.0f;
        }

        if (column < length && (t * tileLength + threadY) < length) {
            tileB[threadY][threadX] = b[(t * tileLength + threadY) * length + column];
        }
        else {
            tileB[threadY][threadX] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < tileLength; k++) {
            sum += tileA[threadY][k] * tileB[k][threadX];
        }

        __syncthreads();
    }

    if (row < length && column < length) {  
        c[(row * length) + column] = sum;
    }
}



int main()
{
    const int matrixSize = 16;
    const float* a = createArray(matrixSize, matrixSize);
    const float* b = createArray(matrixSize, matrixSize);
    float c[(matrixSize * matrixSize)] = { 0 };

    // Print matrices A and B before computation
    printMatrix(a, matrixSize, matrixSize, "A");
    printMatrix(b, matrixSize, matrixSize, "B");

    // Add vectors in parallel.
    cudaError_t cudaStatus = multiplyWithCuda(c, a, b, matrixSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // Print result matrix C after computation
    printMatrix(c, matrixSize, matrixSize, "C");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t multiplyWithCuda(float* c, const float* a, const float* b, unsigned int size)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    if (size >64) {
        printf("Taille de la matrice limité à 64 par 64");
        return cudaErrorInvalidValue;
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, (size * size) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, (size * size) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, (size * size) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, (size * size) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cudaStatus = cudaMemcpy(dev_b, b, (size * size) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



    dim3 threadsParBlock(tileLength, tileLength, 1);
    dim3 nombreDeBlock((size + tileLength - 1) / tileLength, (size + tileLength - 1) / tileLength);

    multiplyKernel << <nombreDeBlock, threadsParBlock >> > (dev_c, dev_a, dev_b, size);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, (size * size) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
