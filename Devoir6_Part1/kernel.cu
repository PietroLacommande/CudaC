
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib> // For rand()
#include <ctime>   // For seeding rand()


typedef enum {
    KERNEL_ELEMENT = 0,
    KERNEL_ROW,
    KERNEL_COLUMN
} kernelType;

cudaError_t multiplyWithCuda(float* c, const float* a, const float* b, unsigned int size, kernelType type);

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


__global__ void multiplyKernelElement(float* c, const float* a, const float* b, int length)
{
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int column = (blockIdx.x * blockDim.x) + threadIdx.x;
    int indexmatC = (row * length) + column;
        
    int linearIndexRow;
    int linearIndexColumn;

    float sum=0;
    //Cette condition permet de rester dans les limites 
    if (row < length && column < length){
        for (int index = 0; index < length; index++) {
            linearIndexRow = (row * length) + index;
            linearIndexColumn = (index* length)+column;
            sum += a[linearIndexRow] * b[linearIndexColumn];
        }
        c[indexmatC] = sum;
    }
    
}

__global__ void multiplyKernelRow(float* c, const float* a, const float* b, int length)
{
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int indexMatC = (row * length);

    int linearIndexRow;
    int linearIndexColumn;

    //Cette condition permet de rester dans les limites 
    if (row < length) {

        for (int column = 0; column < length; column++) {
            float sum = 0;
            for (int index = 0; index < length; index++) {
                linearIndexRow = (row * length) + index;
                linearIndexColumn = (index * length) + column;
                sum += a[linearIndexRow] * b[linearIndexColumn];
            }
            c[(indexMatC + column)] = sum;
        }
        
        
    }

}

__global__ void multiplyKernelColumn(float* c, const float* a, const float* b, int length)
{
    int column = (blockIdx.x * blockDim.x) + threadIdx.x;
    int indexMatC = (column * length);

    int linearIndexRow;
    int linearIndexColumn;

    //Cette condition permet de rester dans les limites 
    if (column < length) {

        for (int row = 0; row < length; row++) {
            float sum = 0;
            for (int index = 0; index < length; index++) {
                linearIndexRow = (row * length) + index;
                linearIndexColumn = (index * length) + column;
                sum += a[linearIndexRow] * b[linearIndexColumn];
            }
            c[(indexMatC + row)] = sum;
        }


    }

}

int main()
{
    const int arraySize = 8;
    const float* a = createArray(arraySize, arraySize);
    const float* b = createArray(arraySize, arraySize);
    float c[(arraySize*arraySize)] = { 0 };

    // Print matrices A and B before computation
    printMatrix(a, arraySize, arraySize, "A");
    printMatrix(b, arraySize, arraySize, "B");

    // Add vectors in parallel.
    cudaError_t cudaStatus = multiplyWithCuda(c, a, b, arraySize, KERNEL_ELEMENT);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // Print result matrix C after computation
    printMatrix(c, arraySize, arraySize, "C");

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
cudaError_t multiplyWithCuda(float* c, const float* a, const float* b, unsigned int size, kernelType type)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

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
        

    // Launch a kernel on the GPU with one thread for each element.
    if (size == 4) {
        dim3 threadsParBlock(size, size, 1);
        dim3 nombreDeBlock(1, 1, 1);
        switch (type) {
            case KERNEL_ELEMENT:
                multiplyKernelElement << <nombreDeBlock, threadsParBlock >> > (dev_c, dev_a, dev_b, size);
                break;
            case KERNEL_ROW:
                multiplyKernelRow << <nombreDeBlock, threadsParBlock >> > (dev_c, dev_a, dev_b, size);
                break;
            case KERNEL_COLUMN:
                multiplyKernelColumn << <nombreDeBlock, threadsParBlock >> > (dev_c, dev_a, dev_b, size);
                break;

            default:
                break;
        }
            
        
    }

    else if (size >= 8 && size <= 64) {
        dim3 threadsParBlock(size, size, 1);
        dim3 nombreDeBlock((size / 8), (size / 8), 1);

        switch (type) {
            case KERNEL_ELEMENT:
                multiplyKernelElement << <nombreDeBlock, threadsParBlock >> > (dev_c, dev_a, dev_b, size);
                break;
            case KERNEL_ROW:
                multiplyKernelRow << <nombreDeBlock, threadsParBlock >> > (dev_c, dev_a, dev_b, size);
                break;
            case KERNEL_COLUMN:
                multiplyKernelColumn << <nombreDeBlock, threadsParBlock >> > (dev_c, dev_a, dev_b, size);
                break;

            default:
                break;
        }
    }
    

    else {
        printf("Undefined behaviour. Please use matrix size between 8 and 64\n");
        return cudaErrorInvalidValue;
    }

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
