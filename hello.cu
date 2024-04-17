#include <stdio.h>

__global__ void helloCUDA()
{
    printf("Hello World from thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main()
{
    // Define grid and block dimensions
    int blocks = 1;
    int threadsPerBlock = 5;

    // Launch the kernel
    helloCUDA<<<blocks, threadsPerBlock>>>();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    return 0;
}
