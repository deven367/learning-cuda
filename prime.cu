#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Define a structure to hold prime numbers
typedef struct {
    uint32_t num;
    int is_prime;
} PrimeNumber;

// Kernel function to find prime numbers
__global__ void findPrimes(uint32_t* numbers, uint32_t* primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Check if the number is prime
        if (numbers[idx] <= 1 || numbers[idx] % 2 == 0) {
            primes[idx].is_prime = 0;
        } else {
            for (int i = 3; i * i <= numbers[idx]; i += 2) {
                if (numbers[idx] % i == 0) {
                    primes[idx].is_prime = 0;
                    break;
                }
            } else {
                primes[idx].is_prime = 1;
            }
        }
    }
}

int main() {
    // Set the number of prime numbers to find
    int n = 1000000;

    // Allocate memory on the host for the numbers and primes
    uint32_t* host_numbers = (uint32_t*) malloc(n * sizeof(uint32_t));
    PrimeNumber* host_primes = (PrimeNumber*) malloc(n * sizeof(PrimeNumber));

    // Initialize the numbers with values from 2 to n
    for (int i = 0; i < n; i++) {
        host_numbers[i] = i + 2; // start from 2
    }

    // Allocate memory on the device for the numbers and primes
    uint32_t* device_numbers;
    PrimeNumber* device_primes;
    cudaMalloc((void**)&device_numbers, n * sizeof(uint32_t));
    cudaMalloc((void**)&device_primes, n * sizeof(PrimeNumber));

    // Copy the numbers to the device
    cudaMemcpy(device_numbers, host_numbers, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    findPrimes<<<numBlocks, blockSize>>>(device_numbers, device_primes, n);

    // Synchronize the kernel
    cudaDeviceSynchronize();

    // Copy the primes from the device to the host
    cudaMemcpy(host_primes, device_primes, n * sizeof(PrimeNumber), cudaMemcpyDeviceToHost);

    // Print the prime numbers
    for (int i = 0; i < n; i++) {
        if (host_primes[i].is_prime) {
            printf("%u\n", host_primes[i].num);
        }
    }

    // Free memory
    free(host_numbers);
    free(host_primes);
    cudaFree(device_numbers);
    cudaFree(device_primes);

    return 0;
}