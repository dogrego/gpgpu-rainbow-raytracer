#include <stdio.h>

// CUDA kernel that runs on the GPU
__global__ void hello_kernel()
{
    // Special variables provided by CUDA:
    // blockIdx.x - block index within the grid
    // threadIdx.x - thread index within the block
    printf("Hello World from GPU thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main()
{
    // Print from CPU
    printf("Hello World from CPU!\n");

    // Launch the kernel on the GPU
    // <<<number of blocks, threads per block>>>
    hello_kernel<<<2, 4>>>();

    // Wait for GPU to finish before CPU continues
    cudaDeviceSynchronize();

    return 0;
}