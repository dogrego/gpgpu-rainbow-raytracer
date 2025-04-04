#include "RainbowRayTracer.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define WIDTH 1024      // width of the generated image
#define HEIGHT 1024     // height of the generated image
#define CHANNEL_NUM 4   // the number of color channels per pixel
#define MAX_BOUNCES 4   // the maximum number of times a light ray can bounce or interact with the sphere during the ray tracing process

int main() {
    std::cout << "Starting CUDA SEQUENTIAL ray tracing..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    unsigned char* d_pixels, * pixels = new unsigned char[WIDTH * HEIGHT * CHANNEL_NUM];
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * CHANNEL_NUM);

    Sphere h_sphere = { {WIDTH / 2.0f, HEIGHT / 2.0f, 50.0f}, 40.0f };
    Light h_light = { {WIDTH / 2.0f, HEIGHT, 100.0f}, {0, -1, -0.5f} };

    Sphere* d_sphere;
    Light* d_light;
    cudaMalloc(&d_sphere, sizeof(Sphere));
    cudaMalloc(&d_light, sizeof(Light));
    cudaMemcpy(d_sphere, &h_sphere, sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemcpy(d_light, &h_light, sizeof(Light), cudaMemcpyHostToDevice);

    renderKernel <<<1, 1 >>> (d_pixels, d_sphere, d_light, 380.0f, 750.0f);

    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, WIDTH * HEIGHT * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();

    stbi_write_png("rainbow_cuda_seq.png", WIDTH, HEIGHT, CHANNEL_NUM, pixels, WIDTH * CHANNEL_NUM);

    std::cout << "CUDA ray tracing completed." << std::endl;
    std::cout << "Image dimensions: " << WIDTH << "x" << HEIGHT << std::endl;
    std::cout << "Max bounces: " << MAX_BOUNCES << std::endl;
    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms.\n";

    cudaFree(d_pixels);
    cudaFree(d_sphere);
    cudaFree(d_light);
    delete[] pixels;

    std::cout << "Output saved to rainbow_cuda_seq.png\n";
    std::cout << "Press Enter to exit...";
    std::cin.ignore();
    return 0;
}
