#include "RainbowRayTracer.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define WIDTH 1024    // width of the generated image
#define HEIGHT 1024   // height of the generated image
#define CHANNEL_NUM 4 // the number of color channels per pixel
#define MAX_BOUNCES 4 // the maximum number of times a light ray can bounce or interact with the sphere during the ray tracing process

/**
 * @brief CUDA kernel to render a scene pixel by pixel.
 *
 * This kernel performs ray tracing to compute the color of each pixel in an image.
 * It simulates light propagation and interaction with a sphere in the scene. The kernel
 * computes the wavelength of light for each pixel and traces rays from a light source,
 * interacting with the sphere. The color is determined based on reflection, refraction,
 * and the Fresnel effect.
 *
 * @param pixels Pointer to the output pixel buffer. Each pixel is represented by 4 channels (RGBA).
 * @param sphere Pointer to a Sphere object in the scene. This sphere is the object being rendered.
 * @param light Pointer to a Light object in the scene. This light source is used to compute the light direction.
 * @param wavelengthStart The starting wavelength (in nm) for the rainbow effect.
 * @param wavelengthEnd The ending wavelength (in nm) for the rainbow effect.
 */
__global__ void renderKernel(unsigned char *pixels, Sphere *sphere, Light *light, float wavelengthStart, float wavelengthEnd)
{
    // Initialize the random number generator state for this thread
    curandState state;

    // Loop over every pixel in the image (height and width of the render)
    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            // Calculate the index in the pixel buffer for the current pixel
            int index = (y * WIDTH + x) * CHANNEL_NUM;

            // Compute the wavelength for the current pixel based on its y-coordinate (for a rainbow effect)
            float wavelength = wavelengthEnd - (wavelengthEnd - wavelengthStart) * (y / float(HEIGHT));

            // Calculate the direction of light from the light source to the sphere
            Vec3 lightDir = (sphere->center - light->pos).normalize();

            // Get the color of the pixel corresponding to the current wavelength
            Color baseColor = wavelengthToRGB(wavelength);

            // Trace the ray from the light source and compute the resulting color at the hit point
            Color color = traceRay(light->pos, lightDir, *sphere, MAX_BOUNCES, light->wavelength, &state);

            // Blend the calculated color with the base color (for a rainbow effect)
            color.r = fminf(255, color.r + baseColor.r);
            color.g = fminf(255, color.g + baseColor.g);
            color.b = fminf(255, color.b + baseColor.b);

            // Store the final color in the output pixel buffer (RGBA)
            pixels[index + 0] = color.r; // Red channel
            pixels[index + 1] = color.g; // Green channel
            pixels[index + 2] = color.b; // Blue channel
            pixels[index + 3] = 255;     // Alpha channel (full opacity)
        }
    }
}

int main()
{
    std::cout << "Starting CUDA SEQUENTIAL ray tracing..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    unsigned char *d_pixels, *pixels = new unsigned char[WIDTH * HEIGHT * CHANNEL_NUM];
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * CHANNEL_NUM);

    Sphere h_sphere = {{WIDTH / 2.0f, HEIGHT / 2.0f, 50.0f}, 40.0f};
    Light h_light = {{WIDTH / 2.0f, HEIGHT, 100.0f}, {0, -1, -0.5f}};

    Sphere *d_sphere;
    Light *d_light;
    cudaMalloc(&d_sphere, sizeof(Sphere));
    cudaMalloc(&d_light, sizeof(Light));
    cudaMemcpy(d_sphere, &h_sphere, sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemcpy(d_light, &h_light, sizeof(Light), cudaMemcpyHostToDevice);

    renderKernel<<<1, 1>>>(d_pixels, d_sphere, d_light, 380.0f, 750.0f);

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
