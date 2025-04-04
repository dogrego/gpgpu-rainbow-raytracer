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
 * @brief Computes perfect reflection direction (R = I - 2·N(N·I))
 *
 * Implements the standard reflection equation for ideal mirrors:
 * R = I - 2·N(N·I) where:
 * - I: Incident direction (unit vector)
 * - N: Surface normal (unit vector)
 * - R: Reflected direction (unit vector)
 *
 * @param I Incident direction (must be normalized)
 * @param N Surface normal (must be normalized)
 * @return Reflected direction (normalized)
 *
 * Note: Equivalent to GLSL reflect() function.
 *       Maintains input vector normalization.
 */
__device__ Vec3 reflect(const Vec3 &I, const Vec3 &N)
{
    return I - N * 2.0f * I.dot(N);
}

/**
 * @brief Computes refraction direction using Snell's law.
 *
 * Implements:
 * n₁·sinθ₁ = n₂·sinθ₂
 * Returns zero vector for total internal reflection (when sinθ₂ > 1)
 *
 * @param incident Incident direction (unit vector)
 * @param normal   Surface normal (unit vector)
 * @param n1       Refractive index of origin medium (η₁)
 * @param n2       Refractive index of destination medium (η₂)
 * @return Refracted direction (unit vector if valid) or
 *         zero vector for total internal reflection
 */
__device__ Vec3 refract(const Vec3 &incident, const Vec3 &normal, float n1, float n2)
{
    float ratio = n1 / n2;                                         // Refractive index ratio
    float cosI = -fmaxf(-1.0f, fminf(1.0f, incident.dot(normal))); // Clamp the dot product to [-1, 1]
    float sinT2 = ratio * ratio * (1.0f - cosI * cosI);            // Sine squared of the refraction angle

    if (sinT2 > 1.0f)
        return Vec3(0, 0, 0); // Total internal reflection: return zero vector

    float cosT = sqrtf(1.0f - sinT2);                         // Cosine of the refraction angle
    return incident * ratio + normal * (ratio * cosI - cosT); // Refraction direction
}

/**
 * @brief Fresnel equations for reflection and refraction blending
 *
 * Implements the Fresnel equations for unpolarized light:
 * R = ½(R_∥ + R_⊥) where:
 * R_∥ = ((n₂·cosθ₁ - n₁·cosθ₂)/(n₂·cosθ₁ + n₁·cosθ₂))²
 * R_⊥ = ((n₁·cosθ₁ - n₂·cosθ₂)/(n₁·cosθ₁ + n₂·cosθ₂))²
 *
 * @param incident Incident ray direction (unit vector)
 * @param normal   Surface normal (unit vector)
 * @param n1       Refractive index of origin medium
 * @param n2       Refractive index of destination medium
 * @return Reflectance ∈ [0.0,1.0] where:
 *         0.0 = full transmission
 *         1.0 = total internal reflection (when sinθ₂ > 1.0)
 */
__device__ float fresnel(const Vec3 &incident, const Vec3 &normal, float n1, float n2)
{
    float cosI = -fmaxf(-1.0f, fminf(1.0f, incident.dot(normal))); // Cosine of the angle of incidence
    float sinT2 = (n1 / n2) * (n1 / n2) * (1.0f - cosI * cosI);    // Sine squared of the angle of refraction

    if (sinT2 > 1.0f)
        return 1.0f; // Total internal reflection: return maximum reflectance

    float cosT = sqrtf(1.0f - sinT2);                                                 // Cosine of the angle of refraction
    float rParallel = ((n2 * cosI) - (n1 * cosT)) / ((n2 * cosI) + (n1 * cosT));      // Reflection for parallel polarization
    float rPerpendicular = ((n1 * cosI) - (n2 * cosT)) / ((n1 * cosI) + (n2 * cosT)); // Reflection for perpendicular polarization

    return (rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f; // Average reflection
}

/**
 * @brief Tests ray-sphere intersection using geometric solution.
 *
 * Solves quadratic equation: t² + 2b·t + c = 0
 * where:
 *   b = (origin - center) · direction
 *   c = (origin - center)² - radius²
 *
 * @param origin  Ray starting point (world space)
 * @param dir     Normalized ray direction
 * @param sphere  Sphere (center + radius) to test
 * @param t       Output: Distance to nearest intersection if found
 * @return true if ray hits sphere (with t > 0), false otherwise
 */
__device__ bool intersectRaySphere(const Vec3 &origin, const Vec3 &dir, const Sphere &sphere, float &t)
{
    Vec3 oc = origin - sphere.center;                     // Vector from ray origin to sphere center
    float b = 2.0f * oc.dot(dir);                         // Dot product term for the quadratic equation
    float c = oc.dot(oc) - sphere.radius * sphere.radius; // Constant term for the quadratic equation
    float discriminant = b * b - 4.0f * c;                // Discriminant of the quadratic equation

    // If the discriminant is positive, there are real intersections
    if (discriminant > 0)
    {
        t = (-b - sqrtf(discriminant)) / 2.0f; // Calculate the intersection distance
        return t > 0;                          // If t is positive, the intersection occurs in the direction of the ray
    }

    // No intersection
    return false;
}

/**
 * @brief Computes wavelength-dependent refractive index for water/air interface.
 *
 * Uses an empirical approximation of the Sellmeier dispersion formula:
 * n(λ) = 1.31477 + 0.0108148/log₁₀(0.00690246λ)
 *
 * @param wavelength Light wavelength in nanometers [380,780]
 * @return Refractive index (n) with:
 *         - n ≈ 1.33 for visible spectrum
 *         - Higher dispersion at shorter wavelengths (blue/violet)
 */
__device__ float wavelengthToRefraction(float wavelength)
{
    return 1.31477f + 0.0108148f / log10f(0.00690246f * wavelength);
}

/**
 * @brief Converts a visible light wavelength to sRGB color with gamma correction.
 *
 * Implements a piecewise linear approximation of the visible spectrum:
 * 380-440nm: Violet to Blue
 * 440-490nm: Blue to Cyan
 * 490-510nm: Cyan to Green
 * 510-580nm: Green to Yellow
 * 580-645nm: Yellow to Red
 * 645-780nm: Red
 *
 * @param wavelength Input wavelength in nanometers [380,780]
 * @return RGBA color with:
 *         - Gamma correction (γ=0.8)
 *         - Intensity falloff at spectrum edges
 *         - Alpha always 255 (opaque)
 *
 * Note: Returns black for wavelengths outside visible range.
 */
__device__ Color wavelengthToRGB(float wavelength)
{
    float gamma = 0.8;
    float intensityMax = 255;
    float factor;
    float r = 0.0, g = 0.0, b = 0.0;

    // Color Mapping
    if (wavelength >= 380 && wavelength < 440)
    {
        // violet to blue
        r = -(wavelength - 440) / (440 - 380);
        b = 1.0;
    }
    else if (wavelength >= 440 && wavelength < 490)
    {
        // blue to cyan
        g = (wavelength - 440) / (490 - 440);
        b = 1.0;
    }
    else if (wavelength >= 490 && wavelength < 510)
    {
        // cyan to green
        g = 1.0;
        b = -(wavelength - 510) / (510 - 490);
    }
    else if (wavelength >= 510 && wavelength < 580)
    {
        // green to yellow
        r = (wavelength - 510) / (580 - 510);
        g = 1.0;
    }
    else if (wavelength >= 580 && wavelength < 645)
    {
        // yellow to orange
        r = 1.0;
        g = -(wavelength - 645) / (645 - 580);
    }
    else if (wavelength >= 645 && wavelength < 781)
    {
        // orange to red
        r = 1.0;
    }

    // Intensity Factor
    if (wavelength >= 380 && wavelength < 420)
        // violet
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380);
    else if (wavelength >= 420 && wavelength < 701)
        // blue to red
        factor = 1.0;
    else if (wavelength >= 701 && wavelength < 781)
        // red
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700);
    else
        // no color - wavelength is outside the visible spectrum
        factor = 0.0;

    return {
        static_cast<unsigned char>(intensityMax * pow(r * factor, gamma)),
        static_cast<unsigned char>(intensityMax * pow(g * factor, gamma)),
        static_cast<unsigned char>(intensityMax * pow(b * factor, gamma)),
        255};
}

/**
 * @brief Traces a light ray through a sphere, simulating wavelength-dependent refraction and reflection.
 *
 * Implements:
 * - Snell's Law for refraction
 * - Fresnel equations for reflectance
 * - Chromatic dispersion via wavelength-to-refraction mapping
 * - Up to maxBounces interactions
 *
 * @param origin      Ray starting position (world space)
 * @param dir         Normalized ray direction
 * @param sphere      Sphere to intersect with (center + radius)
 * @param maxBounces  Maximum ray interactions (refractions+reflections)
 * @param wavelength  Light wavelength in nm (380-780) for dispersion effects
 *
 * @return Accumulated color from all ray interactions
 *
 * Note: Uses Russian roulette termination for reflectance/transmittance
 */
__device__ Color traceRay(const Vec3 &origin, const Vec3 &dir, const Sphere &sphere, int maxBounces, float wavelength, curandState *localRandState)
{
    Color color = {0, 0, 0, 255};
    Vec3 currentDir = dir, currentPos = origin;
    float refractiveIndex = wavelengthToRefraction(wavelength);
    bool isInside = false; // Start outside the sphere

    for (int bounce = 0; bounce < maxBounces; ++bounce)
    {
        float t;
        if (intersectRaySphere(currentPos, currentDir, sphere, t))
        {
            Vec3 hit = currentPos + currentDir * t;
            Vec3 normal = (hit - sphere.center).normalize();

            // Flip normal if ray is inside the sphere
            if (isInside)
                normal = -normal;

            // Compute Fresnel reflectance
            float n1 = isInside ? refractiveIndex : 1.0f;
            float n2 = isInside ? 1.0f : refractiveIndex;
            float reflectance = fresnel(currentDir, normal, n1, n2);

            // Compute reflection/refraction directions
            Vec3 reflected = reflect(currentDir, normal);
            Vec3 refracted = refract(currentDir, normal, n1, n2);

            // Blend color
            Color waveColor = wavelengthToRGB(wavelength);

            color.r = fminf(255.0f, color.r * (1 - reflectance) + waveColor.r * reflectance);
            color.g = fminf(255.0f, color.g * (1 - reflectance) + waveColor.g * reflectance);
            color.b = fminf(255.0f, color.b * (1 - reflectance) + waveColor.b * reflectance);

            // Update ray position/direction
            currentPos = hit;

            // Decide reflection or refraction
            if (refracted.dot(refracted) > 0)
            {
                float randVal = curand_uniform(localRandState); // Get a random float in [0, 1)
                currentDir = (randVal < reflectance) ? reflected : refracted;
                isInside = !isInside;
            }
            else
            {
                currentDir = reflected;
            }
        }
        else
            break; // Ray escaped
    }

    return color;
}

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
