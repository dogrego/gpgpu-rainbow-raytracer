#include "RainbowRayTracer.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#define WIDTH 1024
#define HEIGHT 1024
#define CHANNEL_NUM 4
#define MAX_BOUNCES 4
#define SAMPLES 5

// Constants for random number generation
#define RNG_A 1664525u
#define RNG_C 1013904223u
#define RNG_MASK 0xFFFFFF

// Tile size for optimized memory access
#define TILE_SIZE 16

// vectorized wavelengths per thread
#define NUM_WAVELENGTHS 4

/**
 * Computes perfect reflection direction (R = I - 2·N(N·I))
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
__device__ float3 reflect(const float3 &I, const float3 &N)
{
    return I - N * 2.0f * dot(N, I);
}

/**
 * Computes refraction direction using Snell's law.
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
__device__ float3 refract(const float3 &incident, const float3 &normal, float n1, float n2)
{
    float ratio = n1 / n2;                                          // Refractive index ratio
    float cosI = -fmaxf(-1.0f, fminf(1.0f, dot(incident, normal))); // Clamp the dot product to [-1, 1]
    float sinT2 = ratio * ratio * (1.0f - cosI * cosI);             // Sine squared of the refraction angle

    if (sinT2 > 1.0f)
        return make_float3(0.0f, 0.0f, 0.0f); // Total internal reflection: return zero vector

    float cosT = sqrtf(1.0f - sinT2);                         // Cosine of the refraction angle
    return incident * ratio + normal * (ratio * cosI - cosT); // Refraction direction
}

/**
 * Fresnel equations for reflection and refraction blending
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
__device__ float fresnel(const float3 &incident, const float3 &normal, float n1, float n2)
{
    float cosI = -fmaxf(-1.0f, fminf(1.0f, dot(incident, normal))); // Cosine of the angle of incidence
    float sinT2 = (n1 / n2) * (n1 / n2) * (1.0f - cosI * cosI);     // Sine squared of the angle of refraction

    if (sinT2 > 1.0f)
        return 1.0f; // Total internal reflection: return maximum reflectance

    float cosT = sqrtf(1.0f - sinT2);                                                 // Cosine of the angle of refraction
    float rParallel = ((n2 * cosI) - (n1 * cosT)) / ((n2 * cosI) + (n1 * cosT));      // Reflection for parallel polarization
    float rPerpendicular = ((n1 * cosI) - (n2 * cosT)) / ((n1 * cosI) + (n2 * cosT)); // Reflection for perpendicular polarization

    return (rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f; // Average reflection
}

/**
 * Tests ray-sphere intersection using geometric solution.
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
__device__ bool intersectRaySphere(const float3 &origin, const float3 &dir, const Sphere &sphere, float &t)
{
    float3 oc = origin - sphere.center;                    // Vector from ray origin to sphere center
    float b = 2.0f * dot(oc, dir);                         // Dot product term for the quadratic equation
    float c = dot(oc, oc) - sphere.radius * sphere.radius; // Constant term for the quadratic equation
    float discriminant = b * b - 4.0f * c;                 // Discriminant of the quadratic equation

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
 * Tests intersection between a ray and an infinite plane.
 * @param origin  Ray origin point in world space
 * @param dir     Ray direction vector (does not need normalization)
 * @param plane   Plane defined by normal and distance (normal·X = distance)
 * @param t       Output: Distance to intersection point if valid
 * @return true if ray intersects plane in forward direction, false otherwise
 */
__device__ bool intersectRayPlane(const float3 &origin, const float3 &dir, const Plane &plane, float &t)
{
    // Compute denominator: dot product of plane normal and ray direction
    // Represents cosine of angle between them (0 = parallel)
    float denom = dot(plane.normal, dir);

    // Check if ray and plane are not parallel (denominator > epsilon)
    if (fabsf(denom) > 1e-6f)
    {
        // Calculate intersection distance t:
        // t = (plane distance - dot(normal, origin)) / denominator
        // This solves: origin + t*dir lies on plane normal·X = distance
        t = (plane.distance - dot(plane.normal, origin)) / denom;

        // Only return true for intersections in ray's forward direction
        return t >= 0;
    }

    // Ray is parallel to plane (no intersection possible)
    return false;
}

/**
 * Computes refractive index of water for a given wavelength using Sellmeier's equation.
 * Models the dispersion relationship of water at 20°C.
 *
 * Uses Sellmeier coefficients for pure water at 20°C
 * Implements the standard Sellmeier dispersion formula:
 *       n²(λ) = 1 + Σ(Bᵢλ²)/(λ²-Cᵢ) where λ is wavelength in microns
 * Wavelength conversion: 1nm = 0.001μm
 *
 * @param wavelength Light wavelength in nanometers (380-750nm for visible spectrum)
 * @return Refractive index of water at specified wavelength (unitless)
 */
__device__ float wavelengthToRefraction(float wavelength)
{
    // Convert input wavelength from nanometers to microns
    // (Sellmeier equation expects microns)
    float lambda_microns = wavelength / 1000.0f;

    // Precompute squared wavelength for Sellmeier terms
    float lambda_sq = lambda_microns * lambda_microns;

    // Sellmeier coefficients for water at 20°C (from empirical data)
    // B coefficients represent oscillator strengths
    const float B1 = 5.684027565e-1f; // First oscillator term
    const float B2 = 1.726177391e-1f; // Second oscillator term
    const float B3 = 2.086189578e-2f; // Third oscillator term

    // C coefficients represent resonance wavelengths squared
    const float C1 = 5.101829712e-3f; // First resonance term (μm²)
    const float C2 = 1.821153936e-2f; // Second resonance term (μm²)
    const float C3 = 2.620722293e-2f; // Third resonance term (μm²)

    // Compute Sellmeier dispersion equation:
    // n² = 1 + B₁λ²/(λ²-C₁) + B₂λ²/(λ²-C₂) + B₃λ²/(λ²-C₃)
    float n_squared = 1.0f +
                      (B1 * lambda_sq) / (lambda_sq - C1) + // UV contribution
                      (B2 * lambda_sq) / (lambda_sq - C2) + // Visible contribution
                      (B3 * lambda_sq) / (lambda_sq - C3);  // IR contribution

    // Return refractive index (sqrt of n²)
    return sqrtf(n_squared);
}

/**
 * Converts a visible light wavelength to sRGB color with gamma correction.
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
    float gamma = 0.8f;
    float intensityMax = 255.0f;
    float factor;
    float r = 0.0f, g = 0.0f, b = 0.0f;

    // Color Mapping
    if (wavelength >= 380.0f && wavelength < 440.0f)
    {
        // violet to blue
        r = -(wavelength - 440.0f) / (440.0f - 380.0f);
        b = 1.0f;
    }
    else if (wavelength >= 440.0f && wavelength < 490.0f)
    {
        // blue to cyan
        g = (wavelength - 440.0f) / (490.0f - 440.0f);
        b = 1.0f;
    }
    else if (wavelength >= 490.0f && wavelength < 510.0f)
    {
        // cyan to green
        g = 1.0f;
        b = -(wavelength - 510.0f) / (510.0f - 490.0f);
    }
    else if (wavelength >= 510.0f && wavelength < 580.0f)
    {
        // green to yellow
        r = (wavelength - 510.0f) / (580.0f - 510.0f);
        g = 1.0f;
    }
    else if (wavelength >= 580.0f && wavelength < 645.0f)
    {
        // yellow to orange
        r = 1.0f;
        g = -(wavelength - 645.0f) / (645.0f - 580.0f);
    }
    else if (wavelength >= 645.0f && wavelength < 781.0f)
    {
        // orange to red
        r = 1.0f;
    }

    // Intensity Factor
    if (wavelength >= 380.0f && wavelength < 420.0f)
        // violet
        factor = 0.3f + 0.7f * (wavelength - 380.0f) / (420.0f - 380.0f);
    else if (wavelength >= 420.0f && wavelength < 701.0f)
        // blue to red
        factor = 1.0f;
    else if (wavelength >= 701.0f && wavelength < 781.0f)
        // red
        factor = 0.3f + 0.7f * (780.0f - wavelength) / (780.0f - 700.0f);
    else
        // no color - wavelength is outside the visible spectrum
        factor = 0.0f;

    return {
        (unsigned char)(intensityMax * powf(r * factor, gamma)),
        (unsigned char)(intensityMax * powf(g * factor, gamma)),
        (unsigned char)(intensityMax * powf(b * factor, gamma)),
        255};
}

/**
 * Calculates light absorption coefficient based on wavelength.
 *
 * @param wavelength  Light wavelength in nanometers (nm)
 * @return Absorption coefficient (unitless, higher = more absorption)
 */
__device__ float getAbsorptionCoefficient(float wavelength)
{
    // Violet/UV range (wavelength < 400nm) - minimal absorption
    if (wavelength < 400.0f)
        return 0.01f;

    // Blue to cyan range (400-500nm) - linear increase from 0.01 to 0.04
    else if (wavelength < 500.0f)
        return 0.01f + 0.03f * (wavelength - 400.0f) / 100.0f;

    // Green to yellow range (500-600nm) - linear increase from 0.04 to 0.10
    else if (wavelength < 600.0f)
        return 0.04f + 0.06f * (wavelength - 500.0f) / 100.0f;

    // Orange to red range (600-700nm) - linear increase from 0.10 to 1.00
    else if (wavelength < 700.0f)
        return 0.1f + 0.9f * (wavelength - 600.0f) / 100.0f;

    // Near-infrared range (>700nm) - steep linear increase from 1.00
    else
        return 1.0f + 9.0f * (wavelength - 700.0f) / 80.0f;
}

/**
 * Traces a light ray through the scene, simulating optical effects.
 * @param origin       Starting position of the ray
 * @param dir          Initial direction of the ray (normalized)
 * @param sphere       Sphere object in the scene
 * @param plane        Plane object in the scene
 * @param maxBounces   Maximum number of ray bounces allowed
 * @param wavelength   Light wavelength in nanometers (400-700nm visible range)
 * @return             Computed color after ray tracing
 */
__device__ Color traceRay(const float3 &origin, const float3 &dir, const Sphere &sphere, const Plane &plane,
                          int maxBounces, float wavelength, unsigned int &seed)
{
    // Initialize as transparent black
    Color color = {0, 0, 0, 0};
    float3 currentDir = dir;
    float3 currentPos = origin;

    // Get refractive index based on wavelength (dispersion effect)
    float refractiveIndex = wavelengthToRefraction(wavelength);

    // Track whether we're inside the sphere (for refraction/absorption)
    bool isInside = false;

    // Track light intensity (decreases with absorption and bounces)
    float accumulatedIntensity = 1.0f;

    // Main ray bounce loop
    for (int bounce = 0; bounce < maxBounces; ++bounce)
    {
        // Check for sphere intersection
        float t_sphere;
        bool hit_sphere = intersectRaySphere(currentPos, currentDir, sphere, t_sphere);

        // Check for plane intersection
        float t_plane;
        bool hit_plane = intersectRayPlane(currentPos, currentDir, plane, t_plane);

        // Determine which object was hit first (if any)
        bool sphere_first = hit_sphere && (!hit_plane || t_sphere < t_plane);

        if (sphere_first)
        {
            // SPHERE INTERACTION ==============================================
            float3 intersectionPoint = currentPos + currentDir * t_sphere;

            // Compute surface normal (flipped if inside sphere)
            float3 normal = normalize((intersectionPoint - sphere.center));
            if (isInside)
            {
                normal = -normal;

                // Apply Beer-Lambert absorption inside the sphere
                float absorption = getAbsorptionCoefficient(wavelength);
                accumulatedIntensity *= expf(-absorption * t_sphere);
            }

            // Compute Fresnel reflectance at boundary
            float n1 = isInside ? refractiveIndex : 1.0f; // Current medium
            float n2 = isInside ? 1.0f : refractiveIndex; // New medium
            float reflectance = fresnel(currentDir, normal, n1, n2);

            // Compute reflection and refraction directions
            float3 reflected = reflect(currentDir, normal);
            float3 refracted = refract(currentDir, normal, n1, n2);

            // Optimized random number generation
            seed = (seed * RNG_A + RNG_C);
            float rand_val = (seed & RNG_MASK) / 16777216.0f;

            float continueProb = fmaxf(reflectance, 1.0f - reflectance);
            if (rand_val > continueProb)
                break;

            // Choose between reflection and refraction
            if (dot(refracted, refracted) > 0)
            { // Valid refraction
                seed = (seed * RNG_A + RNG_C);
                rand_val = (seed & RNG_MASK) / 16777216.0f;

                if (rand_val < reflectance / continueProb)
                {
                    currentDir = reflected; // Reflect
                }
                else
                {
                    currentDir = refracted; // Refract
                    isInside = !isInside;   // Toggle inside/outside state
                }
            }
            else
            {
                currentDir = reflected; // Total internal reflection
            }

            // Move slightly off surface to avoid self-intersection
            currentPos = intersectionPoint + currentDir * 1e-4f;
        }
        else if (hit_plane)
        {
            // PLANE INTERACTION (RAINBOW CAPTURE) ============================
            float3 intersectionPoint = currentPos + currentDir * t_plane;

            // Convert wavelength to RGB color
            Color wavelengthColor = wavelengthToRGB(wavelength);
            // Apply accumulated intensity and set opaque
            color.rgba.x = (unsigned char)(wavelengthColor.rgba.x * accumulatedIntensity);
            color.rgba.y = (unsigned char)(wavelengthColor.rgba.y * accumulatedIntensity);
            color.rgba.z = (unsigned char)(wavelengthColor.rgba.z * accumulatedIntensity);
            color.rgba.w = 255;
            break; // Stop tracing after capturing the rainbow
        }
        else
        {
            // MISSED ALL OBJECTS =============================================
            break;
        }

        // Stop if intensity becomes negligible
        if (accumulatedIntensity < 0.01f)
            break;
    }

    return color;
}

__global__ void renderKernel(unsigned char *pixels, const Sphere sphere, const Plane plane,
                             const float3 lightPos, float lightWidth, unsigned int frame_seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    // Visible light spectrum range (380nm-750nm)
    const float wavelengthStart = 380.0f;
    const float wavelengthEnd = 750.0f;
    const float wavelengthStep = (wavelengthEnd - wavelengthStart) / float(HEIGHT);

    // Output color accumulation
    float r_accum = 0.0f, g_accum = 0.0f, b_accum = 0.0f;

    // Unique seed per pixel
    unsigned int seed = frame_seed + y * WIDTH + x;

    // Anti-aliasing: Stratified sampling over light source
    for (int s = 0; s < SAMPLES; ++s)
    {
        // Random jitter for anti-aliasing
        seed = seed * RNG_A + RNG_C;
        float jitterX = (seed & RNG_MASK) / 16777216.0f - 0.5f;
        seed = seed * RNG_A + RNG_C;
        float jitterY = (seed & RNG_MASK) / 16777216.0f - 0.5f;

        float3 samplePos = {
            lightPos.x + jitterX * lightWidth,
            lightPos.y + jitterY * lightWidth,
            lightPos.z};

        // Calculate ray from light to current pixel
        float3 pixelPos = {float(x), float(y), 0.0f};
        float3 dir = normalize((pixelPos - samplePos));

        // Vectorized wavelength loop
#pragma unroll
        for (int w = 0; w < NUM_WAVELENGTHS; ++w)
        {
            // Wavelength varies with vertical position (rainbow effect)
            float wavelength = wavelengthEnd - (float(y * NUM_WAVELENGTHS + w) / float(HEIGHT * NUM_WAVELENGTHS)) * (wavelengthEnd - wavelengthStart);

            // Trace ray through scene
            Color sampleColor = traceRay(samplePos, dir, sphere, plane, MAX_BOUNCES, wavelength, seed);
            // Accumulate color samples (averaged later)
            r_accum += sampleColor.rgba.x / 255.0f;
            g_accum += sampleColor.rgba.y / 255.0f;
            b_accum += sampleColor.rgba.z / 255.0f;
        }
    }

    // Average over samples × wavelengths
    float scale = 1.0f / (SAMPLES * NUM_WAVELENGTHS);
    Color finalColor = {
        (unsigned char)(255.0f * fminf(r_accum * scale, 1.0f)),
        (unsigned char)(255.0f * fminf(g_accum * scale, 1.0f)),
        (unsigned char)(255.0f * fminf(b_accum * scale, 1.0f)),
        255};

    int index = (y * WIDTH + x) * CHANNEL_NUM;
    pixels[index + 0] = finalColor.rgba.x;
    pixels[index + 1] = finalColor.rgba.y;
    pixels[index + 2] = finalColor.rgba.z;
    pixels[index + 3] = finalColor.rgba.w;
}

int main()
{
    std::cout << "Starting CUDA parallel ray tracing..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Allocate unified memory for pixels (accessible by both CPU and GPU)
    unsigned char *d_pixels;
    cudaMallocManaged(&d_pixels, WIDTH * HEIGHT * CHANNEL_NUM);

    // ========== SCENE SETUP ==========
    // Water droplet sphere (centered in scene)
    Sphere sphere = {
        {WIDTH / 2.0f, HEIGHT / 2.0f, 500.0f}, // Center position
        300.0f};                               // Radius

    // Observation plane (catches rainbow rays)
    Plane plane = {
        {0.0f, 0.0f, -1.0f}, // Normal facing camera (negative Z)
        1800.0f,             // Distance from origin
        {0, 0, 0, 255}       // Default black background (RGBA)
    };

    // ========== LIGHT CONFIGURATION ==========
    float3 lightPos = {WIDTH / 2.0f, HEIGHT * 0.8f, -1000.0f}; // Light source position
    float lightWidth = WIDTH * 1.5f;                           // Area light size

    // Generate a random seed for the frame
    unsigned int frame_seed = static_cast<unsigned int>(time(nullptr));

    // Configure CUDA kernel launch parameters
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    // Launch the kernel with optimized grid and block dimensions
    renderKernel<<<gridSize, blockSize>>>(d_pixels, sphere, plane, lightPos, lightWidth, frame_seed);
    cudaDeviceSynchronize();

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy the result back to host memory
    std::vector<unsigned char> pixels(WIDTH * HEIGHT * CHANNEL_NUM);
    cudaMemcpy(pixels.data(), d_pixels, WIDTH * HEIGHT * CHANNEL_NUM, cudaMemcpyDeviceToHost);

    // ========== OUTPUT AND TIMING ==========
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Save as PNG using stb_image
    stbi_write_png("rainbow_cuda_parallel.png", WIDTH, HEIGHT, CHANNEL_NUM, pixels.data(), WIDTH * CHANNEL_NUM);

    std::cout << "CUDA ray tracing completed." << std::endl;
    std::cout << "Image dimensions: " << WIDTH << "x" << HEIGHT << std::endl;
    std::cout << "Max bounces: " << MAX_BOUNCES << std::endl;
    std::cout << "Samples per pixel: " << SAMPLES << std::endl;
    std::cout << "Execution time: " << duration << " ms" << std::endl;
    std::cout << "Output saved to rainbow_cuda_parallel.png" << std::endl;

    // Clean up
    cudaFree(d_pixels);

    return 0;
}