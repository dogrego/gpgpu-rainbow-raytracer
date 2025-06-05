#include "Seq.cuh"
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
__device__ Vec3 reflect(const Vec3 &I, const Vec3 &N)
{
    return I - N * 2.0f * N.dot(I);
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
 * Tests intersection between a ray and an infinite plane.
 * @param origin  Ray origin point in world space
 * @param dir     Ray direction vector (does not need normalization)
 * @param plane   Plane defined by normal and distance (normal·X = distance)
 * @param t       Output: Distance to intersection point if valid
 * @return true if ray intersects plane in forward direction, false otherwise
 */
__device__ bool intersectRayPlane(const Vec3 &origin, const Vec3 &dir, const Plane &plane, float &t)
{
    // Compute denominator: dot product of plane normal and ray direction
    // Represents cosine of angle between them (0 = parallel)
    float denom = plane.normal.dot(dir);

    // Check if ray and plane are not parallel (denominator > epsilon)
    if (fabsf(denom) > 1e-6)
    {
        // Calculate intersection distance t:
        // t = (plane distance - dot(normal, origin)) / denominator
        // This solves: origin + t*dir lies on plane normal·X = distance
        t = (plane.distance - plane.normal.dot(origin)) / denom;

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
        static_cast<unsigned char>(intensityMax * powf(r * factor, gamma)),
        static_cast<unsigned char>(intensityMax * powf(g * factor, gamma)),
        static_cast<unsigned char>(intensityMax * powf(b * factor, gamma)),
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
__device__ Color traceRay(const Vec3 &origin, const Vec3 &dir, const Sphere &sphere, const Plane &plane,
                          int maxBounces, float wavelength, curandState *state)
{
    // Initialize as transparent black
    Color color = {0, 0, 0, 0};
    Vec3 currentDir = dir;
    Vec3 currentPos = origin;

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
            Vec3 intersectionPoint = currentPos + currentDir * t_sphere;

            // Compute surface normal (flipped if inside sphere)
            Vec3 normal = (intersectionPoint - sphere.center).normalize();
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
            Vec3 reflected = reflect(currentDir, normal);
            Vec3 refracted = refract(currentDir, normal, n1, n2);

            // Russian roulette termination - randomly stop based on intensity
            float continueProb = fmaxf(reflectance, 1.0f - reflectance);
            if (curand_uniform(state) > continueProb)
                break;

            // Choose between reflection and refraction
            if (refracted.dot(refracted) > 0)
            { // Valid refraction
                if (curand_uniform(state) < reflectance / continueProb)
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
            Vec3 intersectionPoint = currentPos + currentDir * t_plane;

            // Convert wavelength to RGB color
            Color wavelengthColor = wavelengthToRGB(wavelength);

            // Apply accumulated intensity and set opaque
            color.r = (unsigned char)(wavelengthColor.r * accumulatedIntensity);
            color.g = (unsigned char)(wavelengthColor.g * accumulatedIntensity);
            color.b = (unsigned char)(wavelengthColor.b * accumulatedIntensity);
            color.a = 255;
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
                             const Vec3 lightPos, float lightWidth, unsigned int frame_seed)
{
    // Sequential execution - only thread 0 does all the work
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Visible light spectrum range (380nm-750nm)
        float wavelengthStart = 380.0f;
        float wavelengthEnd = 750.0f;

        curandState state;
        curand_init(frame_seed * WIDTH, 0, 0, &state);

        // ========== MAIN RENDERING LOOP ==========
        for (int y = 0; y < HEIGHT; ++y)
        {
            for (int x = 0; x < WIDTH; ++x)
            {
                Color finalColor = {0, 0, 0, 0}; // Initialize pixel color
                unsigned int seed = frame_seed + y * WIDTH + x;

                // Anti-aliasing: Stratified sampling over light source
                for (int s = 0; s < SAMPLES; ++s)
                {
                    // Jittered sampling within light area
                    float jitterX = curand_uniform(&state) - 0.5f;
                    float jitterY = curand_uniform(&state) - 0.5f;

                    Vec3 samplePos = {
                        lightPos.x + jitterX * lightWidth,
                        lightPos.y + jitterY * lightWidth,
                        lightPos.z};

                    // Calculate ray from light to current pixel
                    Vec3 pixelPos = {float(x), float(y), 0.0f};
                    Vec3 dir = (pixelPos - samplePos).normalize();

                    // Wavelength varies with vertical position (rainbow effect)
                    float wavelength = wavelengthEnd - (wavelengthEnd - wavelengthStart) * (y / float(HEIGHT));

                    // Trace ray through scene
                    Color sampleColor = traceRay(samplePos, dir, sphere, plane, MAX_BOUNCES, wavelength, &state);

                    // Accumulate color samples (averaged later)
                    finalColor.r += sampleColor.r / SAMPLES;
                    finalColor.g += sampleColor.g / SAMPLES;
                    finalColor.b += sampleColor.b / SAMPLES;
                }
                finalColor.a = 255; // Set fully opaque

                // Store final pixel in buffer (RGBA order)
                int index = (y * WIDTH + x) * CHANNEL_NUM;
                pixels[index + 0] = finalColor.r;
                pixels[index + 1] = finalColor.g;
                pixels[index + 2] = finalColor.b;
                pixels[index + 3] = finalColor.a;
            }
        }
    }
}

int main()
{
    std::cout << "Starting CUDA sequential ray tracing..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Allocate unified memory for pixels (accessible by both CPU and GPU)
    unsigned char *d_pixels;
    cudaMallocManaged(&d_pixels, WIDTH * HEIGHT * CHANNEL_NUM);

    // ========== SCENE SETUP ==========
    // Water droplet sphere (centered in scene)
    Sphere sphere = {
        {WIDTH / 2.0f, HEIGHT / 2.0f, 500.0f}, // Center position
        300.0f                                 // Radius
    };

    // Observation plane (catches rainbow rays)
    Plane plane = {
        {0.0f, 0.0f, -1.0f}, // Normal facing camera (negative Z)
        1800.0f,             // Distance from origin
        {0, 0, 0, 255}       // Default black background (RGBA)
    };

    // ========== LIGHT CONFIGURATION ==========
    Vec3 lightPos = {WIDTH / 2.0f, HEIGHT * 0.8f, -1000.0f}; // Light source position
    float lightWidth = WIDTH * 1.5f;                         // Area light size

    // Generate a random seed for the frame
    unsigned int frame_seed = static_cast<unsigned int>(time(nullptr));

    // Launch the kernel with 1 block and 1 thread (sequential)
    renderKernel<<<1, 1>>>(d_pixels, sphere, plane, lightPos, lightWidth, frame_seed);
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
    stbi_write_png("rainbow_cuda_seq.png", WIDTH, HEIGHT, CHANNEL_NUM, pixels.data(), WIDTH * CHANNEL_NUM);

    std::cout << "CUDA ray tracing completed." << std::endl;
    std::cout << "Image dimensions: " << WIDTH << "x" << HEIGHT << std::endl;
    std::cout << "Max bounces: " << MAX_BOUNCES << std::endl;
    std::cout << "Execution time: " << duration << " ms" << std::endl;
    std::cout << "Output saved to rainbow_cuda.png" << std::endl;

    // Clean up
    cudaFree(d_pixels);

    return 0;
}