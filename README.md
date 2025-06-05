# Raytrace the rainbow on the GPU

## Approach

The process is as follows:

1. **Initial Setup**  
   Assume a set of parallel rays coming from a certain direction toward a sphere of water (representing a water droplet).
2. **Ray-Sphere Intersection**  
   For each ray, compute the intersection with the sphere. At the intersection point, calculate the normal vector of the sphere.
3. **Refraction and Reflection**  
   Use **Snell's law** to calculate the relation between the *angle of incidence* and the *angle of refraction* for a given index of refraction (`n`).  
   The value of `n` depends on the wavelength of the light, meaning different colors refract at different angles.
4. **Ray Tracing**  
   Trace the rays through up to 4 intersections with the sphere or until they leave it.
5. **Virtual Screen Setup**  
   Set up a virtual screen plane on the opposite side of the sphere relative to the light source (the sun).  
   Calculate where the rays intersect the plane.
6. **Color Calculation**  
   Record the number of rays that intersect at certain pixels and their corresponding wavelengths.  
   Convolve the wavelengths with color matching functions to convert them into RGB values.
7. **Display**  
   Display the result on the virtual screen either as an image file (e.g., PNG or JPEG) using libraries such as [stb_image_write](https://github.com/nothings/stb/blob/master/stb_image_write.h) or visualize the result in real-time using OpenGL.

Use **vectorization** to follow multiple rays with each thread (CUDA) to improve performance.

## Milestones

- [x] Correct implementation of the 3D geometry and refraction/reflection computation on the CPU
- [x] Correct implementation of the 3D geometry and refraction/reflection computation on the GPU (using CUDA)
- [x] Correct implementation of a vectorized version and performance comparison with the non-vectorized one
- [x] Color display as a static image file
- [ ] Color display with OpenGL on the screen, with interactively changeable viewing angle
- [x] Compute the intensity changes properly at the intersections using the Fresnel equations assuming unpolarized light

## Benchmarks & Optimizations

The following table shows the time it takes on different hardware to calculate the refraction & reflection of 300 light vectors (380 nm to 680 nm)

| Hardware              | Execution time |
|-----------------------|----------------|
| CPU (Single Threaded) | 2168 s         |
| GPU (Non-Vectorized)  | 4229 μs        |
| GPU (Vectorized)      | 439 μs         |

By assigning each pixel to a separate GPU thread, the renderer takes full advantage of the GPU’s parallel architecture, significantly reducing render time. Additionally, choosing a wavelength-based sampling method allowed for more physically accurate color representation and spectral effects, which are often lost in traditional RGB rendering. This design also naturally fit within the SIMD/SIMT model of CUDA, where each thread handles a vector of wavelengths efficiently. These choices were made to balance realism, performance, and hardware compatibility while keeping the code modular and extensible.

- **Vectorization**: Each thread processes 4 wavelengths (`NUM_WAVELENGTHS`) to maximize register usage
- **Memory Coalescing**: `TILE_SIZE=16` ensures aligned global memory access
- **Loop Unrolling**: `#pragma unroll` in `renderKernel` reduces branch overhead

## Requirements

- Visual Studio 2022
- CMake
- NVIDIA CUDA Toolkit (12.8)
- OpenGL

## Tested On

The following generous people have aided me in testing that the program runs on a variety of different hardware configurations:

| Person         | CPU                 | GPU                         | OS                         |
|----------------|---------------------|-----------------------------|----------------------------|
| Dóra Gregorics | AMD Ryzen 5 2600X   | Nvidia GeForce RTX 2060 6GB | Windows 10 (22H2)          |
| Márton Petes   | AMD Ryzen 5 3600    | Nvidia GeForce GTX 1060 6GB | NixOS 23.11 (6.3.4)        |
| Márton Petes   | Intel Core i5-5200U | Nvidia GeForce 840M 2GB     | NixOS 23.05 (6.0.10-zen2)  |
| Zoltán Balázs  | Intel Core i7-8700K | Nvidia GeForce RTX 2080 8GB | Arch Linux (6.3.5-arch1-1) |
