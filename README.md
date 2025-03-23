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

- [ ] Correct implementation of the 3D geometry and refraction/reflection computation on the CPU
- [ ] Correct implementation of the 3D geometry and refraction/reflection computation on the GPU (using CUDA)
- [ ] Correct implementation of a vectorized version and performance comparison with the non-vectorized one
- [ ] Color display as a static image file
- [ ] Color display with OpenGL on the screen, with interactively changeable viewing angle

## Requirements

- Visual Studio 2022
- NVIDIA CUDA Toolkit (12.8)
- OpenGL
