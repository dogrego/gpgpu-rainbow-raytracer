#ifndef RAINBOWRAYTRACER_CUH
#define RAINBOWRAYTRACER_CUH

#include <iostream>
#include <cuda_runtime.h>

// Vector structure for 3D operations
struct Vec3
{
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(float scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    __host__ __device__ Vec3 operator/(float scalar) const { return Vec3(x / scalar, y / scalar, z / scalar); }

    __host__ __device__ float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ float lengthSquared() const { return x * x + y * y + z * z; }
    __host__ __device__ float length() const { return sqrtf(lengthSquared()); }
    __host__ __device__ Vec3 normalize() const { return *this * (1.0f / length()); }
};

// Color
struct Color
{
    unsigned char r, g, b, a;
};

//// Vectorized Color structure
// struct Color {
//     unsigned char r, g, b, a;
//
//     __device__ Color operator+(const Color& other) const {
//         return {
//             static_cast<unsigned char>(r + other.r),
//             static_cast<unsigned char>(g + other.g),
//             static_cast<unsigned char>(b + other.b),
//             255
//         };
//     }
//
//     __device__ Color operator*(float scalar) const {
//         return {
//             static_cast<unsigned char>(r * scalar),
//             static_cast<unsigned char>(g * scalar),
//             static_cast<unsigned char>(b * scalar),
//             255
//         };
//     }
// };

// Sphere
struct Sphere
{
    Vec3 center;
    float radius;
};

// Plane
struct Plane
{
    Vec3 normal;
    float distance;
    Color color;
};

Vec3 reflect(const Vec3 &I, const Vec3 &N);
bool intersectRaySphere(const Vec3 &origin, const Vec3 &dir, const Sphere &sphere, float &t);
Vec3 refract(const Vec3 &incident, const Vec3 &normal, float n1, float n2);
float fresnel(const Vec3 &incident, const Vec3 &normal, float n1, float n2);
float wavelengthToRefraction(float wavelength);
Color traceRay(const Vec3 &origin, const Vec3 &dir, const Sphere &sphere, const Plane &plane, int maxBounces, float wavelength);

#endif // RAINBOWRAYTRACER_CUH
