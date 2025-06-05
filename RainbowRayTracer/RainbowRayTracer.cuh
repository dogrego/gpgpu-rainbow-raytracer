#ifndef RAINBOWRAYTRACER_CUH
#define RAINBOWRAYTRACER_CUH

#include <cuda_runtime.h>

// float3
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float scalar) {
    return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__host__ __device__ inline float3 operator*(float scalar, const float3& a) {
    return a * scalar;
}

__host__ __device__ inline float3 operator/(const float3& a, float scalar) {
    float inv = 1.0f / scalar;
    return a * inv;
}

__host__ __device__ inline float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline float lengthSquared(const float3& v) {
    return dot(v, v);
}

__host__ __device__ inline float3 normalize(const float3& v) {
    return v / length(v);
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

struct Color {
    float4 rgba;

    __device__ Color() {
        rgba = make_float4(0.f, 0.f, 0.f, 255.f);
    }

    __device__ Color(float r, float g, float b, float a) {
        rgba = make_float4(r, g, b, a);
    }

    __device__ Color operator+(const Color& other) const {
        return Color(
            rgba.x + other.rgba.x,
            rgba.y + other.rgba.y,
            rgba.z + other.rgba.z,
            rgba.z + other.rgba.z
        );
    }

    __device__ Color operator*(float scalar) const {
        return Color(
            rgba.x * scalar,
            rgba.y * scalar,
            rgba.z * scalar,
            rgba.w * scalar
        );
    }

    __device__ Color& operator+=(const Color& other) {
        rgba.x += other.rgba.x;
        rgba.y += other.rgba.y;
        rgba.z += other.rgba.z;
        rgba.z += other.rgba.z;
        return *this;
    }
};

// Sphere structure with aligned memory
struct alignas(16) Sphere {
    float3 center;
    float radius;
};

// Plane structure with aligned memory
struct alignas(16) Plane {
    float3 normal;
    float distance;
    Color color;
};

// Device functions declarations
__device__ float3 reflect(const float3& I, const float3& N);
__device__ float3 refract(const float3& incident, const float3& normal, float n1, float n2);
__device__ float fresnel(const float3& incident, const float3& normal, float n1, float n2);
__device__ bool intersectRaySphere(const float3& origin, const float3& dir, const Sphere& sphere, float& t);
__device__ bool intersectRayPlane(const float3& origin, const float3& dir, const Plane& plane, float& t);
__device__ float wavelengthToRefraction(float wavelength);
__device__ Color wavelengthToRGB(float wavelength);
__device__ float getAbsorptionCoefficient(float wavelength);
__device__ Color traceRay(const float3& origin, const float3& dir, const Sphere& sphere, const Plane& plane,
    int maxBounces, float wavelength, unsigned int& seed);

#endif // RAINBOWRAYTRACER_CUH