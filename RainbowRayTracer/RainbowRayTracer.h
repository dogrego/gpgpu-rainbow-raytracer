﻿#ifndef RAINBOWRAYTRACER_H
#define RAINBOWRAYTRACER_H

#include <iostream>

// Vector structure for 3D operations
struct Vec3
{
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    Vec3 operator/(float scalar) const { return Vec3(x / scalar, y / scalar, z / scalar); }

    float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
    float lengthSquared() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(lengthSquared()); }
    Vec3 normalize() const { return *this * (1.0f / length()); }
};

// Color
struct Color
{
    unsigned char r, g, b, a;
};

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
Color wavelengthToRGB(float wavelength);
Vec3 refract(const Vec3 &incident, const Vec3 &normal, float n1, float n2);
float fresnel(const Vec3 &incident, const Vec3 &normal, float n1, float n2);
float wavelengthToRefraction(float wavelength);
Color traceRay(const Vec3 &origin, const Vec3 &dir, const Sphere &sphere, const Plane &plane, int maxBounces, float wavelength);

#endif // RAINBOWRAYTRACER_H