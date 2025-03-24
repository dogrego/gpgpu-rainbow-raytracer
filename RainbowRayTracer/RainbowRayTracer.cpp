#include "RainbowRayTracer.h"
#include <iostream>
#include <cmath>
#include <iomanip>

Vec3 reflect(const Vec3 &I, const Vec3 &N)
{
    return I - N * 2.0f * N.dot(I);
}

bool intersectRaySphere(const Vec3& origin, const Vec3& dir, const Sphere& sphere, float& t) {
    Vec3 oc = origin - sphere.center;
    float b = 2.0f * oc.dot(dir);
    float c = oc.dot(oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.0f * c;

    if (discriminant > 0) {
        t = (-b - std::sqrt(discriminant)) / 2.0f;
        return t > 0;
    }
    return false;
}

std::ostream &operator<<(std::ostream &os, const Vec3 &v)
{
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

void runIntersectionTest(const Vec3& origin, const Vec3& dir, const Sphere& sphere, const std::string& testName) {
    float t = -1.0f;
    bool intersects = intersectRaySphere(origin, dir, sphere, t);

    std::cout << "Test: " << testName << std::endl;
    std::cout << "Origin: " << origin << std::endl;
    std::cout << "Direction: " << dir << std::endl;
    std::cout << "Sphere Center: " << sphere.center << std::endl;
    std::cout << "Sphere Radius: " << sphere.radius << std::endl;

    if (intersects) {
        std::cout << "RESULT: Intersects at t = " << std::setprecision(4) << t << std::endl;
        Vec3 intersection = { origin.x + dir.x * t, origin.y + dir.y * t, origin.z + dir.z * t };
        std::cout << "Intersection Point: " << intersection << std::endl;
    }
    else {
        std::cout << "RESULT: No intersection" << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
}

int main() {
    // Test sphere
    Sphere sphere = { {0.0f, 0.0f, 0.0f}, 1.0f };

    // Test cases
    runIntersectionTest(
        { 0.0f, 0.0f, -5.0f },  // Origin
        { 0.0f, 0.0f, 1.0f },    // Direction (pointing towards sphere)
        sphere,
        "Straight-on intersection"
    );

    runIntersectionTest(
        { 0.0f, 0.0f, 5.0f },    // Origin
        { 0.0f, 0.0f, 1.0f },    // Direction (pointing away from sphere)
        sphere,
        "Ray pointing away"
    );

    runIntersectionTest(
        { 0.0f, 2.0f, -5.0f },   // Origin
        { 0.0f, 0.0f, 1.0f },    // Direction (ray misses sphere)
        sphere,
        "Ray misses sphere"
    );

    runIntersectionTest(
        { 0.5f, 0.5f, -5.0f },   // Origin
        { 0.0f, 0.0f, 1.0f },    // Direction (glancing hit)
        sphere,
        "Glancing intersection"
    );

    runIntersectionTest(
        { 0.0f, 0.0f, 0.0f },    // Origin inside sphere
        { 0.0f, 0.0f, 1.0f },    // Direction
        sphere,
        "Origin inside sphere"
    );

    runIntersectionTest(
        { 0.0f, 0.0f, -1.0f },   // Origin on sphere surface
        { 0.0f, 0.0f, 1.0f },    // Direction
        sphere,
        "Origin on sphere surface"
    );

    return 0;
}