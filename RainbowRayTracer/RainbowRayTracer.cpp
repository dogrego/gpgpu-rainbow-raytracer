#include "RainbowRayTracer.h"

Vec3 reflect(const Vec3 &I, const Vec3 &N)
{
    return I - N * 2.0f * N.dot(I);
}

std::ostream &operator<<(std::ostream &os, const Vec3 &v)
{
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

int main()
{
    // Test 1: Simple reflection of a horizontal surface
    Vec3 incident1 = {1, -1, 0};
    Vec3 normal1 = {0, 1, 0};
    Vec3 reflected1 = reflect(incident1, normal1);

    std::cout << "Test 1 - Reflection off horizontal surface:" << std::endl;
    std::cout << "Incident: " << incident1 << std::endl;
    std::cout << "Normal:   " << normal1 << std::endl;
    std::cout << "Reflected:" << reflected1 << std::endl
              << std::endl;

    // Test 2: Reflection at 45 degree angle
    Vec3 incident2 = {1, -1, 0};
    Vec3 normal2 = {0.707f, 0.707f, 0}; // 45 degree normal
    Vec3 reflected2 = reflect(incident2, normal2);

    std::cout << "Test 2 - 45 degree reflection:" << std::endl;
    std::cout << "Incident: " << incident2 << std::endl;
    std::cout << "Normal:   " << normal2 << std::endl;
    std::cout << "Reflected:" << reflected2 << std::endl
              << std::endl;

    // Test 3: Vertical reflection
    Vec3 incident3 = {1, 0, 0};
    Vec3 normal3 = {1, 0, 0}; // Vertical normal
    Vec3 reflected3 = reflect(incident3, normal3);

    std::cout << "Test 3 - Vertical reflection:" << std::endl;
    std::cout << "Incident: " << incident3 << std::endl;
    std::cout << "Normal:   " << normal3 << std::endl;
    std::cout << "Reflected:" << reflected3 << std::endl;

    return 0;
}