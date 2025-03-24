#include "RainbowRayTracer.h"
#include <iostream>
#include <cmath>

Vec3 reflect(const Vec3 &I, const Vec3 &N)
{
    return I - N * 2.0f * N.dot(I);
}

Vec3 refract(const Vec3& incident, const Vec3& normal, float n1, float n2)
{
    float ratio = n1 / n2;                                               // Refractive index ratio
    float cosI = -std::max(-1.0f, std::min(1.0f, incident.dot(normal))); // Clamp the dot product to [-1, 1]
    float sinT2 = ratio * ratio * (1.0f - cosI * cosI);                  // Sine squared of the refraction angle

    if (sinT2 > 1.0f)
        return Vec3(0, 0, 0); // Total internal reflection: return zero vector

    float cosT = std::sqrt(1.0f - sinT2);                     // Cosine of the refraction angle
    return incident * ratio + normal * (ratio * cosI - cosT); // Refraction direction
}

float fresnel(const Vec3& incident, const Vec3& normal, float n1, float n2)
{
    float cosI = -std::max(-1.0f, std::min(1.0f, incident.dot(normal))); // Cosine of the angle of incidence
    float sinT2 = (n1 / n2) * (n1 / n2) * (1.0f - cosI * cosI);          // Sine squared of the angle of refraction

    if (sinT2 > 1.0f)
        return 1.0f; // Total internal reflection: return maximum reflectance

    float cosT = std::sqrt(1.0f - sinT2);                                             // Cosine of the angle of refraction
    float rParallel = ((n2 * cosI) - (n1 * cosT)) / ((n2 * cosI) + (n1 * cosT));      // Reflection for parallel polarization
    float rPerpendicular = ((n1 * cosI) - (n2 * cosT)) / ((n1 * cosI) + (n2 * cosT)); // Reflection for perpendicular polarization

    return (rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f; // Average reflection
}

bool intersectRaySphere(const Vec3& origin, const Vec3& dir, const Sphere& sphere, float& t)
{
    Vec3 oc = origin - sphere.center;                     // Vector from ray origin to sphere center
    float b = 2.0f * oc.dot(dir);                         // Dot product term for the quadratic equation
    float c = oc.dot(oc) - sphere.radius * sphere.radius; // Constant term for the quadratic equation
    float discriminant = b * b - 4.0f * c;                // Discriminant of the quadratic equation

    // If the discriminant is positive, there are real intersections
    if (discriminant > 0)
    {
        t = (-b - std::sqrt(discriminant)) / 2.0f; // Calculate the intersection distance
        return t > 0;                              // If t is positive, the intersection occurs in the direction of the ray
    }

    // No intersection
    return false;
}

float wavelengthToRefraction(float wavelength)
{
    // a simplified empirical approximation of the Sellmeier equation
    return 1.31477 + 0.0108148 / (std::log10(0.00690246 * wavelength));
}

Color wavelengthToRGB(float wavelength)
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
        255 };
}

std::ostream &operator<<(std::ostream &os, const Vec3 &v)
{
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

int main() {


    return 0;
}