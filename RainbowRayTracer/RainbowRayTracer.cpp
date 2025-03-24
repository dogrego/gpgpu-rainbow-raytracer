#include "RainbowRayTracer.h"
#include <iostream>
#include <cmath>

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
    // Quick visual check of key colors
    std::cout << "Quick Color Check:\n";
    std::cout << "380nm (Violet): ";
    std::cout << static_cast<int>(wavelengthToRGB(380).r) << ","
        << static_cast<int>(wavelengthToRGB(380).g) << ","
        << static_cast<int>(wavelengthToRGB(380).b) << "\n";

    std::cout << "470nm (Blue):   ";
    std::cout << static_cast<int>(wavelengthToRGB(470).r) << ","
        << static_cast<int>(wavelengthToRGB(470).g) << ","
        << static_cast<int>(wavelengthToRGB(470).b) << "\n";

    std::cout << "520nm (Green):  ";
    std::cout << static_cast<int>(wavelengthToRGB(520).r) << ","
        << static_cast<int>(wavelengthToRGB(520).g) << ","
        << static_cast<int>(wavelengthToRGB(520).b) << "\n";

    std::cout << "580nm (Yellow): ";
    std::cout << static_cast<int>(wavelengthToRGB(580).r) << ","
        << static_cast<int>(wavelengthToRGB(580).g) << ","
        << static_cast<int>(wavelengthToRGB(580).b) << "\n";

    std::cout << "650nm (Red):    ";
    std::cout << static_cast<int>(wavelengthToRGB(650).r) << ","
        << static_cast<int>(wavelengthToRGB(650).g) << ","
        << static_cast<int>(wavelengthToRGB(650).b) << "\n";

    // Check invisible wavelengths
    std::cout << "\nInvisible Check:\n";
    std::cout << "300nm (UV):     ";
    std::cout << static_cast<int>(wavelengthToRGB(300).r) << ","
        << static_cast<int>(wavelengthToRGB(300).g) << ","
        << static_cast<int>(wavelengthToRGB(300).b) << "\n";

    std::cout << "800nm (IR):     ";
    std::cout << static_cast<int>(wavelengthToRGB(800).r) << ","
        << static_cast<int>(wavelengthToRGB(800).g) << ","
        << static_cast<int>(wavelengthToRGB(800).b) << "\n";

    return 0;
}