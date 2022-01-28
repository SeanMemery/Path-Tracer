#pragma once

#include <cmath>

class vec3 {
public:
    float x, y, z;

    vec3(float _x,float _y, float _z) : x(_x), y(_y), z(_z) {}
    vec3() : x(0), y(0), z(0) {}
    
    float operator[](int ind) {
        switch(ind) {
            case 0:
                return x;
                break;
            case 1:
                return y;
                break;
            case 2:
                return z;
                break;
        }
        return 0;
    }
    void operator+=(vec3 eq) {
        x += eq.x;
        y += eq.y;
        z += eq.z;
    }
    void operator-=(vec3 eq) {
        x -= eq.x;
        y -= eq.y;
        z -= eq.z;
    }
    void operator/=(float f) {
        x /= f;
        y /= f;
        z /= f;
    }
    void operator*=(vec3 v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
    }
    vec3 operator*(float f) {
        return vec3(x*f, y*f, z*f);
    }
    vec3 operator+(vec3 r) {
        return vec3(x+r.x, y+r.y, z+r.z);
    }
    vec3 operator/(float f) {
        return vec3(x/f, y/f, z/f);
    }
    vec3 operator-(){
        return vec3(-x, -y, -z);
    }
    vec3 operator-(vec3 r) {
        return vec3(x-r.x, y-r.y, z-r.z);
    }

    float dot(vec3 d) {
        return x*d.x + y*d.y + z*d.z;
    }
    float square() {
        return (x*x + y*y + z*z);
    }
    vec3 normalize() {
        float sum = sqrt(square());
        x /= sum;
        y /= sum;
        z /= sum;
        return *this;
    }
    vec3 cross(vec3 c) {
        return vec3(
			y * c.z - c.y * z,
			z * c.x - c.z * x,
			x * c.y - c.x * y);
    }
    float sumDiv(float div) {
        return (x+y+z)/div;
    }

};
