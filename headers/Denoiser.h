#pragma once

#include "GLOBALS.h"
#include "vec3.h"


struct DenoisingInf {
    vec3 finalCol;
	vec3 normal;
	vec3 firstBounce;
	vec3 albedo1;
    vec3 albedo2;
    vec3 prevCol;
    vec3 worldPos;
    vec3 denoisedCol;
    vec3 stdDevVecs[6] = {vec3(), vec3(), vec3(), vec3(), vec3(), vec3()};
    float stdDevs[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float variances[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float depth, directLight;

    float wcSum;
};

struct GPUInf {
    float col[3] = {0.0f, 0.0f, 0.0f};
	float normal[3] = {0.0f, 0.0f, 0.0f};
	float albedo1[3] = {0.0f, 0.0f, 0.0f};
    float albedo2[3] = {0.0f, 0.0f, 0.0f};
    float worldPos[3] = {0.0f, 0.0f, 0.0f};
    float directLight = 0.0f;
    float stdDevs[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float variances[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
};

struct FilterVals {
    float x;
    float y;
    float z;
    float wcSum;
};

class Denoiser {
public:

    Denoiser() {
        initDenoisingInf();
    }

    void initDenoisingInf() {
        auto numPixels = xRes*yRes;

        delete denoisingInf;
        denoisingInf = new DenoisingInf[numPixels];  

        delete targetCol;
        targetCol = new vec3[numPixels];
    }

    void denoise() {
        switch(currentRenderer) {
            case 0:
                CPUDenoise();
                break;
            case 1:
                OMPDenoise();
                break;
            case 2:
                CUDADenoise();
                break;
            case 3:
                OpenGLDenoise();
                break;
            case 4:
                SkePUDenoise();
                break;
            case 5:
                SkePUDenoise();
                break;
            case 6:
                SkePUDenoise();
                break;
        }
    }

    void CPUDenoise();
    void OMPDenoise();
    void CUDADenoise();
    void OpenGLDenoise();
    void SkePUDenoise();

    DenoisingInf* denoisingInf;
    vec3* targetCol;
};