#pragma once

#include "GLOBALS.h"
#include "vec3.h"
#include <skepu>
#include "DenoiserNN.h"

using namespace std::chrono;
typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::milli > milli_second_;
#include <chrono>

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

    Denoiser() {}

    void denoise();

    void CPUDenoise();
    void OMPDenoise();
    void CUDADenoise();
    void OpenGLDenoise();
    void SkePUDenoise();

    static FilterVals SkePUFilter(skepu::Region2D<GPUInf> r);
    float getMSE();
    void saveTargetCol();
};