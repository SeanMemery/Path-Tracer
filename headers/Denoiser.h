#pragma once

#include "GLOBALS.h"
#include "vec3.h"
#include <skepu>
#include "DenoiserNN.h"

// CUDA Include 
#include "CUDAHeader.h"

using namespace std::chrono;
typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::milli > milli_second_;
#include <chrono>

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
    void saveTargetCol();
};