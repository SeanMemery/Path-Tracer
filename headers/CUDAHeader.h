#pragma once

#include "GLOBALS.h"

namespace CUDADenoiserNN {

    void ForwardProp();
    void BackProp();
    void InitBuffers();
    void FreeBuffers();

    static ForPropIn* CUDAFPIn;
    static ForPropOut* CUDAFPOut;
    static FPConstants* CUDAFPConstants;

    static FilterDerivIn*  CUDAFIn;
    static FilterDerivOut* CUDAFOut;
    static BPConstants* CUDAConstants;
    static SkePUBPIn*  CUDAIn;
    static SkePUBPOut* CUDAOut;

};

namespace CUDADenoiser {

    void denoise();
    void InitBuffers();
    void FreeBuffers();

    static GPUInf* CUDAIn;
    static FilterVals* CUDAOut;
    static CUDADenoiseConstants* CUDAConstants;
};

class float3;

namespace CUDARender {

    static Constants* CUDAConstants;
    static RandomSeeds*  CUDASeeds;
    static ReturnStruct* CUDAReturn;
    static float3* CUDAPostScreen;
    static float3* CUDADisplay;

    static float* gpuExposure;

    void render();
    void PostProcess();
    void UpdateConstants();
    void CUDAAutoExp();

};