#pragma once

#include "GLOBALS.h"

namespace CUDADenoiserNN {

    void ForwardProp();
    void BackProp();

};

namespace CUDADenoiser {

    void denoise();

};

class float3;

namespace CUDARender {

    static Constants* CUDAConstants;
    static RandomSeeds*  CUDASeeds;
    static ReturnStruct* CUDAReturn;
    static float3* CUDAPostScreen;
    static float3* CUDADisplay;

    void render();
    void PostProcess();
    void UpdateConstants();

};