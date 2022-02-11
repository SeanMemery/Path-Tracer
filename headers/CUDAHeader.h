#pragma once

#include "GLOBALS.h"

namespace CUDADenoiserNN {

    void ForwardProp();
    void BackProp();

};

namespace CUDADenoiser {

    void denoise();

};

namespace CUDARender {

    void render();
    void UpdateConstants();
    void UpdateCam();

    static uint*      CUDAVertexIndices;
    static float*     CUDAVertices;
    static float*     CUDAObjAttributes;
    static Constants* CUDAConstants;

};