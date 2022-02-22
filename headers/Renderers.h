#pragma once

// Holds all rendering algorithms: CPU, OpenMP, CUDA, OpenGL, SkePU
//     - Must handle converting scene to gpu inputs

// CPU: 0, OMP: 1, CUDA: 2, OpenGL: 3, SkePU CPU: 4, SkepPU OMP, SkePU CUDA: 6 

#include "GLOBALS.h"

#include "Denoiser.h"
#include "Camera.h"
#include "Scene.h"
#include <skepu>
#include <algorithm>

// CUDA Heaeder File
#include "CUDAHeader.h"

using namespace std::chrono;
typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::milli > milli_second_;
#include <chrono>

class Renderers {
public:

	Renderers() {constants = Constants();}

	void Render() {

        auto renderTimer = clock_::now();

        switch(currentRenderer) {
            case 0:
                CPURender();
                break;
            case 1:
                OMPRender();
                break;
            case 2:
                CUDARender();
                break;
            case 3:
                OpenGLRender();
                break;
            case 4:
                SkePURender();
                break;
            case 5:
                SkePURender();
                break;
            case 6:
                SkePURender();
                break;
        }

        renderTime = std::chrono::duration_cast<milli_second_>(clock_::now() - renderTimer).count();
    }

    void CPURender();
    void OMPRender();
    void CUDARender();
    void OpenGLRender();
    void SkePURender();

    void UpdateConstants();
    void UpdateCam(bool c = true);
    

    void AutoExposure() {

        auto exposureTImer = clock_::now();

        switch(currentRenderer) {
            case 0:
                CPUAutoExp();
                break;
            case 1:
                OMPAutoExp();
                break;
            case 2:
                CUDAAutoExp();
                break;
            case 3:
                OpenGLAutoExp();
                break;
            case 4:
                SkePUAutoExp();
                break;
            case 5:
                SkePUAutoExp();
                break;
            case 6:
                SkePUAutoExp();
                break;
        }

        exposureTime = std::chrono::duration_cast<milli_second_>(clock_::now() - exposureTImer).count();
    }

    void CPUAutoExp();
    void OMPAutoExp();
    void CUDAAutoExp(){}
    void OpenGLAutoExp(){}
    void SkePUAutoExp();

    uint64_t GloRandS[2];
};