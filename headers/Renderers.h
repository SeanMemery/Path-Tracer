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

using namespace std::chrono;
typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::milli > milli_second_;
#include <chrono>

struct Constants {
    float camPos[3], camForward[3], camRight[3], camUp[3];
	float maxAngle;
	int numShapes;
	int RESV, RESH;
	float maxAngleV, maxAngleH, focalLength;

	uint64_t GloRandS[2];

	float backgroundColour[3];

	float shapes[20][16];

	int importantShapes[5];
	uint numImportantShapes;
	uint getDenoiserInf;
};

struct ReturnStruct {
	float xyz[3];
	float normal[3];
	float albedo1[3];
	float albedo2[3];
	float worldPos[3];
	float directLight;
	uint raysSent;
};

struct RandomSeeds {
	long s1;
	long s2;
};

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
        totalTime += renderTime;

    }

    void CPURender();
    void OMPRender();
    void CUDARender();
    void OpenGLRender();
    void SkePURender();

    void UpdateConstants();

    Constants constants;

};