#pragma once

// Holds all rendering algorithms: CPU, OpenMP, CUDA, OpenGL, SkePU
//     - Must handle converting scene to gpu inputs

// CPU: 0, OMP: 1, CUDA: 2, OpenGL: 3, SkePU CPU: 4, SkepPU OMP, SkePU CUDA: 6 

#include "GLOBALS.h"
#include "Scene.h"
#include <skepu>

#include <algorithm>

// For Rands
using namespace std::chrono;
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
	float firstBounce[3];
	float albedo1[3];
	float albedo2[3];
	float worldPos[3];
	float directLight;
	float depth;
	uint raysSent;
};

struct RandomSeeds {
	long s1;
	long s2;
};

class Renderers {
public:

	Renderers() {constants = Constants();}

    void Render();

    void CPURender();
    void OMPRender();
    void CUDARender();
    void OpenGLRender();
    void SkePURender();

    void UpdateConstants();

    Constants constants;

};