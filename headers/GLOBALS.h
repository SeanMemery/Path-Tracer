#pragma once

#include <string>
#include "vec3.h"
#include "ext/imgui.h"

class Camera;
class Denoiser;
class Renderers;
class DenoiserNN;
class Scene;

// Settings
extern int xRes, yRes, xScreen, yScreen, maxDepth, currentRenderer, rayCount, sampleCount;
extern bool denoising, moving, quit, rendering, refresh;
extern unsigned int mainTexture; 
extern std::string skepuBackend;
extern double renderTime, denoiseTime, epochTime, totalTime;

// Post Processing 
extern float exposure, g;
extern int displayMetric;

// Denoising
extern DenoiserNN denoiserNN;
extern int denoisingN, trainingEpoch, denoisingBackend;
extern std::string denoisingSkePUBackend;
extern bool training, weightsLoaded;

// Objects
extern Scene scene;
extern ImGuiWindowFlags window_flags;
extern Camera cam;
extern Denoiser denoiser;
extern Renderers renderer;

// Screens
extern vec3* preScreen;
extern vec3* postScreen;

// Denoising Screens
extern vec3* normal;
extern vec3* albedo1;
extern vec3* albedo2;
extern vec3* directLight;
extern vec3* worldPos;

extern vec3* denoisedCol;
extern vec3* targetCol;

struct DenoisingInf {
    vec3 stdDevVecs[6];
    float stdDev[6];
    float variances[7];
    float wcSum;

    DenoisingInf() {
        int c = 0;
        for (c=0; c<6; c++)
            stdDevVecs[c] = vec3();
        for (c=0; c<7; c++) {
            stdDev[c] = 0.0f;
            variances[c] = 0.0f;
        }
        wcSum = 0.0f;
    }
};

extern DenoisingInf* denoisingInf;




