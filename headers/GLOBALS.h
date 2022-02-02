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
extern float randSamp;
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
extern float* layerTwoValues; // 10 vals per pixel
extern float* layerThreeValues; // 10 vals per pixel

namespace GLOBALS {

    static void DeleteScreens(bool delTarget) {
        delete preScreen   ;
        delete postScreen  ;
        delete normal      ;
        delete albedo1     ;
        delete albedo2     ;
        delete directLight ;
        delete worldPos    ;
        delete denoisedCol ;
        delete denoisingInf;

        if (delTarget)
            delete targetCol;

        delete layerTwoValues;
        delete layerThreeValues;
    }

    static void InitScreens(bool initTarget) {
        preScreen      = new vec3[xRes*yRes];
        postScreen     = new vec3[xRes*yRes];
        normal         = new vec3[xRes*yRes];
        albedo1        = new vec3[xRes*yRes];
        albedo2        = new vec3[xRes*yRes];
        directLight    = new vec3[xRes*yRes];
        worldPos       = new vec3[xRes*yRes];
        denoisedCol    = new vec3[xRes*yRes];
        denoisingInf   = new DenoisingInf[xRes*yRes];

        if (initTarget)
            targetCol = new vec3[xRes*yRes];

        layerTwoValues = new float[10*xRes*yRes];
        layerThreeValues = new float[10*xRes*yRes];
    }
};




