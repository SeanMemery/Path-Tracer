#pragma once

#include <string>
#include "vec3.h"
#include "ext/imgui.h"
#include <vector>

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
extern double denoiseTime, epochTime, exposureTime;

// Total times
extern double renderTime, imguiTime, postProcessTime, totalTime, screenUpdateTime, totalRenderTime;

extern int rootThreadsPerBlock;

// Post Processing 
extern float exposure, g;
extern int displayMetric;

// Denoising
extern DenoiserNN denoiserNN;
extern int denoisingN, trainingEpoch, denoisingBackend;
extern std::string denoisingSkePUBackend;
extern bool training, weightsLoaded, skipCudaDenoise;

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

struct Constants {
    float camPos[3], camForward[3], camRight[3], camUp[3];
	float maxAngle, randSamp;
	int numShapes, maxDepth;
	int RESV, RESH;
	float maxAngleV, maxAngleH, focalLength;

	int shapes[50][3];
    float objAttributes[450];
    float matList[50][6];

	uint importantShapes[10];
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

extern Constants constants;

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

struct ForPropIn {

    float normal;
    float alb1;
    float alb2;
    float worldPos;
    float directLight;
    float stdDev[6];

};
struct ForPropOut {

    float l2[10];
    float l3[10];
    float variances[7];

    // Secondary Features
    float meansSingle[5];
    float sdSingle[5];
    float meansBlock[5];
    float sdBlock[5];
    float gradients[5];
    float meanDeviation[5];
    float MAD[5];
    float L;

};
struct SkePUFPConstants {

    int samples;
    float onetwo[360];
    float twothree[100];
    float threefour[70];

};
struct FilterDerivIn {
    float preScreen[3];
    float normal[3];
    float alb1[3];
    float alb2[3];
    float worldPos[3];
    float denoisedCol[3];
    float directLight;
    float stdDev[6];
    float variances[7];
    float wcSum;
};
struct FilterDerivOut {
    float paramXYZ[7][3];
};
struct SkePUBPIn {

    float targetCol[3];
    float denoisedCol[3];
    FilterDerivOut deriv;

    float s[36];
    float l2[10];
    float l3[10];

};
struct SkePUBPOut {

    float onetwo[360];
    float twothree[100];
    float threefour[70];

};

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




