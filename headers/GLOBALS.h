#pragma once

#include <string>
#include "vec3.h"
#include "ext/imgui.h"
#include <vector>


class Camera;
class Denoiser;
class DenoiserNN;
class Scene;
class ManageConstants;

// Settings
extern int xRes, yRes, xScreen, yScreen, maxDepth, rayCount, sampleCount;
extern bool denoising, moving, quit, rendering, refresh;
extern unsigned int mainTexture; 
extern float randSamp;
extern double renderTime, denoiseTime, epochTime, totalTime;
extern uint64_t GloRandS[2];

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
extern ManageConstants mConstants;

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

struct SecondaryFeatures {

    // K Features: normal, alb1, alb2, worldPos, directLight

    // - K Means of single pixel, K std deviations of single pixel, K Means of 7x7 block, K std deviations of 7x7 block (20 total)
	// - Magnitude of gradients of K features of single pixel (sobel operator) (5 total)
	// - Sum of the abs difference between K of each pixel in 3x3 block and the mean of that 3x3 block (5 total)
	// - MAD of K features, so median of values minus median value, in NxN block (5 total)
	// - 1/totalSamples (1 total)

    float meansSingle[5];
    float sdSingle[5];
    float meansBlock[5];
    float sdBlock[5];
    float gradients[5];
    float meanDeviation[5];
    float MAD[5];
    float L;

    float operator()(int index) {
        switch(index) {
            case 0: return meansSingle[0]; break;
            case 1: return meansSingle[1]; break;
            case 2: return meansSingle[2]; break;
            case 3: return meansSingle[3]; break;
            case 4: return meansSingle[4]; break;
            case 5: return sdSingle[0]; break;
            case 6: return sdSingle[1]; break;
            case 7: return sdSingle[2]; break;
            case 8: return sdSingle[3]; break;
            case 9: return sdSingle[4]; break;
            case 10: return meansBlock[0]; break;
            case 11: return meansBlock[1]; break;
            case 12: return meansBlock[2]; break;
            case 13: return meansBlock[3]; break;
            case 14: return meansBlock[4]; break;
            case 15: return sdBlock[0]; break;
            case 16: return sdBlock[1]; break;
            case 17: return sdBlock[2]; break;
            case 18: return sdBlock[3]; break;
            case 19: return sdBlock[4]; break;
            case 20: return gradients[0]; break;
            case 21: return gradients[1]; break;
            case 22: return gradients[2]; break;
            case 23: return gradients[3]; break;
            case 24: return gradients[4]; break;
            case 25: return meanDeviation[0]; break;
            case 26: return meanDeviation[1]; break;
            case 27: return meanDeviation[2]; break;
            case 28: return meanDeviation[3]; break;
            case 29: return meanDeviation[4]; break;
            case 30: return MAD[0]; break;
            case 31: return MAD[1]; break;
            case 32: return MAD[2]; break;
            case 33: return MAD[3]; break;
            case 34: return MAD[4]; break;
            case 35: return L; break;
        }
        return 0.0f;
    }

};

extern DenoisingInf* denoisingInf;
extern float* layerTwoValues; // 10 vals per pixel
extern float* layerThreeValues; // 10 vals per pixel

extern SecondaryFeatures* sFeatures;
    // Neural Net

        // Layer 1: 36 Inputs 
        // Layer 2: 10 nodes, hidden layer
        // Layer 3: 10 nodes, hidden layer
        // Layer 4: 7 Outputs

        // Weights: (1-2): 360, (2-3): 100, (3-4): 70
        extern float onetwo[360];
        // Layer two vals in GLOBALS
        extern float twothree[100];
        // Layer three vals in GLOBALS
        extern float threefour[80];

        extern float learningRate;
        extern int samplesWhenTraining;

    // Neural Net
struct Constants {
    float camPos[3], camForward[3], camRight[3], camUp[3];
	float maxAngle, randSamp;
	int numShapes, maxDepth;
	int RESV, RESH;
	float maxAngleV, maxAngleH, focalLength;

	int shapes[50][3];
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

// Value Buffers
extern std::vector<uint> vertexIndices;
extern std::vector<float> vertices;
extern std::vector<float> objAttributes;

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




