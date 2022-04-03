#pragma once

#include "GLOBALS.h"
#include <skepu>
#include "Denoiser.h"
#include "Renderers.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// CUDA Include 
#include "CUDAHeader.h"

using namespace std::chrono;
typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::milli > milli_second_;
#include <chrono>

struct PrimaryFeatures{

    // Primary features, averaged when vectors

    float normal;
    float alb1;
    float alb2;
    float worldPos;
    float directLight;
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

class DenoiserNN {
public:
    
    // DenoisingInf are primary features
    // Need to generate the 36 secondary features from these primary features
    // NN generates the variances of the primary features from these secondary features

    // Backend Functions
    void ForwardProp() {
        switch(denoisingBackend) {
            case 0:
                CPUForwardProp();
                break;
            case 1:
                OMPForwardProp();
                break;
            case 2:
                CUDAForwardProp();
                break;
            case 3:
                OpenGLForwardProp();
                break;
            case 4:
                SkePUForwardProp();
                break;
            case 5:
                SkePUForwardProp();
                break;
            case 6:
                if (skipCudaDenoise)
                    denoisingSkePUBackend = "openmp";
                SkePUForwardProp();
                if (skipCudaDenoise)
                    denoisingSkePUBackend = "cuda";                
                break;
        }
    }
    void BackProp() {
        switch(denoisingBackend) {
            case 0:
                CPUBackProp();
                break;
            case 1:
                OMPBackProp();
                break;
            case 2:
                CUDABackProp();
                break;
            case 3:
                OpenGLBackProp();
                break;
            case 4:
                SkePUBackProp();
                break;
            case 5:
                SkePUBackProp();
                break;
            case 6:
                if (skipCudaDenoise)
                    denoisingSkePUBackend = "openmp";
                SkePUBackProp();
                if (skipCudaDenoise)
                    denoisingSkePUBackend = "cuda";                
                break;
        }
    }

    // Forward Prop
    void CPUForwardProp();
    void OMPForwardProp();
    void CUDAForwardProp(){ CUDADenoiserNN::ForwardProp(); }
    void OpenGLForwardProp(){}
    void SkePUForwardProp();

    // Back Prop
    vec3 CPUFilterDerivative(int j, int i, int var);
    void CPUBackProp();
    vec3 OMPFilterDerivative(int j, int i, int var);
    void OMPBackProp();
    void CUDABackProp(){ CUDADenoiserNN::BackProp();  }
    void OpenGLBackProp(){}
    void SkePUBackProp();

    void TrainNN();

    void GenRelMSE();
    void RandomizeWeights();
    void OutputWeights(std::string name);
    bool LoadWeights(std::string name);
    void InitTraining();
    void EndTraining();
    void AppendTrainingFile();

    DenoiserNN() {
        pFeatures = new PrimaryFeatures[xRes*yRes];
        sFeatures = new SecondaryFeatures[xRes*yRes];

        layerTwoValues = new float[10*xRes*yRes];
        layerThreeValues = new float[10*xRes*yRes];
    }

    PrimaryFeatures* pFeatures;
    SecondaryFeatures* sFeatures;

    // Neural Net

        // Layer 1: 36 Inputs 
        // Layer 2: 10 nodes, hidden layer
        // Layer 3: 10 nodes, hidden layer
        // Layer 4: 7 Outputs

        // Weights: (1-2): 360, (2-3): 100, (3-4): 70
        float onetwo[360];
        // Layer two vals in GLOBALS
        float twothree[100];
        // Layer three vals in GLOBALS
        float threefour[80];

        float learningRate = 0.0001f;
        int samplesWhenTraining = 4;

    // Neural Net

    float relMSE;
};
