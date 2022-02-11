#pragma once

#include "GLOBALS.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// CUDA Include 
#include "CUDAHeader.h"

#include <chrono>
using namespace std::chrono;
typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::milli > milli_second_;

class DenoiserNN {
public:
    

    void TrainNN();

    void saveTargetCol();
    void GenRelMSE();
    void RandomizeWeights();
    void OutputWeights(std::string name);
    bool LoadWeights(std::string name);
    void InitTraining();
    void EndTraining();
    void AppendTrainingFile();

    DenoiserNN() {
        sFeatures = new SecondaryFeatures[xRes*yRes];
        layerTwoValues = new float[10*xRes*yRes];
        layerThreeValues = new float[10*xRes*yRes];
    }

    float relMSE;
};
