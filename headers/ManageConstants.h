#pragma once

#include "GLOBALS.h"
#include "Camera.h"
#include "Scene.h"
#include <algorithm>

// CUDA Heaeder File
#include "CUDAHeader.h"

#include <chrono>
using namespace std::chrono;
typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::milli > milli_second_;


class ManageConstants {
public:

	ManageConstants() {constants = Constants();}
    void UpdateConstants();
    void UpdateCam();

};