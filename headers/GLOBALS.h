#pragma once

#include <string>
#include "Camera.h"
#include "Scene.h"

extern int xRes, yRes, maxDepth, currentRenderer, rayCount, sampleCount;
extern bool denoising, moving, quit, rendering, refresh;
extern unsigned int mainTexture; 
extern std::string skepuBackend;

extern Scene scene;
extern ImGuiWindowFlags window_flags;
extern Camera cam;
extern vec3* screen;




