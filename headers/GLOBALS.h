#pragma once

#include <string>
#include "Scene.h"

class Camera;
class Denoiser;

// Settings
extern int xRes, yRes, xScreen, yScreen, maxDepth, currentRenderer, rayCount, sampleCount;
extern bool denoising, moving, quit, rendering, refresh;
extern unsigned int mainTexture; 
extern std::string skepuBackend;
extern double renderTime;

// Post Processing 
extern float exposure, g;
extern int displayMetric;

// Objects
extern Scene scene;
extern ImGuiWindowFlags window_flags;
extern Camera cam;
extern Denoiser denoiser;

// Screens
extern vec3* preScreen;
extern vec3* postScreen;




