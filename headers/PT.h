#pragma once

#include "GLOBALS.h"
#include "vec3.h"
#include "Renderers.h"
#include "Scene.h"
#include "Obj.h"
#include "Mat.h"
#include "Camera.h"
#include "Denoiser.h"

#include "ext/imgui.h"
#include "ext/imgui_impl_sdl.h"
#include "ext/imgui_impl_opengl3.h"
#include <SDL/SDL.h>
#include <GL/glew.h>

#include <sstream>
#include <fstream>

class PT {
public:

    PT();

    void RenderLoop();
    void ImGui();
    void ProcessInput();
    void PostProcess();

    void SaveImage(char * name);
    void SwitchRenderModes();

    Renderers renderer;

    // -1: Regular View, 0 Target Col, 1: denoised image, 2: image, 3: normal, 4: albedo1, 5:albedo2, 6: firstBounce, 7: depth, 8: directLight, 9: worldPos
    const char* displayNames[11] = {"None", "Target Col", "Denoised Col", "Image", "Normal", "Albedo 1", "Albedo 2", "First Bounce", "Depth", "Direct Light", "World Pos"};

    // ImGui Vars
    int objEdit;
    char fileName[32];
    float resPerc, screenPerc;
    SDL_Window* sdlWindow;


};