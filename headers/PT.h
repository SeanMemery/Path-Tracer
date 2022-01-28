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

    // Screens
    void InitScreens();
    void DeleteScreens();

    Renderers renderer;

    // 0: Regular View, 1 Target Col, 2: denoised image, 3: normal, 4: albedo1, 5: albedo2, 6: directLight, 7: worldPos
    const char* displayNames[8] = {"Image", "Target Col", "Denoised Col", "Normal", "Albedo 1", "Albedo 2", "Direct Light", "World Pos"};

    // ImGui Vars
    int objEdit;
    char fileName[32];
    float resPerc, screenPerc;
    SDL_Window* sdlWindow;


};