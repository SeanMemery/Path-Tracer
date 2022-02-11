#pragma once

#include "GLOBALS.h"
#include "vec3.h"
#include "ManageConstants.h"
#include "Scene.h"
#include "Obj.h"
#include "Mat.h"
#include "Camera.h"
#include "DenoiserNN.h"

#include "ext/imgui.h"
#include "ext/imgui_impl_sdl.h"
#include "ext/imgui_impl_opengl3.h"
#include <SDL/SDL.h>
#include <GL/glew.h>

#include <sstream>
#include <fstream>

#include <chrono>
using namespace std::chrono;
typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::milli > milli_second_;


class PT {
public:

    PT();

    void RenderLoop();
    void ImGui();
    void ProcessInput();
    void PostProcess();

    void SaveImage(char * name);
    void RefreshScreen();

    // 0: Regular View,  denoised image, 2: normal, 3: albedo1, 4: albedo2, 5: directLight, 6: worldPos, 7 targetCol
    const char* displayNames[8] = {"Image", "Denoised Col", "Normal", "Albedo 1", "Albedo 2", "Direct Light", "World Pos", "Target Col"};

    // ImGui Vars
    int objEdit, lRateInt;
    char fileName[32], weightsName[32], weightsNameSave[32], sceneName[32];
    float resPerc, screenPerc;
    SDL_Window* sdlWindow;


    void render() {
        auto renderTimer = clock_::now();

        CUDARender::render();

        renderTime = std::chrono::duration_cast<milli_second_>(clock_::now() - renderTimer).count();
        totalTime += renderTime;
    }


};