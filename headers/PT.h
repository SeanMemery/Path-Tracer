#pragma once

#include "GLOBALS.h"
#include "vec3.h"
#include "Renderers.h"
#include "Scene.h"
#include "Obj.h"
#include "Mat.h"
#include "Camera.h"

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

    void SaveImage(char * name);
    void SwitchRenderModes();

    Renderers renderer;

    // ImGui Vars
    char fileName[32];
    float resPerc;
    SDL_Window* sdlWindow;


};