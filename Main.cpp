#include "PT.h"
#include <SDL/SDL.h>
#include <GL/glew.h>

enum WindowFlags { INVISIBLE = 0x1, FULLSCREEN = 0x2, BORDERLESS = 0x4 }; // set to bit values so they can be combined with |


int main(int argc, char** argv)
{
	PT main;

    // Create window with graphics context
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

	auto currentFlags = FULLSCREEN;

    Uint32 flags = SDL_WINDOW_OPENGL;
	if (currentFlags & INVISIBLE)
		flags |= SDL_WINDOW_HIDDEN;
	if (currentFlags & FULLSCREEN)
		flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
	if (currentFlags & BORDERLESS)
		flags |= SDL_WINDOW_BORDERLESS;

	main.sdlWindow = SDL_CreateWindow("Path Tracer Project", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1920, 1078, flags);
	if (main.sdlWindow == nullptr) {
		printf("SDL window could not be created!");
        return 0;
    }

	auto _glContext = SDL_GL_CreateContext(main.sdlWindow);
	if (_glContext == nullptr) {
		printf("SDL_GL context could not be created!");
        return 0;
    }
	int currError = SDL_GL_MakeCurrent(main.sdlWindow, _glContext);
	if (currError != 0) {
		printf("Could not make context current!", currError);
        return 0;
    }
	GLenum error = glewInit();
	if (error != GLEW_OK) {
		printf("Could not initialize glew!", error);
        return 0;
    }
	std::printf("*** OpenGL Version: %s ***\n", glGetString(GL_VERSION));
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	SDL_GL_SetSwapInterval(0); // VSYNC
	
	SDL_GL_MakeCurrent(main.sdlWindow, _glContext);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplSDL2_InitForOpenGL(main.sdlWindow, _glContext);
	ImGui_ImplOpenGL3_Init("#version 330");

	// ImGui Window Flags
	window_flags = 0;
	window_flags |= ImGuiWindowFlags_NoMove;
	window_flags |= ImGuiWindowFlags_NoResize; // FOR NOW 
	window_flags |= ImGuiWindowFlags_NoCollapse;  
	window_flags |= ImGuiWindowFlags_NoTitleBar;

	// PT Texture Setup
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &mainTexture);
	glBindTexture(GL_TEXTURE_2D, mainTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	
	main.RenderLoop();

	return 0;

}