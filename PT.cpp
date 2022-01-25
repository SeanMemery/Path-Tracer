#include "PT.h"

int xRes, yRes, maxDepth, currentRenderer, rayCount, sampleCount;
bool denoising, moving, quit, rendering, refresh;
unsigned int mainTexture; 
std::string skepuBackend;

Scene scene;
ImGuiWindowFlags window_flags;
Camera cam;
vec3* screen;

PT::PT() {

    xRes = 1920;
    yRes = 1080;
    maxDepth = 5;
    currentRenderer = 6;
    rendering = true;
    renderer = Renderers();

    cam = Camera();
    cam.pos = vec3(0, 0, -9);
    cam.forward = vec3(0, 0, 1);
    cam.up = vec3(0, 1, 0);
    cam.right = vec3(1, 0, 0);
    cam.focalLen = 1.0f;
    cam.hfov = 120;
    cam.vfov = 90;

    resPerc = 1.0f;

    scene = Scene();
    scene.InitScene();
    renderer.UpdateConstants();

    screen = new vec3[xRes*yRes];
}

void PT::RenderLoop() {

    while(!quit) {

        ProcessInput();

        ImGui();

        if (refresh) {
            SwitchRenderModes();
            renderer.UpdateConstants();
        }

        if (rendering)
            renderer.Render();

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, xRes, yRes, 0, GL_RGB, GL_FLOAT, &screen[0]);
                
        SDL_GL_SwapWindow(sdlWindow);
        
    }
}

void PT::SwitchRenderModes(){
    if (moving) {
        xRes = 1920*0.3f;
        yRes = 1080*0.3f;
    } else {
        xRes = 1920*resPerc;
        yRes = 1080*resPerc;
    }
    sampleCount = 0;
    screen = new vec3[xRes*yRes];
}

void PT::ProcessInput() {
    SDL_Event evnt;

	while (SDL_PollEvent(&evnt)) {
		switch (evnt.type) {
		default:
			ImGui_ImplSDL2_ProcessEvent(&evnt);
			if (!ImGui::GetIO().WantCaptureMouse && !ImGui::GetIO().WantCaptureKeyboard) {
				if (evnt.type == SDL_MOUSEBUTTONDOWN)
					continue;
			}
			break;
		}
	}

	if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape)))
		quit = true;

    if (ImGui::IsKeyDown(26))
		cam.moveDir(cam.forward);
	if (ImGui::IsKeyDown(22))
		cam.moveDir(cam.forward);
	if (ImGui::IsKeyDown(225))
		cam.moveDir(cam.up);
	if (ImGui::IsKeyDown(224))
		cam.moveDir(cam.up);
	if (ImGui::IsKeyDown(7))
		cam.rotateAroundAxis(CamAxis::FORWARD);
	if (ImGui::IsKeyDown(4))
		cam.rotateAroundAxis(CamAxis::REVFORWARD);
	if (ImGui::IsKeyDown(20))
		cam.rotateAroundAxis(CamAxis::UP);
	if (ImGui::IsKeyDown(8))
		cam.rotateAroundAxis(CamAxis::REVUP);
	if (ImGui::IsKeyDown(27))
		cam.rotateAroundAxis(CamAxis::RIGHT);
	if (ImGui::IsKeyDown(29))
		cam.rotateAroundAxis(CamAxis::REVRIGHT);
}

void PT::ImGui() {

	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame(sdlWindow);
	ImGui::NewFrame();

	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(1920, 1080), ImGuiCond_Always);
	ImGui::Begin("Render", NULL, window_flags | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoBringToFrontOnFocus);


	ImGui::Begin("Settings", NULL, 0);

    ImGui::Text("Rays Per Fram %d", rayCount);
    ImGui::Text("Avg Time Per Frame");
    refresh = ImGui::SliderFloat("Resolution Strength ( 1920x1080 )", &resPerc, 0.01f, 2.0f);
    ImGui::Text("Resolution: (%d x %d)", xRes, yRes);
	ImGui::Text("Max Depth: %d", maxDepth);
    ImGui::SliderInt("Rendering Mode", &currentRenderer, 0, 6);
    switch(currentRenderer) {
        case 0:
            ImGui::Text("CPU");
            break;
        case 1:
            ImGui::Text("OpenMP");
            break;
        case 2:
            ImGui::Text("CUDA");
            break;
        case 3:
            ImGui::Text("OpenGL");
            break;
        case 4:
            ImGui::Text("SkePU CPU");
            skepuBackend = "cpu";
            break;
        case 5:
            ImGui::Text("SkePU OpenMP");
            skepuBackend = "openmp";
            break;
        case 6:
            ImGui::Text("SkePU CUDA");
            skepuBackend = "cuda";
            break;
    }	
    ImGui::Checkbox("Denoise", &denoising);
    ImGui::InputText("Name Render", fileName, IM_ARRAYSIZE(fileName));
	if (ImGui::Button("Save Render"))
	    SaveImage(fileName);

    ImGui::End();

    ImGui::Image((void*)(intptr_t)mainTexture, ImVec2(xRes, yRes), ImVec2(0, 1), ImVec2(1, 0), ImVec4(1, 1, 1, 1), ImVec4(0, 0, 0, 0));

    ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());		

}

void PT::SaveImage(char * name) {
	std::stringstream outString;
	outString << "P3\n" << xRes << ' ' << yRes << "\n255\n";
	for (int j = yRes - 1; j >= 0; --j) {
		for (int i = 0; i < xRes; ++i) {
			int r = static_cast<int>(255.999 * screen[i + xRes * j].x);
			int g = static_cast<int>(255.999 * screen[i + xRes * j].y);
			int b = static_cast<int>(255.999 * screen[i + xRes * j].z);
			outString << r << ' ' << g << ' ' << b << '\n';
		}
	}
	std::ofstream myfile;
	std::stringstream fileName;
	fileName << "../Renders/" << name << ".ppm";
	myfile.open(fileName.str());
	myfile << outString.str();
	myfile.close();
}

