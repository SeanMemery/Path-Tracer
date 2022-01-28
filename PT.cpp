#include "PT.h"

int xRes, yRes, xScreen, yScreen, maxDepth, currentRenderer, rayCount, sampleCount;
bool denoising, moving, quit, rendering, refresh;
unsigned int mainTexture; 
float exposure, g;
int displayMetric;
std::string skepuBackend;
double renderTime;

Scene scene;
ImGuiWindowFlags window_flags;
Camera cam;
Denoiser denoiser;

vec3* preScreen;
vec3* postScreen;

PT::PT() :
    fileName("")
 {

    screenPerc = 0.963f;
    resPerc = 0.8f;

    xRes = 1920 * resPerc;
    yRes = 1080 * resPerc;
    xScreen = 1920 * screenPerc;
    yScreen = 1080 * screenPerc;
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

    exposure = 1.3f;
    g = 1.3f;
    objEdit = 0;
    displayMetric = -1;

    scene = Scene();
    scene.InitScene();
    renderer.UpdateConstants();
    denoiser = Denoiser();

    preScreen = new vec3[xRes*yRes];
    postScreen = new vec3[xRes*yRes];
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

        PostProcess();

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, xRes, yRes, 0, GL_RGB, GL_FLOAT, &postScreen[0]);
                
        SDL_GL_SwapWindow(sdlWindow);
        
    }
}

void PT::PostProcess() {
    vec3 col;
    DenoisingInf info;
    for (int ind = 0; ind < xRes*yRes; ind++) {

        if (!denoising) {
            col = preScreen[ind];

            // Samples
            col /= sampleCount;
        }
        else {
            switch(displayMetric) {
                info  = denoiser.denoisingInf[ind];
                case -1:
                    col = preScreen[ind] / sampleCount;
                    break;
                case 0:
                    col = denoiser.targetCol[ind] / sampleCount;
                    break;
                case 1:
                    col = info.denoisedCol / sampleCount;
                    break;
                case 2:
                    col = info.finalCol / sampleCount;
                    break;
                case 3:
                    col = ((info.normal / sampleCount)+vec3(1.0f, 1.0f, 1.0f)) / 2.0f;
                    break;
                case 4:
                    col = info.albedo1 / sampleCount;
                    break;
                case 5:
                    col = info.albedo2 / sampleCount;
                    break;
                case 6:
                    col = ((info.firstBounce / sampleCount)+vec3(1.0f, 1.0f, 1.0f)) / 2.0f;
                    break;
                case 7:
                    col = vec3(5.0f/(info.depth / sampleCount),5.0f/(info.depth / sampleCount),5.0f/(info.depth / sampleCount));
                    break;
                case 8:
                    col = vec3(info.directLight, info.directLight, info.directLight) / sampleCount;
                    break;
                case 9:
                    col = vec3(1.0f/fabs(info.worldPos.x / sampleCount), 1.0f/fabs(info.worldPos.y / sampleCount), 1.0f/fabs(info.worldPos.z / sampleCount));
                    break;
            }
        }

        // Exposure
        col /= exposure;

        // Gamma
        col = vec3(pow(col.x, 1.0f/g), pow(col.y, 1.0f/g), pow(col.z, 1.0f/g));

        postScreen[ind] = col;

    }
}

void PT::SwitchRenderModes(){
    if (moving) {
        xRes = 1920*0.25f;
        yRes = 1080*0.25f;
        moving = false;
        refresh = true;
    } else {
        xRes = 1920*resPerc;
        yRes = 1080*resPerc;
        refresh = false;
    }
    xScreen = 1920*screenPerc;
    yScreen = 1080*screenPerc;
    sampleCount = 0;

    // Reset Screens
    delete preScreen;
    delete postScreen;
    preScreen = new vec3[xRes*yRes];
    postScreen = new vec3[xRes*yRes];
    denoiser.initDenoisingInf();
}

void PT::ProcessInput() {
    SDL_Event evnt;

	while (SDL_PollEvent(&evnt)) {
		switch (evnt.type) {
		default:
			ImGui_ImplSDL2_ProcessEvent(&evnt);
			// if (!ImGui::GetIO().WantCaptureMouse && !ImGui::GetIO().WantCaptureKeyboard) {
			// 	if (evnt.type == SDL_MOUSEBUTTONDOWN)
			// 		continue;
			// }
			break;
		}
	}

	if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape)))
		quit = true;

    if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Space)))
		rendering = !rendering;

    if (!ImGui::GetIO().WantCaptureKeyboard && rendering) {
        // Forward
        if (ImGui::IsKeyDown(26))
            cam.moveDir(cam.forward);
        // Back
        if (ImGui::IsKeyDown(22))
            cam.moveDir(-cam.forward);
        if (ImGui::IsKeyDown(225))
            cam.moveDir(cam.up);
        if (ImGui::IsKeyDown(224))
            cam.moveDir(-cam.up);
        // Right
        if (ImGui::IsKeyDown(7))
            cam.rotateAroundAxis(CamAxis::UP);
        // Left
        if (ImGui::IsKeyDown(4))
            cam.rotateAroundAxis(CamAxis::REVUP);
        if (ImGui::IsKeyDown(20))
            cam.rotateAroundAxis(CamAxis::FORWARD);
        if (ImGui::IsKeyDown(8))
            cam.rotateAroundAxis(CamAxis::REVFORWARD);
        if (ImGui::IsKeyDown(27))
            cam.rotateAroundAxis(CamAxis::RIGHT);
        if (ImGui::IsKeyDown(29))
            cam.rotateAroundAxis(CamAxis::REVRIGHT);
    }
}

void PT::ImGui() {

	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame(sdlWindow);
	ImGui::NewFrame();

	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(1920, 1080), ImGuiCond_Always);
	ImGui::Begin("Render", NULL, window_flags | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::Image((void*)(intptr_t)mainTexture, ImVec2( xScreen, yScreen), ImVec2(0, 1), ImVec2(1, 0), ImVec4(1, 1, 1, 1), ImVec4(0, 0, 0, 0));

    // General Settings
    {
        ImGui::Begin("Settings", NULL, 0);

        ImGui::Text("Last Frame Time %.3f ms", renderTime);
        ImGui::Text("Rays Per Frame %d", rayCount);
        float numMillRays = rayCount / 1000000.0f;
        ImGui::Text("Time Per Million Rays %.3f", renderTime/numMillRays);
        ImGui::Text("Number of Samples %d", sampleCount);

        refresh |= ImGui::SliderFloat("Resolution Strength ( 1920x1080 )", &resPerc, 0.01f, 2.0f);
        refresh |= ImGui::SliderFloat("Screen Size ( 1920x1080 )", &screenPerc, 0.01f, 2.0f);

        // Post Process
        ImGui::SliderFloat("Exposure", &exposure, 1.0f, 10.0f);
        ImGui::SliderFloat("Gamma", &g, 1.0f, 10.0f);

        ImGui::Text("Screen Size: (%d x %d)", xScreen, yScreen);
        ImGui::Text("Resolution: (%d x %d)", xRes, yRes);
        ImGui::Text("Max Depth: %d", maxDepth);
        refresh |= ImGui::SliderInt("Rendering Mode", &currentRenderer, 0, 6);
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
        refresh |= ImGui::Checkbox("Denoise", &denoising);
        ImGui::InputText("Name Render", fileName, IM_ARRAYSIZE(fileName));
        if (ImGui::Button("Save Render"))
            SaveImage(fileName);

        ImGui::End();
    }

    // Object Settings
    {
        ImGui::Begin("Objects", NULL, 0);

        if (ImGui::Button("New Sphere")) {
            scene.AddShape(0);
            objEdit = scene.objList.size() - 1;
            refresh = true;
        }
        if (ImGui::Button("New AABB")) {
            scene.AddShape(1);
            objEdit = scene.objList.size() - 1;
            refresh = true;
        }

        ImGui::SliderInt("Obj Edit", &objEdit, 0, scene.objList.size()-1);

        ImGui::Text("Number of Objects: %d", scene.objList.size());
        ImGui::Text("-------------------Obj-------------------");
        if (ImGui::Button("Remove Shape")) {
            scene.RemoveShape(objEdit);
            refresh = true;
        }
        if (scene.objList[objEdit]->inImportantList) {
            if (ImGui::Button("Remove from Imp List")) {
                scene.RemoveFromImpList(objEdit);
                refresh = true;
            }
        } else {
            if (ImGui::Button("Add to Imp List")) {
                scene.AddToImpList(objEdit);
                refresh = true;
            }
        }
        refresh |= scene.objList[objEdit]->ImGuiEdit();
        ImGui::Text("-------------------Obj-------------------");
        ImGui::Text("");
        ImGui::Text("-------------------Mat-------------------");
        refresh |= scene.objList[objEdit]->material.ImGuiEdit();
        ImGui::Text("-------------------Mat-------------------");


        ImGui::End();
    }

    // Denoising Settings
    {
        if (denoising) {

            ImGui::Begin("Denoisng", NULL, 0);

            // if (denoiser.hasTarget) {
            //     if (ImGui::Button("Reset Target")) {
            //         denoiser.hasTarget=false;
            //         denoiser.display(-1);
            //     }
            //     if (ImGui::Button("Show Target"))
            //         denoiser.display(-3);
            // }
            // else {
            //     if (ImGui::Button("Save Target Image")) 
            //         denoiser.saveTargetCol();
            // }

            // Update screen texture based on input metric
            // 0 Target Col, 1: denoised image, 2: image, 3: normal, 4: albedo1, 5:albedo2, 6: firstBounce, 7: depth, 8: directLight, 9: worldPos

            ImGui::SliderInt("Display", &displayMetric, -1, 9);
            ImGui::Text(displayNames[displayMetric+1]);
        
            ImGui::End();
        }
    }

    ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());		

}

void PT::SaveImage(char * name) {
	std::stringstream outString;
	outString << "P3\n" << xRes << ' ' << yRes << "\n255\n";
	for (int j = yRes - 1; j >= 0; --j) {
		for (int i = 0; i < xRes; ++i) {
			int r = static_cast<int>(255.999 * postScreen[i + xRes * j].x);
			int g = static_cast<int>(255.999 * postScreen[i + xRes * j].y);
			int b = static_cast<int>(255.999 * postScreen[i + xRes * j].z);
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

