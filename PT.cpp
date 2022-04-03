#include "PT.h"

int xRes, yRes, xScreen, yScreen, maxDepth, currentRenderer, rayCount, sampleCount, trainingCount;
bool denoising, moving, quit, rendering, refresh, trainingLimitBool;
unsigned int mainTexture; 
float exposure, g, randSamp, avgTMR, lRateInt, lRateIntMax;
int displayMetric, rootThreadsPerBlock;
std::string skepuBackend;
double renderTime, denoiseTime, epochTime, totalTime, exposureTime, imguiTime, postProcessTime, screenUpdateTime, totalRenderTime, trainingTime;

Scene scene;
ImGuiWindowFlags window_flags;
Camera cam;
Denoiser denoiser;
Renderers renderer;
Constants constants;

// Denoising
DenoiserNN denoiserNN;
int denoisingN, trainingEpoch, denoisingBackend;
std::string denoisingSkePUBackend;
bool training, weightsLoaded, skipCudaDenoise;
float* layerTwoValues; 
float* layerThreeValues;

// Screens
vec3* preScreen;
vec3* postScreen;
vec3* normal;
vec3* albedo1;
vec3* albedo2;
vec3* directLight;
vec3* worldPos;
vec3* denoisedCol;
vec3* targetCol;
DenoisingInf* denoisingInf;

// Value Buffers
std::vector<uint> vertexIndices;
std::vector<float> vertices;
std::vector<float> objAttributes;

PT::PT() :
    fileName(""),
    weightsName(""),
    weightsNameSave(""),
    sceneName("")
 {

    screenPerc = 0.963f;
    resPerc = 0.8f;

    xRes = 1920 * resPerc;
    yRes = 1080 * resPerc;
    xScreen = 1920 * screenPerc;
    yScreen = 1080 * screenPerc;
    currentRenderer = 2;
    denoisingBackend = 2;
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

    exposure = 2.0f;
    g = 2.1f;
    objEdit = 0;
    displayMetric = 0;
    denoisingN = 1;
    maxDepth = 4;
    lRateInt = 6.0f;
    lRateIntMax = lRateInt;
    randSamp = 0.005f;
    rootThreadsPerBlock = 8;
    trainingCount = 0;
    avgTMR = 0;

    scene = Scene();
    scene.InitScene();

    denoiser = Denoiser();
    denoiserNN = DenoiserNN();

    GLOBALS::InitScreens(true);
    CUDADenoiser::InitBuffers();
    CUDADenoiserNN::InitBuffers();

    renderer.UpdateConstants();
    renderer.UpdateCam();
}

void PT::UpdateScreen() {
    auto screenUpdateTimer = clock_::now();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, xRes, yRes, 0, GL_RGB, GL_FLOAT, &postScreen[0]);
    SDL_GL_SwapWindow(sdlWindow);
    screenUpdateTime = std::chrono::duration_cast<milli_second_>(clock_::now() - screenUpdateTimer).count();
}

void PT::RenderLoop() {

    while(!quit) {

        auto frameTimer = clock_::now();

        ImGui();

        if (rendering) {
            if (refresh) {RefreshScreen();}
            renderer.Render();
        }
        else if (training) {
            denoiserNN.TrainNN();
            if (!training)
                denoiserNN.EndTraining();   
        }


        PostProcess();
        UpdateScreen();

        totalTime = std::chrono::duration_cast<milli_second_>(clock_::now() - frameTimer).count();
        if (rendering)
            totalRenderTime += totalTime;
    }
}

void PT::PostProcess() {

    auto ppTimer = clock_::now();

    CUDARender::PostProcess();

    postProcessTime = std::chrono::duration_cast<milli_second_>(clock_::now() - ppTimer).count();

}

void PT::RefreshScreen(){
    if (moving) {
        xRes = 1920*resPerc*0.5f;
        yRes = 1080*resPerc*0.5f;
        moving = false;
        refresh = true;
        renderer.UpdateCam();
    } else {
        xRes = 1920*resPerc;
        yRes = 1080*resPerc;
        xScreen = 1920*screenPerc;
        yScreen = 1080*screenPerc;
        renderer.UpdateConstants();
        refresh = false;
    }

    sampleCount = 0;
    totalRenderTime = 0;
    avgTMR = 0;

    // Reset Screens
    GLOBALS::DeleteScreens(true);
    CUDADenoiser::FreeBuffers();
    CUDADenoiserNN::FreeBuffers();
    GLOBALS::InitScreens(true);
    CUDADenoiser::InitBuffers();
    CUDADenoiserNN::InitBuffers();
}

void PT::ProcessInput() {
    SDL_Event evnt;

	while (SDL_PollEvent(&evnt)) {
		switch (evnt.type) {
		default:
			ImGui_ImplSDL2_ProcessEvent(&evnt);
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

    auto imguiTimer = clock_::now();

    ProcessInput();

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

        ImGui::Text("-----------------Time-----------------");

        ImGui::Text("ImGui:        %.3f ms (%.3f %)", imguiTime,        100.0f*imguiTime       /totalTime);
        ImGui::Text("Render:       %.3f ms (%.3f %)", renderTime,       100.0f*renderTime      /totalTime);
        ImGui::Text("Post Process: %.3f ms (%.3f %)", postProcessTime,  100.0f*postProcessTime /totalTime);
        ImGui::Text("Screen:       %.3f ms (%.3f %)", screenUpdateTime, 100.0f*screenUpdateTime/totalTime);
        ImGui::Text("Total:        %.3f ms", totalTime);

        ImGui::Text("-----------------Time-----------------");

        float numMillRays = rayCount / 1000000.0f;
        numMillRays = numMillRays == 0 ? 1 : numMillRays;
        ImGui::Text("Rays Per Frame %d",          rayCount);
        float TMR = renderTime/numMillRays;
        ImGui::Text("Time Per Million Rays %.3f", TMR);
        if (rendering)
            avgTMR += TMR;
        ImGui::Text("Avg Time Per Million Rays %.3f", avgTMR/(float)sampleCount);  
        ImGui::Text("Number of Samples %d",       sampleCount);
        ImGui::Text("Rendering Time %.3f s",      totalRenderTime/1000.0f);

        refresh |= ImGui::SliderFloat("Resolution Strength ( 1920x1080 )", &resPerc,   0.01f, 2.0f);
        refresh |= ImGui::SliderFloat("Screen Size ( 1920x1080 )",         &screenPerc, 0.01f, 2.0f);
        ImGui::Text("Resolution: (%d x %d)", xRes, yRes);
        ImGui::Text("Screen Size: (%d x %d)", xScreen, yScreen);

        // Post Process
        ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f);
        ImGui::SliderFloat("Gamma", &g, 1.0f, 10.0f);

        refresh |= ImGui::SliderFloat("Random Sampling Strength", &randSamp, 0.0f, 0.25f);
        refresh |= ImGui::SliderInt("Max Depth", &maxDepth, 1, 12);
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
                ImGui::Text("Root Threads Per Block: %d", rootThreadsPerBlock);
                if (ImGui::Button("Incr rTHB")) {
                    rootThreadsPerBlock += 2;
                    refresh = true;
                }
                if (ImGui::Button("Decr rTHB") && rootThreadsPerBlock > 2) {
                    rootThreadsPerBlock -= 2;
                    refresh = true;
                }
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
        ImGui::Text("Auto Exposure Time: %.3f", exposureTime);
        if (ImGui::Button("Auto Expose"))
            renderer.AutoExposure();
        ImGui::InputText("Name Render", fileName, IM_ARRAYSIZE(fileName));
        if (ImGui::Button("Save Render"))
            SaveImage(fileName);

        ImGui::End();
    }

    // Object Settings
    {
        ImGui::Begin("Objects", NULL, 0);
        ImGui::Text("-------------------Scene-------------------");
        ImGui::InputText("Scene Name", sceneName, IM_ARRAYSIZE(sceneName));
        if (ImGui::Button("Load Scene")) {
            refresh |= scene.LoadScene(sceneName);
            objEdit = 0;
        }
        if (ImGui::Button("Save Scene"))
            scene.SaveScene(sceneName);
        ImGui::Text("-------------------Scene-------------------");

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

        if (scene.objList.size() > 0) {

            ImGui::SliderInt("Obj Edit", &objEdit, 0, scene.objList.size()-1);
            ImGui::Text("Number of Objects: %d", scene.objList.size());
            ImGui::Text("-------------------Obj-------------------");
            if (ImGui::Button("Copy Object")) {
                scene.CopyObject(objEdit);
                objEdit = scene.objList.size() - 1;
            }
            if (ImGui::Button("Remove Object")) {
                scene.RemoveShape(objEdit);
                refresh = true;
                objEdit = 0;
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
            if (ImGui::Button("New Mat")) {
                scene.AddMat(0);
                scene.objList[objEdit]->mat_ind = scene.matList.size()-1;
                refresh = true;
            }
            refresh |= scene.matList[scene.objList[objEdit]->mat_ind]->ImGuiEdit();
            ImGui::Text("-------------------Mat-------------------");
        }

        ImGui::End();
    }

    // Denoising Settings
    {
        if (denoising) {

            ImGui::Begin("Denoisng", NULL, 0);

            ImGui::SliderInt("Denoising Backend", &denoisingBackend, 0, 6);
                switch(denoisingBackend) {
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
                        denoisingSkePUBackend = "cpu";
                        break;
                    case 5:
                        ImGui::Text("SkePU OpenMP");
                        denoisingSkePUBackend = "openmp";
                        break;
                    case 6:
                        ImGui::Text("SkePU CUDA");
                        denoisingSkePUBackend = "cuda";
                        ImGui::Checkbox("Skip CUDA Denoise", &skipCudaDenoise);
                        break;
                }

            // Update screen texture based on input metric
            // 0: Regular View, 1: denoised image, 2: normal, 3: albedo1, 4: albedo2, 5: directLight, 6: worldPos, 7 tagetCol

            ImGui::SliderInt("Display", &displayMetric, 0, 7);
            ImGui::Text(displayNames[displayMetric]);

            if (!rendering) {

                ImGui::InputInt("Denoising N", &denoisingN);

                ImGui::Text("Denoise Time: %.3f", denoiseTime);

                ImGui::Text("");
                    
                ImGui::InputText("Weights File", weightsName, IM_ARRAYSIZE(weightsName));
                if (ImGui::Button("Load Weights File"))
                    denoiserNN.LoadWeights(weightsName); 
                if (weightsLoaded)
                    if (ImGui::Button("Denoise Image"))
                        denoiser.denoise();                             

                ImGui::InputFloat("Learning Rate (Inverse)", &lRateInt, 1, 16);
                ImGui::InputInt("Sample for Training", &denoiserNN.samplesWhenTraining);

                if (!training) {
                    ImGui::InputInt("Training Limit", &trainingCount, 0, 10000);
                    trainingLimitBool = trainingCount > 0;
                    if (trainingLimitBool)
                        ImGui::InputFloat("Target Learning Rate", &lRateIntMax, 1, 16);
                    if (ImGui::Button("Start Training")) 
                        denoiserNN.InitTraining();
                } else {
                    ImGui::Text("Epoch: %d", trainingEpoch);
                    ImGui::Text("Epoch Time: %.3f ms", epochTime);
                    ImGui::Text("Training Time: %.3f s", trainingTime/1000.0f);
                    ImGui::Text("RelMSE: %.3f", denoiserNN.relMSE);
                    ImGui::Text("Learning Rate %.12f", denoiserNN.learningRate);
                    if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Tab)) || ImGui::Button("Stop Training"))
                        denoiserNN.EndTraining();   
                }

                ImGui::InputText("Save Weights File As", weightsNameSave, IM_ARRAYSIZE(weightsNameSave));
                if (ImGui::Button("Save Weights File"))
                    denoiserNN.OutputWeights(weightsNameSave); 
                if (ImGui::Button("Randomize Weights (Overwrite File!)")) {
                    denoiserNN.RandomizeWeights();
                    denoiserNN.OutputWeights("NNWeights");
                    denoiserNN.LoadWeights("NNWeights");
                }  


            }
            else {
                ImGui::Text("---- Need to Pause Rendering ----");
            }
        
            ImGui::End();
        }
    }

    ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());	

    imguiTime = std::chrono::duration_cast<milli_second_>(clock_::now() - imguiTimer).count();
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

