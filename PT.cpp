#include "PT.h"

int xRes, yRes, xScreen, yScreen, maxDepth, currentRenderer, rayCount, sampleCount;
bool denoising, moving, quit, rendering, refresh;
unsigned int mainTexture; 
float exposure, g, randSamp;
int displayMetric, rootThreadsPerBlock;
std::string skepuBackend;
double renderTime, denoiseTime, epochTime, totalTime;
uint64_t GloRandS[2];

Scene scene;
ImGuiWindowFlags window_flags;
Camera cam;
ManageConstants mConstants;
Constants constants;

// Denoising
DenoiserNN denoiserNN;
int denoisingN, trainingEpoch, denoisingBackend;
std::string denoisingSkePUBackend;
bool training, weightsLoaded, skipCudaDenoise;
float* layerTwoValues; 
float* layerThreeValues;
SecondaryFeatures* sFeatures;
float onetwo[360];
float twothree[100];
float threefour[80];
float learningRate = 0.0001f;
int samplesWhenTraining = 4;

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
    currentRenderer = 6;
    denoisingBackend = 1;
    rendering = true;
    mConstants = ManageConstants();

    cam = Camera();
    cam.pos = vec3(0, 0, -9);
    cam.forward = vec3(0, 0, 1);
    cam.up = vec3(0, 1, 0);
    cam.right = vec3(1, 0, 0);
    cam.focalLen = 1.0f;
    cam.hfov = 120;
    cam.vfov = 90;

    exposure = 1.0f;
    g = 1.3f;
    objEdit = 0;
    displayMetric = 0;
    denoisingN = 1;
    maxDepth = 4;
    lRateInt = 6;
    randSamp = 0.005f;
    rootThreadsPerBlock = 16;

    scene = Scene();
    scene.InitScene();
    mConstants.UpdateConstants();

    denoiserNN = DenoiserNN();

    GLOBALS::InitScreens(true);
}

void PT::RenderLoop() {

    while(!quit) {

        ProcessInput();
        ImGui();

        if (rendering) {
            if (refresh) {RefreshScreen();}
            render();
        }
        else if (training)
            denoiserNN.TrainNN();

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
                case 0:
                    col = preScreen[ind] / sampleCount;
                    break;
                case 1:
                    col = denoisedCol[ind];
                    break;
                case 2:
                    col = ((normal[ind] / sampleCount)+vec3(1.0f, 1.0f, 1.0f)) / 2.0f;
                    break;
                case 3:
                    col = albedo1[ind] / sampleCount;
                    break;
                case 4:
                    col = albedo2[ind] / sampleCount;
                    break;
                case 5:
                    col = directLight[ind] / sampleCount;
                    break;
                case 6:
                    col = vec3(1.0f/fabs(worldPos[ind].x / sampleCount), 1.0f/fabs(worldPos[ind].y / sampleCount), 1.0f/fabs(worldPos[ind].z / sampleCount));
                    break;
                case 7:
                    col = targetCol[ind];
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

void PT::RefreshScreen(){
    if (moving) {
        xRes = 1920*0.25f;
        yRes = 1080*0.25f;
        moving = false;
        refresh = true;
        mConstants.UpdateCam();
    } else {
        xRes = 1920*resPerc;
        yRes = 1080*resPerc;
        xScreen = 1920*screenPerc;
        yScreen = 1080*screenPerc;
        mConstants.UpdateConstants();
        refresh = false;
    }

    sampleCount = 0;
    totalTime = 0;

    // Reset Screens
    GLOBALS::DeleteScreens(true);
    GLOBALS::InitScreens(true);
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
        ImGui::Text("Total Render Time %.3f s", totalTime / 1000.0f);

        refresh |= ImGui::SliderFloat("Resolution Strength ( 1920x1080 )", &resPerc, 0.01f, 2.0f);
        refresh |= ImGui::SliderFloat("Screen Size ( 1920x1080 )", &screenPerc, 0.01f, 2.0f);
        ImGui::Text("Resolution: (%d x %d)", xRes, yRes);
        ImGui::Text("Screen Size: (%d x %d)", xScreen, yScreen);

        // Post Process
        ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f);
        ImGui::SliderFloat("Gamma", &g, 1.0f, 10.0f);

        refresh |= ImGui::SliderFloat("Random Sampling Strength", &randSamp, 0.0f, 0.25f);
        refresh |= ImGui::SliderInt("Max Depth", &maxDepth, 1, 12);
        ImGui::Text("Root Threads Per Block: %d", rootThreadsPerBlock);
        if (ImGui::Button("Incr rTHB")) {
            rootThreadsPerBlock += 2;
            refresh = true;
        }
        if (ImGui::Button("Decr rTHB") && rootThreadsPerBlock > 2) {
            rootThreadsPerBlock -= 2;
            refresh = true;
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

        ImGui::InputText("Scene File", sceneName, IM_ARRAYSIZE(sceneName));
        if (ImGui::Button("Load Scene")){
            refresh |= scene.LoadScene(sceneName);
        }

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
            if (ImGui::Button("Remove Shape")) {
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
            refresh |= scene.matList[scene.objList[objEdit]->mat_ind]->ImGuiEdit();
            ImGui::Text("-------------------Mat-------------------");
        }

        ImGui::End();
    }

    // Denoising Settings
    {
        if (denoising) {

            ImGui::Begin("Denoisng", NULL, 0);

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
                        CUDADenoiser::denoise();                             

                if (ImGui::InputInt("Learning Rate (Inverse)", &lRateInt, 0, 16))
                    learningRate = 1.0f / pow(10, lRateInt);
                ImGui::InputInt("Sample for Training", &samplesWhenTraining);

                if (!training) {
                    if (ImGui::Button("Start Training")) 
                        denoiserNN.InitTraining();
                } else {
                    ImGui::Text("Epoch: %d", trainingEpoch);
                    ImGui::Text("Epoch Time: %.3f", epochTime);
                    ImGui::Text("RelMSE: %.3f", denoiserNN.relMSE);
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

