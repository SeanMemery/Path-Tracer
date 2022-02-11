#include "DenoiserNN.h"

void DenoiserNN::GenRelMSE() {

    relMSE = 0.0f;
    int totalPixels = xRes*yRes;

    #pragma omp parallel for
    for (int ind = 0; ind < totalPixels; ind++) {

        // Gen RelMSE
        relMSE += (denoisedCol[ind].x - targetCol[ind].x)*(denoisedCol[ind].x - targetCol[ind].x)/(targetCol[ind].x*targetCol[ind].x + 0.00001f);
        relMSE += (denoisedCol[ind].y - targetCol[ind].y)*(denoisedCol[ind].y - targetCol[ind].y)/(targetCol[ind].y*targetCol[ind].y + 0.00001f);
        relMSE += (denoisedCol[ind].z - targetCol[ind].z)*(denoisedCol[ind].z - targetCol[ind].z)/(targetCol[ind].z*targetCol[ind].z + 0.00001f);
    }

    relMSE *= sampleCount/2.0f;
}

void DenoiserNN::saveTargetCol() {
    for (int j = 0; j < yRes; j++) {
        for (int i = 0; i < xRes; i++) {
            int index = xRes*j + i;
            targetCol[index] = preScreen[index]/sampleCount;
        }
    }
}

void DenoiserNN::InitTraining() {

    training = true;

    // Get Target
	saveTargetCol();
	// Load Current Values
	if (!LoadWeights("NNWeights"))
		RandomizeWeights();
    weightsLoaded = true;

    trainingEpoch = 0;

    // Create Training File
    std::ofstream oFile("ErrorLog.txt");
    if (oFile.is_open()) {
        oFile << "Res: (" << xRes << "x" << yRes << ") Samples: " << samplesWhenTraining << " L Rate: " << learningRate << "," << std::endl;  
        oFile.close();
    }

}
void DenoiserNN::AppendTrainingFile() {
    std::ofstream oFile("ErrorLog.txt", std::ios_base::app);
    if (oFile.is_open()) {
        oFile << relMSE << "," << std::endl;  
        oFile.close();
    }
}
void DenoiserNN::EndTraining() {

    training = false;

	// Save Trained Weights
	OutputWeights("NNWeights");
}

void DenoiserNN::TrainNN() {

            // Save Epoch Time
            auto epochTimer = clock_::now();

            // Reset Sample Count
            sampleCount = 0;

            // Initialize
            GLOBALS::DeleteScreens(false);
            GLOBALS::InitScreens(false);

			// Gen new image
            for (int sample = 0; sample < samplesWhenTraining; sample++)
			    CUDARender::render();

            // Denoise (and forward prop)
            CUDADenoiser::denoise();

			// Get Error Value
			GenRelMSE();

            // Append Training File
            AppendTrainingFile();

			// Back Prop
            CUDADenoiserNN::BackProp();

            // Epoch
            trainingEpoch++;

            // Save Epoch Time
            epochTime = std::chrono::duration_cast<milli_second_>(clock_::now() - epochTimer).count();

}

// Weight Functions

static const int A = 49;
static const int B = 21;
static const int C = 28;
static const int R = 17;

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}
static inline uint64_t xoroshiro128PP() {
	const uint64_t s0 = GloRandS[0];
	uint64_t s1 = GloRandS[1];
	const uint64_t result_plus = rotl(s0 + s1, R) + s0;
	s1 ^= s0;
	GloRandS[0] = rotl(s0, A) ^ s1 ^ (s1 << B);
	GloRandS[1] = rotl(s1, C);
	return result_plus;
}
float RandBetween(float min, float max) {
    if (min > max)
	    return 0;
    double f = (double)xoroshiro128PP() / (1ULL << 63) / (double)2;
    f *= max - min;
    f += min;
    return f;
}
void DenoiserNN::RandomizeWeights() {
    int ind;
    for (ind = 0; ind < 360; ind ++)
        onetwo[ind] = RandBetween(-0.5, 0.5);
    for (ind = 0; ind < 100; ind ++)
        twothree[ind] = RandBetween(-0.5, 0.5);
    for (ind = 0; ind < 70; ind ++)
        threefour[ind] = RandBetween(-0.5, 0.5);
}
void DenoiserNN::OutputWeights(std::string name) {
    if (name.size()==0)
        name = std::string("NNWeights");
    name = std::string("../Weights/").append(name);
    name.append(".txt");
    std::ofstream oFile(name);
    if (oFile.is_open()) {
        int ind;
        for (ind = 0; ind < 360; ind ++)
            oFile << onetwo[ind] << " ";
        oFile << "\n";
        for (ind = 0; ind < 100; ind ++)
            oFile << twothree[ind] << " ";
        oFile << "\n";
        for (ind = 0; ind < 70; ind ++)
            oFile << threefour[ind] << " "; 
        oFile << "\n";
        oFile.close();
    }
    else
        std::cout << "Could not write weights file !" << std::endl;
}
bool DenoiserNN::LoadWeights(std::string name) {
    if (name.size()==0)
        name = std::string("NNWeights");
    name = std::string("../Weights/").append(name);
    name.append(".txt");
    std::ifstream rFile(name);
    std::string newLine;
    int ind;

    if (!rFile.is_open()){
        std::cout << "Cannot open " << name << "!" << std::endl;
        return false;
    }

    // Line 1
    if (getline(rFile, newLine)) {
        std::istringstream in(newLine);
        for (ind = 0; ind < 360; ind++)
            in >> onetwo[ind];
    } else {std::cout << "Invalid file " << name << "!" << std::endl; return false;}

    // Line 2
    if (getline(rFile, newLine)) {
        std::istringstream in(newLine);
        for (ind = 0; ind < 100; ind++)
            in >> twothree[ind];
    } else {std::cout << "Invalid file " << name << "!" << std::endl; return false;}

    // Line 3
    if (getline(rFile, newLine)) {
        std::istringstream in(newLine);
        for (ind = 0; ind < 70; ind++)
            in >> threefour[ind];
    } else {std::cout << "Invalid file " << name << "!" << std::endl; return false;}

    weightsLoaded = true;

    return true;
}