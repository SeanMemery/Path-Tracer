#include "DenoiserNN.h"


// Compute Functions
void DenoiserNN::GeneratePFeatures() {
    
    int totalPixels = xRes*yRes;
    delete pFeatures;
    pFeatures = new PrimaryFeatures[totalPixels];

    int c;
    for (int ind = 0; ind < totalPixels; ind++) {
        pFeatures[ind].normal      = 0.0f;
        pFeatures[ind].alb1        = 0.0f;
        pFeatures[ind].alb2        = 0.0f;
        pFeatures[ind].worldPos    = 0.0f;
        pFeatures[ind].directLight = 0.0f;

        for (c = 0; c < 3; c++) {
            pFeatures[ind].normal   += normal[ind][c]   / 3.0f;
            pFeatures[ind].alb1     += albedo1[ind][c]  / 3.0f;
            pFeatures[ind].alb2     += albedo2[ind][c]  / 3.0f;
            pFeatures[ind].worldPos += worldPos[ind][c] / 3.0f;
        }
        pFeatures[ind].directLight = directLight[ind].x;
    }
}
void DenoiserNN::GenerateSFeatures() {

    // - K Means of single pixel, K std deviations of single pixel, K Means of 7x7 block, K std deviations of 7x7 block (20 total)
	// - Magnitude of gradients of K features of single pixel (sobel operator) (5 total)
	// - Sum of the abs difference between K of each pixel in 3x3 block and the mean of that 3x3 block (5 total)
	// - MAD of K features, so median of values minus median value, in NxN block (5 total)
	// - 1/totalSamples (1 total)

        int totalPixels = xRes*yRes;

        delete sFeatures;
        sFeatures = new SecondaryFeatures[totalPixels];

        int ind, c = 0;
        int ijInd = 0;
        int jFixed, iFixed = 0;
        SecondaryFeatures* s;
        float Gx[9] = { 1,  2,  1, 
                        0,  0,  0, 
                       -1, -2, -1};
        float Gy[9] = { 1,  0,  -1, 
                        2,  0,  -2, 
                        1,  0,  -1};
        int linearInd = 0;
        float valuesForMAD[5][9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float madMedians[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float medianGetter[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        int K, i, j;
        for (int jMain = 0; jMain < yRes; jMain++) { 
            for (int iMain = 0; iMain < xRes; iMain++) { 

                float meansForMD[5], GxSum[5], GySum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};       

                ind = jMain*xRes+ iMain;
                s = &sFeatures[ind];

                // Reset Values
                for ( K=0; K < 5; K++) {
                    s->meansSingle[K] = 0.0f;
                    s->sdSingle[K] = 0.0f;
                    s->meansBlock[K] = 0.0f;
                    s->sdBlock[K] = 0.0f;
                    s->meanDeviation[K] = 0.0f;
                    s->gradients[K] = 0.0f;
                    s->MAD[K] = 0.0f;
                }
                s->L = 0.0f;

                // K single Means
                s->meansSingle[0] = pFeatures[ind].normal     ;
                s->meansSingle[1] = pFeatures[ind].alb1       ;
                s->meansSingle[2] = pFeatures[ind].alb2       ;
                s->meansSingle[3] = pFeatures[ind].worldPos   ;
                s->meansSingle[4] = pFeatures[ind].directLight;

                // K single std devs, same as denosiing inf values
                for (c=0; c<3; c++) {
				    s->sdSingle[0]    += sqrt(denoisingInf[ind].stdDevVecs[1][c]/sampleCount);
				    s->sdSingle[1]    += sqrt(denoisingInf[ind].stdDevVecs[2][c]/sampleCount);
				    s->sdSingle[2]    += sqrt(denoisingInf[ind].stdDevVecs[3][c]/sampleCount);
				    s->sdSingle[3]    += sqrt(denoisingInf[ind].stdDevVecs[4][c]/sampleCount);
                }
                s->sdSingle[4] += sqrt(denoisingInf[ind].stdDevVecs[5][0]/sampleCount);

                // K block means
                for ( j = -3; j <= 3; j++) { 
                    jFixed = jMain + j < 0 ? 0 : (jMain + j >= yRes ? yRes-1 : jMain + j);
                    for ( i = -3; i <= 3; i++) { 
                        iFixed = iMain + i < 0 ? 0 : (iMain + i >= xRes ? xRes-1 : iMain + i);
                        ijInd = jFixed*xRes + iFixed;

                        s->meansBlock[0] += pFeatures[ijInd].normal      / 49.0f; 
                        s->meansBlock[1] += pFeatures[ijInd].alb1        / 49.0f; 
                        s->meansBlock[2] += pFeatures[ijInd].alb2        / 49.0f; 
                        s->meansBlock[3] += pFeatures[ijInd].worldPos    / 49.0f; 
                        s->meansBlock[4] += pFeatures[ijInd].directLight / 49.0f; 

                        // 3x3 means for mean deviation
                        if (abs(j) <= 1 && abs(i) <= 1) {
                            meansForMD[0] += pFeatures[ijInd].normal      / 9.0f; 
                            meansForMD[1] += pFeatures[ijInd].alb1        / 9.0f; 
                            meansForMD[2] += pFeatures[ijInd].alb2        / 9.0f; 
                            meansForMD[3] += pFeatures[ijInd].worldPos    / 9.0f; 
                            meansForMD[4] += pFeatures[ijInd].directLight / 9.0f; 
                        }
                    }
                }
                // K block std dev (std dev of col to mean col, or avg of std dev of each ? )
                for ( j = -3; j <= 3; j++) { 
                    jFixed = jMain + j < 0 ? 0 : (jMain + j >= yRes ? yRes-1 : jMain + j);
                    for ( i = -3; i <= 3; i++) { 
                        iFixed = iMain + i < 0 ? 0 : (iMain + i >= xRes ? xRes-1 : iMain + i);
                        ijInd = jFixed*xRes + iFixed;

                        s->sdBlock[0] += pow((pFeatures[ijInd].normal      - s->meansBlock[0]),2)/ 49.0f; 
                        s->sdBlock[1] += pow((pFeatures[ijInd].alb1        - s->meansBlock[1]),2)/ 49.0f; 
                        s->sdBlock[2] += pow((pFeatures[ijInd].alb2        - s->meansBlock[2]),2)/ 49.0f; 
                        s->sdBlock[3] += pow((pFeatures[ijInd].worldPos    - s->meansBlock[3]),2)/ 49.0f; 
                        s->sdBlock[4] += pow((pFeatures[ijInd].directLight - s->meansBlock[4]),2)/ 49.0f; 
                    }
                }
                s->sdBlock[0] = sqrt(s->sdBlock[0]);
                s->sdBlock[1] = sqrt(s->sdBlock[1]);
                s->sdBlock[2] = sqrt(s->sdBlock[2]);
                s->sdBlock[3] = sqrt(s->sdBlock[3]);
                s->sdBlock[4] = sqrt(s->sdBlock[4]);
                // K Gradients (3x3 Block)
                // K Mean Deviations (3x3 block)
                for ( j = -1; j <= 1; j++) { 
                    jFixed = jMain + j < 0 ? 0 : (jMain + j >= yRes ? yRes-1 : jMain + j);
                    for ( i = -1; i <= 1; i++) { 
                        iFixed = iMain + i < 0 ? 0 : (iMain + i >= xRes ? xRes-1 : iMain + i);
                        ijInd = jFixed*xRes + iFixed;

                        linearInd = (j+1)*3 + i + 1;

                        // Sobel operator
                        GxSum[0] += Gx[linearInd]*pFeatures[ijInd].normal;
                        GySum[0] += Gy[linearInd]*pFeatures[ijInd].normal;
                        GxSum[1] += Gx[linearInd]*pFeatures[ijInd].alb1;
                        GySum[1] += Gy[linearInd]*pFeatures[ijInd].alb1;
                        GxSum[2] += Gx[linearInd]*pFeatures[ijInd].alb2;
                        GySum[2] += Gy[linearInd]*pFeatures[ijInd].alb2;
                        GxSum[3] += Gx[linearInd]*pFeatures[ijInd].worldPos;
                        GySum[3] += Gy[linearInd]*pFeatures[ijInd].worldPos;
                        GxSum[4] += Gx[linearInd]*pFeatures[ijInd].directLight;
                        GySum[4] += Gy[linearInd]*pFeatures[ijInd].directLight;

                        // Mean Abs Diff
                        s->meanDeviation[0] += fabs(pFeatures[ijInd].normal      - meansForMD[0])/ 9.0f; 
                        s->meanDeviation[1] += fabs(pFeatures[ijInd].alb1        - meansForMD[1])/ 9.0f; 
                        s->meanDeviation[2] += fabs(pFeatures[ijInd].alb2        - meansForMD[2])/ 9.0f; 
                        s->meanDeviation[3] += fabs(pFeatures[ijInd].worldPos    - meansForMD[3])/ 9.0f; 
                        s->meanDeviation[4] += fabs(pFeatures[ijInd].directLight - meansForMD[4])/ 9.0f; 

                        // Collect MAD Values
                        valuesForMAD[0][linearInd] = pFeatures[ijInd].normal     ;
                        valuesForMAD[1][linearInd] = pFeatures[ijInd].alb1       ;
                        valuesForMAD[2][linearInd] = pFeatures[ijInd].alb2       ;
                        valuesForMAD[3][linearInd] = pFeatures[ijInd].worldPos   ;
                        valuesForMAD[4][linearInd] = pFeatures[ijInd].directLight;
                    }
                }
                s->gradients[0] = sqrt(GxSum[0]*GxSum[0] + GySum[0]*GySum[0]);
                s->gradients[1] = sqrt(GxSum[1]*GxSum[1] + GySum[1]*GySum[1]);
                s->gradients[2] = sqrt(GxSum[2]*GxSum[2] + GySum[2]*GySum[2]);
                s->gradients[3] = sqrt(GxSum[3]*GxSum[3] + GySum[3]*GySum[3]);
                s->gradients[4] = sqrt(GxSum[4]*GxSum[4] + GySum[4]*GySum[4]);
                // MAD
                for (int feature = 0; feature < 5; feature++) {

                    // Reset median getter
                    medianGetter[0] = 0.0f;
                    medianGetter[1] = 0.0f;
                    medianGetter[2] = 0.0f;
                    medianGetter[3] = 0.0f;
                    medianGetter[4] = 0.0f;

                    // Get median
                    for (int v = 0; v < 9; v++) {
                        for (int m = 0; m < 5; m++) {
                            if (valuesForMAD[feature][v] > medianGetter[m]) {
                                if (m == 4) {
                                    // Shift all values down
                                    medianGetter[3] = medianGetter[4];
                                    medianGetter[2] = medianGetter[3];
                                    medianGetter[1] = medianGetter[2];
                                    medianGetter[0] = medianGetter[1];

                                    medianGetter[4] = valuesForMAD[feature][v];
                                }
                            }
                            else if (m>0) {
                                medianGetter[m-1] = valuesForMAD[feature][v];
                                break;
                            }
                        }
                    }

                    // Adjust values
                    for (int v = 0; v < 9; v++) {
                        valuesForMAD[feature][v] = fabs(valuesForMAD[feature][v] - medianGetter[0]);
                    }

                    // Reset median getter
                    medianGetter[0] = 0.0f;
                    medianGetter[1] = 0.0f;
                    medianGetter[2] = 0.0f;
                    medianGetter[3] = 0.0f;
                    medianGetter[4] = 0.0f;

                    // Get median again
                    for (int v = 0; v < 9; v++) {
                        for (int m = 0; m < 5; m++) {
                            if (valuesForMAD[feature][v] > medianGetter[m]) {
                                if (m == 4) {
                                    // Shift all values down
                                    medianGetter[3] = medianGetter[4];
                                    medianGetter[2] = medianGetter[3];
                                    medianGetter[1] = medianGetter[2];
                                    medianGetter[0] = medianGetter[1];

                                    medianGetter[4] = valuesForMAD[feature][v];
                                }
                            }
                            else if (m>0) {
                                for (int mm=m-1;mm>=0;mm--)
                                    medianGetter[mm] = medianGetter[mm+1];
                                medianGetter[m] = valuesForMAD[feature][v];
                                break;
                            }
                        }
                    }
                    // Final MAD value
                    s->MAD[feature] = medianGetter[0];
                }
                // L
                s->L = 1.0f/sampleCount;

            }
        }
}
float actFunc(float in) {
    return 1.0f/(1.0f + exp(-in));
}
float softPlus(float in) {
    return log(1.0f + exp(in));
}
void DenoiserNN::ComputeVariances() {
    int node, weight, numNodes;
    int totalPixels = xRes*yRes;
    DenoisingInf* info;

    for (int pixel = 0; pixel < totalPixels; pixel++) {

        // Layer 1 - 2
        for (node=0; node<10; node++) {
            layerTwoValues[10*pixel + node] = 0.0f;
            for (weight=0; weight<36; weight++) 
                layerTwoValues[10*pixel + node] += sFeatures[pixel](weight)*onetwo[36*node + weight];
            layerTwoValues[10*pixel + node] = actFunc(layerTwoValues[10*pixel + node]);
        }

        // Layer 2 - 3
        for (node=0; node<10; node++) {
            layerThreeValues[10*pixel + node] = 0.0f;
            for (weight=0; weight<10; weight++)
                layerThreeValues[10*pixel + node] += layerTwoValues[10*pixel + weight]*twothree[10*node + weight];
            layerThreeValues[10*pixel + node] = actFunc(layerThreeValues[10*pixel + node]);
        }

        // Layer 3 - 4
        info = &denoisingInf[pixel];
        for (node=0; node<7; node++) {
            info->variances[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                info->variances[node] += layerThreeValues[10*pixel + weight]*threefour[10*node + weight];
            info->variances[node] = softPlus(info->variances[node]);
        }


    }
}
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

float DenoiserNN::GetFilterDerivative(int j, int i, int var) {

    // Contribution to the filtered colour of the filter paramater

    int tW = yRes;
    int tH = xRes;
    int N = denoisingN;
    int iIndex = j*tW + i;
    int jIndex;
    int dVal;
    DenoisingInf info, infoj;
    vec3 vecSum;
    info = denoisingInf[iIndex];
    float fDeriv, weightOverParam = 0.0f;

    float dVals[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dValMult = 1.0f;

    // For all pixels j around i
    for (int j1 = -N; j1 <= N; j1++) {
        for (int i1 = -N; i1 <= N; i1++) {

            if (j1==j && i1==i)
                continue;

            jIndex = j+j1<0 ? 0 : (j+j1>=tH ? tH-1 : j+j1);
            jIndex *= tW;
            jIndex += i+i1<0 ? 0 : (i+i1>=tH ? tW-1 : i+i1);

            infoj = denoisingInf[jIndex];

            vecSum = (preScreen[jIndex]*info.wcSum - denoisedCol[iIndex]) / info.wcSum;

            // Index d value
            dVals[0] = (pow(j1-j,2)+pow(i1-i,2)) / (2.0f * info.variances[0] + 0.000001f);
            // Colour d value
            dVals[1] = (pow(preScreen[iIndex].x - preScreen[jIndex].x,2)+pow(preScreen[iIndex].y - preScreen[jIndex].y,2) + pow(preScreen[iIndex].z - preScreen[jIndex].z,2)) 
            / (2.0f * info.variances[1] * info.stdDevVecs[0].sumDiv(sampleCount) + infoj.stdDevVecs[0].sumDiv(sampleCount)  + 0.000001f);
            // Normal d value
            dVals[2] = (pow(normal[iIndex].x - normal[jIndex].x,2)+pow(normal[iIndex].y - normal[jIndex].y,2) + pow(normal[iIndex].z - normal[jIndex].z,2)) 
            / (2.0f * info.variances[2] * info.stdDevVecs[1].sumDiv(sampleCount)  + 0.000001f);
            // Alb1 d value
            dVals[3] = (pow(albedo1[iIndex].x - albedo1[jIndex].x,2)+pow(albedo1[iIndex].y - albedo1[jIndex].y,2) + pow(albedo1[iIndex].z - albedo1[jIndex].z,2)) 
            / (2.0f * info.variances[3] * info.stdDevVecs[2].sumDiv(sampleCount)  + 0.000001f);
            // Alb2 d value
            dVals[4] = (pow(albedo2[iIndex].x - albedo2[jIndex].x,2)+pow(albedo2[iIndex].y - albedo2[jIndex].y,2) + pow(albedo2[iIndex].z - albedo2[jIndex].z,2)) 
            / (2.0f * info.variances[4] * info.stdDevVecs[3].sumDiv(sampleCount)  + 0.000001f);
            // worldPos d value
            dVals[5] = (pow(worldPos[iIndex].x - worldPos[jIndex].x,2)+pow(worldPos[iIndex].y - worldPos[jIndex].y,2) + pow(worldPos[iIndex].z - worldPos[jIndex].z,2)) 
            / (2.0f * info.variances[5] * info.stdDevVecs[4].sumDiv(sampleCount)  + 0.000001f);
            // directLight d value
            dVals[6] = pow(directLight[iIndex].x - directLight[jIndex].x,2) / (2.0f * info.variances[6] * info.stdDevVecs[5].sumDiv(sampleCount) + 0.000001f);


            for (dVal=0;dVal<7;dVal++) {
                dVals[dVal] += 0.000001f;
                dValMult *= exp(-dVals[dVal]) + 0.000001f;;
            }

            weightOverParam = dValMult * dVals[var] * 2.0f / info.variances[var];

            fDeriv += (vecSum.x + vecSum.y + vecSum.z)*weightOverParam;

        }      
    } 

    return fDeriv;   

}

void DenoiserNN::BackPropWeights(){
    int pixels = xRes*yRes;
    int pixel, var, w;
    float errorOverColour, colourOverParam, paramOverWeight;
    vec3 tCol;

    // All pixels
    for (int j=0;j<yRes;j++) {
        for (int i=0; i<xRes; i++) {

            pixel = j*xRes + i;
            tCol = targetCol[pixel];

            // Derivative One: samples * (cFiltered - cTarget)/(cTarget*cTarget)
            errorOverColour = 0.0f;
            errorOverColour += sampleCount * (denoisedCol[pixel].x - tCol.x) / (tCol.x*tCol.x+0.0001f);
            errorOverColour += sampleCount * (denoisedCol[pixel].y - tCol.y) / (tCol.y*tCol.y+0.0001f);
            errorOverColour += sampleCount * (denoisedCol[pixel].z - tCol.z) / (tCol.z*tCol.z+0.0001f);

            // Filter Paramaters (index, col, K)
            for (var=0;var<7;var++) {

                // Derivative Two: cross-bilateral filter derivative
                colourOverParam = GetFilterDerivative(j, i, var);

                // Weights One
                for (w=0;w<360;w++){
                    // Derivative Three: filter/weight = secondary feature input at weight index
                    paramOverWeight = sFeatures[pixel](w % 36);
                    onetwo[w] -= learningRate*errorOverColour*colourOverParam*paramOverWeight;
                }
                // Weights Two
                for (w=0;w<100;w++){
                    // Derivative Three: filter/weight = second layer value at weight index
                    paramOverWeight = layerTwoValues[10*pixel + w % 10];
                    twothree[w] -= learningRate*errorOverColour*colourOverParam*paramOverWeight;
                }
                // Weights Three
                for (w=0;w<70;w++){
                    // Derivative Three: filter/weight = third layer value at weight index
                    paramOverWeight = layerThreeValues[10*pixel + w % 10];
                    threefour[w] -= learningRate*errorOverColour*colourOverParam*paramOverWeight;
                }
            }
        }
    }
}

void DenoiserNN::DeleteScreens() {
    delete preScreen ;
    delete postScreen ;
    delete normal ;
    delete albedo1 ;
    delete albedo2 ;
    delete directLight;
    delete worldPos;
    delete denoisedCol ;
    //delete targetCol;
    delete layerTwoValues;
    delete layerThreeValues;
}

void DenoiserNN::InitScreens() {
    preScreen = new vec3[xRes*yRes];
    postScreen = new vec3[xRes*yRes];
    normal = new vec3[xRes*yRes];
    albedo1 = new vec3[xRes*yRes];
    albedo2 = new vec3[xRes*yRes];
    directLight = new vec3[xRes*yRes];
    worldPos = new vec3[xRes*yRes];
    denoisedCol = new vec3[xRes*yRes];
    //targetCol = new vec3[xRes*yRes];
    denoisingInf = new DenoisingInf[xRes*yRes];

    layerTwoValues = new float[10*xRes*yRes];
    layerThreeValues = new float[10*xRes*yRes];
}

void DenoiserNN::InitTraining() {

    training = true;

    // Get Target
	denoiser.saveTargetCol();
	// Load Current Values
	if (!LoadWeights())
		RandomizeWeights();

    trainingEpoch = 0;
}
void DenoiserNN::EndTraining() {

    training = false;

	// Save Trained Weights
	OutputWeights();
}

void DenoiserNN::TrainNN() {

            // Reset sample count
            sampleCount = 0;

            // Initialize
            DeleteScreens();
			InitScreens();

			// Gen new image
            for (int sample = 0; sample < samplesWhenTraining; sample++)
			    renderer.Render();

			// Forward Prop
			GeneratePFeatures();
		    GenerateSFeatures();
			ComputeVariances();

            // Denoise
            denoiser.CPUDenoise();

			// Get Error Value
			GenRelMSE();

			// Back Prop
            BackPropWeights();

            // Epoch
            trainingEpoch++;

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
	const uint64_t s0 = renderer.constants.GloRandS[0];
	uint64_t s1 = renderer.constants.GloRandS[1];
	const uint64_t result_plus = rotl(s0 + s1, R) + s0;
	s1 ^= s0;
	renderer.constants.GloRandS[0] = rotl(s0, A) ^ s1 ^ (s1 << B);
	renderer.constants.GloRandS[1] = rotl(s1, C);
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
void DenoiserNN::OutputWeights() {
    std::ofstream oFile("NNWeights.txt");
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
bool DenoiserNN::LoadWeights() {
    std::ifstream rFile("NNWeights.txt");
    std::string newLine;
    int ind;

    if (!rFile.is_open()){
        std::cout << "Cannot open NNWeights.txt!" << std::endl;
        return false;
    }

    // Line 1
    if (getline(rFile, newLine)) {
        std::istringstream in(newLine);
        for (ind = 0; ind < 360; ind++)
            in >> onetwo[ind];
    } else {std::cout << "Invalid NNWeights.txt file! " << std::endl; return false;}

    // Line 2
    if (getline(rFile, newLine)) {
        std::istringstream in(newLine);
        for (ind = 0; ind < 100; ind++)
            in >> twothree[ind];
    } else {std::cout << "Invalid NNWeights.txt file! " << std::endl; return false;}

    // Line 3
    if (getline(rFile, newLine)) {
        std::istringstream in(newLine);
        for (ind = 0; ind < 70; ind++)
            in >> threefour[ind];
    } else {std::cout << "Invalid NNWeights.txt file! " << std::endl; return false;}

    return true;
}
