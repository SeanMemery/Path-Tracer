#define SKEPU_PRECOMPILED 1
#define SKEPU_OPENMP 1
#define SKEPU_CUDA 1
#include "DenoiserNN.h"

float actFunc(float in) {
    return 1.0f/(1.0f + exp(-in));
}
float softPlus(float in) {
    return log(1.0f + exp(in));
}

// Forward Prop Functions

void DenoiserNN::CPUForwardProp() {
    int totalPixels = xRes*yRes;

    // Primary Features
    {
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
                pFeatures[ind].normal   += normal[ind][c]   / (3.0f * sampleCount);
                pFeatures[ind].alb1     += albedo1[ind][c]  / (3.0f * sampleCount);
                pFeatures[ind].alb2     += albedo2[ind][c]  / (3.0f * sampleCount);
                pFeatures[ind].worldPos += worldPos[ind][c] / (3.0f * sampleCount);
            }
            pFeatures[ind].directLight = directLight[ind].x / sampleCount;
        }
    }


    // Secondary Features
    {
        // - K Means of single pixel, K std deviations of single pixel, K Means of 7x7 block, K std deviations of 7x7 block (20 total)
        // - Magnitude of gradients of K features of single pixel (sobel operator) (5 total)
        // - Sum of the abs difference between K of each pixel in 3x3 block and the mean of that 3x3 block (5 total)
        // - MAD of K features, so median of values minus median value, in NxN block (5 total)
        // - 1/totalSamples (1 total)

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
        float medianGetter[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        int K, i, j;
        for (int jMain = 0; jMain < yRes; jMain++) { 
            for (int iMain = 0; iMain < xRes; iMain++) { 

                float meansForMD[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float GxSum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float GySum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};       

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
			    s->sdSingle[0] = denoisingInf[ind].stdDev[1];
			    s->sdSingle[1] = denoisingInf[ind].stdDev[2];
			    s->sdSingle[2] = denoisingInf[ind].stdDev[3];
			    s->sdSingle[3] = denoisingInf[ind].stdDev[4];
                s->sdSingle[4] = denoisingInf[ind].stdDev[5];

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
    
    // Variances
    {
        int node, weight, numNodes;
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

}
void DenoiserNN::OMPForwardProp() {
    int totalPixels = xRes*yRes;

    // Primary Features
    {
        delete pFeatures;
        pFeatures = new PrimaryFeatures[totalPixels];

        #pragma omp parallel for 
        for (int ind = 0; ind < totalPixels; ind++) {
            pFeatures[ind].normal      = 0.0f;
            pFeatures[ind].alb1        = 0.0f;
            pFeatures[ind].alb2        = 0.0f;
            pFeatures[ind].worldPos    = 0.0f;
            pFeatures[ind].directLight = 0.0f;

            for (int c = 0; c < 3; c++) {
                pFeatures[ind].normal   += normal[ind][c]   / (3.0f * sampleCount);
                pFeatures[ind].alb1     += albedo1[ind][c]  / (3.0f * sampleCount);
                pFeatures[ind].alb2     += albedo2[ind][c]  / (3.0f * sampleCount);
                pFeatures[ind].worldPos += worldPos[ind][c] / (3.0f * sampleCount);
            }
            pFeatures[ind].directLight = directLight[ind].x / sampleCount;
        }
    }


    // Secondary Features
    {
        // - K Means of single pixel, K std deviations of single pixel, K Means of 7x7 block, K std deviations of 7x7 block (20 total)
        // - Magnitude of gradients of K features of single pixel (sobel operator) (5 total)
        // - Sum of the abs difference between K of each pixel in 3x3 block and the mean of that 3x3 block (5 total)
        // - MAD of K features, so median of values minus median value, in NxN block (5 total)
        // - 1/totalSamples (1 total)

        delete sFeatures;
        sFeatures = new SecondaryFeatures[totalPixels];

        float Gx[9] = { 1,  2,  1, 
                        0,  0,  0, 
                       -1, -2, -1};
        float Gy[9] = { 1,  0,  -1, 
                        2,  0,  -2, 
                        1,  0,  -1};

        #pragma omp parallel for 
        for (int jMain = 0; jMain < yRes; jMain++) {
            #pragma omp parallel for  
            for (int iMain = 0; iMain < xRes; iMain++) { 

                int ind, c = 0;
                int ijInd = 0;
                int jFixed, iFixed = 0;
                SecondaryFeatures* s;

                int linearInd = 0;
                float valuesForMAD[5][9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
                                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
                                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
                                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
                                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float medianGetter[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                int K, i, j;

                float meansForMD[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float GxSum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float GySum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};       

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
			    s->sdSingle[0] = denoisingInf[ind].stdDev[1];
			    s->sdSingle[1] = denoisingInf[ind].stdDev[2];
			    s->sdSingle[2] = denoisingInf[ind].stdDev[3];
			    s->sdSingle[3] = denoisingInf[ind].stdDev[4];
                s->sdSingle[4] = denoisingInf[ind].stdDev[5];

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
    
    // Variances
    {

        #pragma omp parallel for
        for (int pixel = 0; pixel < totalPixels; pixel++) {

            int node, weight, numNodes;
            DenoisingInf* info;

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

}
static ForPropOut SkePUFPFunc(skepu::Region2D<ForPropIn> r, SkePUFPConstants sConstants) {


    // Secondary Features
    ForPropOut out;
    {
        // - K Means of single pixel, K std deviations of single pixel, K Means of 7x7 block, K std deviations of 7x7 block (20 total)
        // - Magnitude of gradients of K features of single pixel (sobel operator) (5 total)
        // - Sum of the abs difference between K of each pixel in 3x3 block and the mean of that 3x3 block (5 total)
        // - MAD of K features, so median of values minus median value, in NxN block (5 total)
        // - 1/totalSamples (1 total)

        int ind, c = 0;
        int ijInd = 0;
        int jFixed, iFixed = 0;
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
        float medianGetter[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        int i, j;

        float meansForMD[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float GxSum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float GySum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; 

        for (c=0;c<10;c++) {
            out.l2[c] = 0.0f;
            out.l3[c] = 0.0f;
        }     
        for (c=0;c<7;c++) 
            out.variances[c] = 0.0f;
        for (c=0; c<5; c++) {
            out.meansSingle[c] = 0.0f;
            out.sdSingle[c] = 0.0f;
            out.meansBlock[c] = 0.0f;
            out.sdBlock[c] = 0.0f;
            out.gradients[c] = 0.0f;
            out.meanDeviation[c] = 0.0f;
            out.MAD[c] = 0.0f; 
        }
        out.L = 0.0f;

        // K single Means
        out.meansSingle[0] = r(0,0).normal;
        out.meansSingle[1] = r(0,0).alb1;
        out.meansSingle[2] = r(0,0).alb2;
        out.meansSingle[3] = r(0,0).worldPos;
        out.meansSingle[4] = r(0,0).directLight;

        // K single std devs
		out.sdSingle[0] = r(0,0).stdDev[1];
		out.sdSingle[1] = r(0,0).stdDev[2];
		out.sdSingle[2] = r(0,0).stdDev[3];
		out.sdSingle[3] = r(0,0).stdDev[4];
        out.sdSingle[4] = r(0,0).stdDev[5];

        // K block means
        for ( j = -3; j <= 3; j++) { 
            for ( i = -3; i <= 3; i++) { 
                out.meansBlock[0] += r(j, i).normal      / 49.0f; 
                out.meansBlock[1] += r(j, i).alb1        / 49.0f; 
                out.meansBlock[2] += r(j, i).alb2        / 49.0f; 
                out.meansBlock[3] += r(j, i).worldPos    / 49.0f; 
                out.meansBlock[4] += r(j, i).directLight / 49.0f; 
                // 3x3 means for mean deviation
                if (abs(j) <= 1 && abs(i) <= 1) {
                    meansForMD[0] += r(j, i).normal      / 9.0f; 
                    meansForMD[1] += r(j, i).alb1        / 9.0f; 
                    meansForMD[2] += r(j, i).alb2        / 9.0f; 
                    meansForMD[3] += r(j, i).worldPos    / 9.0f; 
                    meansForMD[4] += r(j, i).directLight / 9.0f; 
                }
            }
        }
        // K block std dev (std dev of col to mean col, or avg of std dev of each ? )
        for ( j = -3; j <= 3; j++) { 
            for ( i = -3; i <= 3; i++) { 
                out.sdBlock[0] += pow((r(j,i).normal      - out.meansBlock[0]),2); 
                out.sdBlock[1] += pow((r(j,i).alb1        - out.meansBlock[1]),2); 
                out.sdBlock[2] += pow((r(j,i).alb2        - out.meansBlock[2]),2); 
                out.sdBlock[3] += pow((r(j,i).worldPos    - out.meansBlock[3]),2); 
                out.sdBlock[4] += pow((r(j,i).directLight - out.meansBlock[4]),2); 
            }
        }
        out.sdBlock[0] = sqrt(out.sdBlock[0] / 49.0f);
        out.sdBlock[1] = sqrt(out.sdBlock[1] / 49.0f);
        out.sdBlock[2] = sqrt(out.sdBlock[2] / 49.0f);
        out.sdBlock[3] = sqrt(out.sdBlock[3] / 49.0f);
        out.sdBlock[4] = sqrt(out.sdBlock[4] / 49.0f);
        // K Gradients (3x3 Block)
        // K Mean Deviations (3x3 block)
        for ( j = -1; j <= 1; j++) { 
            for ( i = -1; i <= 1; i++) { 
                linearInd = (j+1)*3 + i + 1;
                // Sobel operator
                GxSum[0] += Gx[linearInd]*r(j,i).normal;
                GySum[0] += Gy[linearInd]*r(j,i).normal;
                GxSum[1] += Gx[linearInd]*r(j,i).alb1;
                GySum[1] += Gy[linearInd]*r(j,i).alb1;
                GxSum[2] += Gx[linearInd]*r(j,i).alb2;
                GySum[2] += Gy[linearInd]*r(j,i).alb2;
                GxSum[3] += Gx[linearInd]*r(j,i).worldPos;
                GySum[3] += Gy[linearInd]*r(j,i).worldPos;
                GxSum[4] += Gx[linearInd]*r(j,i).directLight;
                GySum[4] += Gy[linearInd]*r(j,i).directLight;

                // Mean Abs Diff
                out.meanDeviation[0] += fabs(r(j,i).normal      - meansForMD[0])/ 9.0f; 
                out.meanDeviation[1] += fabs(r(j,i).alb1        - meansForMD[1])/ 9.0f; 
                out.meanDeviation[2] += fabs(r(j,i).alb2        - meansForMD[2])/ 9.0f; 
                out.meanDeviation[3] += fabs(r(j,i).worldPos    - meansForMD[3])/ 9.0f; 
                out.meanDeviation[4] += fabs(r(j,i).directLight - meansForMD[4])/ 9.0f; 

                // Collect MAD Values
                valuesForMAD[0][linearInd] = r(j,i).normal     ;
                valuesForMAD[1][linearInd] = r(j,i).alb1       ;
                valuesForMAD[2][linearInd] = r(j,i).alb2       ;
                valuesForMAD[3][linearInd] = r(j,i).worldPos   ;
                valuesForMAD[4][linearInd] = r(j,i).directLight;
            }
        }
        out.gradients[0] = sqrt(GxSum[0]*GxSum[0] + GySum[0]*GySum[0]);
        out.gradients[1] = sqrt(GxSum[1]*GxSum[1] + GySum[1]*GySum[1]);
        out.gradients[2] = sqrt(GxSum[2]*GxSum[2] + GySum[2]*GySum[2]);
        out.gradients[3] = sqrt(GxSum[3]*GxSum[3] + GySum[3]*GySum[3]);
        out.gradients[4] = sqrt(GxSum[4]*GxSum[4] + GySum[4]*GySum[4]);
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
            out.MAD[feature] = medianGetter[0];
        }
        // L
        out.L = 1.0f/sConstants.samples;

    }
    
    // Variances
    {
        int node, weight, numNodes;

        // Layer 1 - 2
        for (node=0; node<10; node++) {
            out.l2[node] = 0.0f;

            // Go through all secondary features
            {
                out.l2[node] += out.meansSingle[0]   * sConstants.onetwo[36*node + 0]; 
                out.l2[node] += out.meansSingle[1]   * sConstants.onetwo[36*node + 1]; 
                out.l2[node] += out.meansSingle[2]   * sConstants.onetwo[36*node + 2]; 
                out.l2[node] += out.meansSingle[3]   * sConstants.onetwo[36*node + 3]; 
                out.l2[node] += out.meansSingle[4]   * sConstants.onetwo[36*node + 4]; 
                out.l2[node] += out.sdSingle[0]      * sConstants.onetwo[36*node + 5];
                out.l2[node] += out.sdSingle[1]      * sConstants.onetwo[36*node + 6];
                out.l2[node] += out.sdSingle[2]      * sConstants.onetwo[36*node + 7];
                out.l2[node] += out.sdSingle[3]      * sConstants.onetwo[36*node + 8];
                out.l2[node] += out.sdSingle[4]      * sConstants.onetwo[36*node + 9];
                out.l2[node] += out.meansBlock[0]    * sConstants.onetwo[36*node + 10];
                out.l2[node] += out.meansBlock[1]    * sConstants.onetwo[36*node + 11];
                out.l2[node] += out.meansBlock[2]    * sConstants.onetwo[36*node + 12];
                out.l2[node] += out.meansBlock[3]    * sConstants.onetwo[36*node + 13];
                out.l2[node] += out.meansBlock[4]    * sConstants.onetwo[36*node + 14];
                out.l2[node] += out.sdBlock[0]       * sConstants.onetwo[36*node + 15];
                out.l2[node] += out.sdBlock[1]       * sConstants.onetwo[36*node + 16];
                out.l2[node] += out.sdBlock[2]       * sConstants.onetwo[36*node + 17];
                out.l2[node] += out.sdBlock[3]       * sConstants.onetwo[36*node + 18];
                out.l2[node] += out.sdBlock[4]       * sConstants.onetwo[36*node + 19];
                out.l2[node] += out.gradients[0]     * sConstants.onetwo[36*node + 20];
                out.l2[node] += out.gradients[1]     * sConstants.onetwo[36*node + 21];
                out.l2[node] += out.gradients[2]     * sConstants.onetwo[36*node + 22];
                out.l2[node] += out.gradients[3]     * sConstants.onetwo[36*node + 23];
                out.l2[node] += out.gradients[4]     * sConstants.onetwo[36*node + 24];
                out.l2[node] += out.meanDeviation[0] * sConstants.onetwo[36*node + 25];
                out.l2[node] += out.meanDeviation[1] * sConstants.onetwo[36*node + 26];
                out.l2[node] += out.meanDeviation[2] * sConstants.onetwo[36*node + 27];
                out.l2[node] += out.meanDeviation[3] * sConstants.onetwo[36*node + 28];
                out.l2[node] += out.meanDeviation[4] * sConstants.onetwo[36*node + 29];
                out.l2[node] += out.MAD[0]           * sConstants.onetwo[36*node + 30];
                out.l2[node] += out.MAD[1]           * sConstants.onetwo[36*node + 31];
                out.l2[node] += out.MAD[2]           * sConstants.onetwo[36*node + 32];
                out.l2[node] += out.MAD[3]           * sConstants.onetwo[36*node + 33];
                out.l2[node] += out.MAD[4]           * sConstants.onetwo[36*node + 34];
                out.l2[node] += out.L                * sConstants.onetwo[36*node + 35];
            }
            
            out.l2[node] = 1.0f/(1.0f + exp(-out.l2[node]));
        }

        // Layer 2 - 3
        for (node=0; node<10; node++) {
            out.l3[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                out.l3[node] += out.l2[weight]*sConstants.twothree[10*node + weight];
            out.l3[node] = 1.0f/(1.0f + exp(-out.l3[node]));
        }

        // Layer 3 - 4
        for (node=0; node<7; node++) {
            out.variances[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                out.variances[node] += out.l3[weight]*sConstants.threefour[10*node + weight];
            out.variances[node] = log(1.0f + exp(out.variances[node]));
        }

    }

    return out;    
}

struct skepu_userfunction_skepu_skel_2convol_SkePUFPFunc
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
constexpr static bool usesPRNG = 0;
constexpr static size_t randomCount = SKEPU_NO_RANDOM;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<SkePUFPConstants>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = ForPropOut;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ ForPropOut CU(skepu::Region2D<ForPropIn> r, SkePUFPConstants sConstants)
{


    // Secondary Features
    ForPropOut out;
    {
        // - K Means of single pixel, K std deviations of single pixel, K Means of 7x7 block, K std deviations of 7x7 block (20 total)
        // - Magnitude of gradients of K features of single pixel (sobel operator) (5 total)
        // - Sum of the abs difference between K of each pixel in 3x3 block and the mean of that 3x3 block (5 total)
        // - MAD of K features, so median of values minus median value, in NxN block (5 total)
        // - 1/totalSamples (1 total)

        int ind, c = 0;
        int ijInd = 0;
        int jFixed, iFixed = 0;
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
        float medianGetter[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        int i, j;

        float meansForMD[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float GxSum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float GySum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; 

        for (c=0;c<10;c++) {
            out.l2[c] = 0.0f;
            out.l3[c] = 0.0f;
        }     
        for (c=0;c<7;c++) 
            out.variances[c] = 0.0f;
        for (c=0; c<5; c++) {
            out.meansSingle[c] = 0.0f;
            out.sdSingle[c] = 0.0f;
            out.meansBlock[c] = 0.0f;
            out.sdBlock[c] = 0.0f;
            out.gradients[c] = 0.0f;
            out.meanDeviation[c] = 0.0f;
            out.MAD[c] = 0.0f; 
        }
        out.L = 0.0f;

        // K single Means
        out.meansSingle[0] = r(0,0).normal;
        out.meansSingle[1] = r(0,0).alb1;
        out.meansSingle[2] = r(0,0).alb2;
        out.meansSingle[3] = r(0,0).worldPos;
        out.meansSingle[4] = r(0,0).directLight;

        // K single std devs
		out.sdSingle[0] = r(0,0).stdDev[1];
		out.sdSingle[1] = r(0,0).stdDev[2];
		out.sdSingle[2] = r(0,0).stdDev[3];
		out.sdSingle[3] = r(0,0).stdDev[4];
        out.sdSingle[4] = r(0,0).stdDev[5];

        // K block means
        for ( j = -3; j <= 3; j++) { 
            for ( i = -3; i <= 3; i++) { 
                out.meansBlock[0] += r(j, i).normal      / 49.0f; 
                out.meansBlock[1] += r(j, i).alb1        / 49.0f; 
                out.meansBlock[2] += r(j, i).alb2        / 49.0f; 
                out.meansBlock[3] += r(j, i).worldPos    / 49.0f; 
                out.meansBlock[4] += r(j, i).directLight / 49.0f; 
                // 3x3 means for mean deviation
                if (abs(j) <= 1 && abs(i) <= 1) {
                    meansForMD[0] += r(j, i).normal      / 9.0f; 
                    meansForMD[1] += r(j, i).alb1        / 9.0f; 
                    meansForMD[2] += r(j, i).alb2        / 9.0f; 
                    meansForMD[3] += r(j, i).worldPos    / 9.0f; 
                    meansForMD[4] += r(j, i).directLight / 9.0f; 
                }
            }
        }
        // K block std dev (std dev of col to mean col, or avg of std dev of each ? )
        for ( j = -3; j <= 3; j++) { 
            for ( i = -3; i <= 3; i++) { 
                out.sdBlock[0] += pow((r(j,i).normal      - out.meansBlock[0]),2); 
                out.sdBlock[1] += pow((r(j,i).alb1        - out.meansBlock[1]),2); 
                out.sdBlock[2] += pow((r(j,i).alb2        - out.meansBlock[2]),2); 
                out.sdBlock[3] += pow((r(j,i).worldPos    - out.meansBlock[3]),2); 
                out.sdBlock[4] += pow((r(j,i).directLight - out.meansBlock[4]),2); 
            }
        }
        out.sdBlock[0] = sqrt(out.sdBlock[0] / 49.0f);
        out.sdBlock[1] = sqrt(out.sdBlock[1] / 49.0f);
        out.sdBlock[2] = sqrt(out.sdBlock[2] / 49.0f);
        out.sdBlock[3] = sqrt(out.sdBlock[3] / 49.0f);
        out.sdBlock[4] = sqrt(out.sdBlock[4] / 49.0f);
        // K Gradients (3x3 Block)
        // K Mean Deviations (3x3 block)
        for ( j = -1; j <= 1; j++) { 
            for ( i = -1; i <= 1; i++) { 
                linearInd = (j+1)*3 + i + 1;
                // Sobel operator
                GxSum[0] += Gx[linearInd]*r(j,i).normal;
                GySum[0] += Gy[linearInd]*r(j,i).normal;
                GxSum[1] += Gx[linearInd]*r(j,i).alb1;
                GySum[1] += Gy[linearInd]*r(j,i).alb1;
                GxSum[2] += Gx[linearInd]*r(j,i).alb2;
                GySum[2] += Gy[linearInd]*r(j,i).alb2;
                GxSum[3] += Gx[linearInd]*r(j,i).worldPos;
                GySum[3] += Gy[linearInd]*r(j,i).worldPos;
                GxSum[4] += Gx[linearInd]*r(j,i).directLight;
                GySum[4] += Gy[linearInd]*r(j,i).directLight;

                // Mean Abs Diff
                out.meanDeviation[0] += fabs(r(j,i).normal      - meansForMD[0])/ 9.0f; 
                out.meanDeviation[1] += fabs(r(j,i).alb1        - meansForMD[1])/ 9.0f; 
                out.meanDeviation[2] += fabs(r(j,i).alb2        - meansForMD[2])/ 9.0f; 
                out.meanDeviation[3] += fabs(r(j,i).worldPos    - meansForMD[3])/ 9.0f; 
                out.meanDeviation[4] += fabs(r(j,i).directLight - meansForMD[4])/ 9.0f; 

                // Collect MAD Values
                valuesForMAD[0][linearInd] = r(j,i).normal     ;
                valuesForMAD[1][linearInd] = r(j,i).alb1       ;
                valuesForMAD[2][linearInd] = r(j,i).alb2       ;
                valuesForMAD[3][linearInd] = r(j,i).worldPos   ;
                valuesForMAD[4][linearInd] = r(j,i).directLight;
            }
        }
        out.gradients[0] = sqrt(GxSum[0]*GxSum[0] + GySum[0]*GySum[0]);
        out.gradients[1] = sqrt(GxSum[1]*GxSum[1] + GySum[1]*GySum[1]);
        out.gradients[2] = sqrt(GxSum[2]*GxSum[2] + GySum[2]*GySum[2]);
        out.gradients[3] = sqrt(GxSum[3]*GxSum[3] + GySum[3]*GySum[3]);
        out.gradients[4] = sqrt(GxSum[4]*GxSum[4] + GySum[4]*GySum[4]);
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
            out.MAD[feature] = medianGetter[0];
        }
        // L
        out.L = 1.0f/sConstants.samples;

    }
    
    // Variances
    {
        int node, weight, numNodes;

        // Layer 1 - 2
        for (node=0; node<10; node++) {
            out.l2[node] = 0.0f;

            // Go through all secondary features
            {
                out.l2[node] += out.meansSingle[0]   * sConstants.onetwo[36*node + 0]; 
                out.l2[node] += out.meansSingle[1]   * sConstants.onetwo[36*node + 1]; 
                out.l2[node] += out.meansSingle[2]   * sConstants.onetwo[36*node + 2]; 
                out.l2[node] += out.meansSingle[3]   * sConstants.onetwo[36*node + 3]; 
                out.l2[node] += out.meansSingle[4]   * sConstants.onetwo[36*node + 4]; 
                out.l2[node] += out.sdSingle[0]      * sConstants.onetwo[36*node + 5];
                out.l2[node] += out.sdSingle[1]      * sConstants.onetwo[36*node + 6];
                out.l2[node] += out.sdSingle[2]      * sConstants.onetwo[36*node + 7];
                out.l2[node] += out.sdSingle[3]      * sConstants.onetwo[36*node + 8];
                out.l2[node] += out.sdSingle[4]      * sConstants.onetwo[36*node + 9];
                out.l2[node] += out.meansBlock[0]    * sConstants.onetwo[36*node + 10];
                out.l2[node] += out.meansBlock[1]    * sConstants.onetwo[36*node + 11];
                out.l2[node] += out.meansBlock[2]    * sConstants.onetwo[36*node + 12];
                out.l2[node] += out.meansBlock[3]    * sConstants.onetwo[36*node + 13];
                out.l2[node] += out.meansBlock[4]    * sConstants.onetwo[36*node + 14];
                out.l2[node] += out.sdBlock[0]       * sConstants.onetwo[36*node + 15];
                out.l2[node] += out.sdBlock[1]       * sConstants.onetwo[36*node + 16];
                out.l2[node] += out.sdBlock[2]       * sConstants.onetwo[36*node + 17];
                out.l2[node] += out.sdBlock[3]       * sConstants.onetwo[36*node + 18];
                out.l2[node] += out.sdBlock[4]       * sConstants.onetwo[36*node + 19];
                out.l2[node] += out.gradients[0]     * sConstants.onetwo[36*node + 20];
                out.l2[node] += out.gradients[1]     * sConstants.onetwo[36*node + 21];
                out.l2[node] += out.gradients[2]     * sConstants.onetwo[36*node + 22];
                out.l2[node] += out.gradients[3]     * sConstants.onetwo[36*node + 23];
                out.l2[node] += out.gradients[4]     * sConstants.onetwo[36*node + 24];
                out.l2[node] += out.meanDeviation[0] * sConstants.onetwo[36*node + 25];
                out.l2[node] += out.meanDeviation[1] * sConstants.onetwo[36*node + 26];
                out.l2[node] += out.meanDeviation[2] * sConstants.onetwo[36*node + 27];
                out.l2[node] += out.meanDeviation[3] * sConstants.onetwo[36*node + 28];
                out.l2[node] += out.meanDeviation[4] * sConstants.onetwo[36*node + 29];
                out.l2[node] += out.MAD[0]           * sConstants.onetwo[36*node + 30];
                out.l2[node] += out.MAD[1]           * sConstants.onetwo[36*node + 31];
                out.l2[node] += out.MAD[2]           * sConstants.onetwo[36*node + 32];
                out.l2[node] += out.MAD[3]           * sConstants.onetwo[36*node + 33];
                out.l2[node] += out.MAD[4]           * sConstants.onetwo[36*node + 34];
                out.l2[node] += out.L                * sConstants.onetwo[36*node + 35];
            }
            
            out.l2[node] = 1.0f/(1.0f + exp(-out.l2[node]));
        }

        // Layer 2 - 3
        for (node=0; node<10; node++) {
            out.l3[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                out.l3[node] += out.l2[weight]*sConstants.twothree[10*node + weight];
            out.l3[node] = 1.0f/(1.0f + exp(-out.l3[node]));
        }

        // Layer 3 - 4
        for (node=0; node<7; node++) {
            out.variances[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                out.variances[node] += out.l3[weight]*sConstants.threefour[10*node + weight];
            out.variances[node] = log(1.0f + exp(out.variances[node]));
        }

    }

    return out;    
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE ForPropOut OMP(skepu::Region2D<ForPropIn> r, SkePUFPConstants sConstants)
{


    // Secondary Features
    ForPropOut out;
    {
        // - K Means of single pixel, K std deviations of single pixel, K Means of 7x7 block, K std deviations of 7x7 block (20 total)
        // - Magnitude of gradients of K features of single pixel (sobel operator) (5 total)
        // - Sum of the abs difference between K of each pixel in 3x3 block and the mean of that 3x3 block (5 total)
        // - MAD of K features, so median of values minus median value, in NxN block (5 total)
        // - 1/totalSamples (1 total)

        int ind, c = 0;
        int ijInd = 0;
        int jFixed, iFixed = 0;
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
        float medianGetter[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        int i, j;

        float meansForMD[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float GxSum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float GySum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; 

        for (c=0;c<10;c++) {
            out.l2[c] = 0.0f;
            out.l3[c] = 0.0f;
        }     
        for (c=0;c<7;c++) 
            out.variances[c] = 0.0f;
        for (c=0; c<5; c++) {
            out.meansSingle[c] = 0.0f;
            out.sdSingle[c] = 0.0f;
            out.meansBlock[c] = 0.0f;
            out.sdBlock[c] = 0.0f;
            out.gradients[c] = 0.0f;
            out.meanDeviation[c] = 0.0f;
            out.MAD[c] = 0.0f; 
        }
        out.L = 0.0f;

        // K single Means
        out.meansSingle[0] = r(0,0).normal;
        out.meansSingle[1] = r(0,0).alb1;
        out.meansSingle[2] = r(0,0).alb2;
        out.meansSingle[3] = r(0,0).worldPos;
        out.meansSingle[4] = r(0,0).directLight;

        // K single std devs
		out.sdSingle[0] = r(0,0).stdDev[1];
		out.sdSingle[1] = r(0,0).stdDev[2];
		out.sdSingle[2] = r(0,0).stdDev[3];
		out.sdSingle[3] = r(0,0).stdDev[4];
        out.sdSingle[4] = r(0,0).stdDev[5];

        // K block means
        for ( j = -3; j <= 3; j++) { 
            for ( i = -3; i <= 3; i++) { 
                out.meansBlock[0] += r(j, i).normal      / 49.0f; 
                out.meansBlock[1] += r(j, i).alb1        / 49.0f; 
                out.meansBlock[2] += r(j, i).alb2        / 49.0f; 
                out.meansBlock[3] += r(j, i).worldPos    / 49.0f; 
                out.meansBlock[4] += r(j, i).directLight / 49.0f; 
                // 3x3 means for mean deviation
                if (abs(j) <= 1 && abs(i) <= 1) {
                    meansForMD[0] += r(j, i).normal      / 9.0f; 
                    meansForMD[1] += r(j, i).alb1        / 9.0f; 
                    meansForMD[2] += r(j, i).alb2        / 9.0f; 
                    meansForMD[3] += r(j, i).worldPos    / 9.0f; 
                    meansForMD[4] += r(j, i).directLight / 9.0f; 
                }
            }
        }
        // K block std dev (std dev of col to mean col, or avg of std dev of each ? )
        for ( j = -3; j <= 3; j++) { 
            for ( i = -3; i <= 3; i++) { 
                out.sdBlock[0] += pow((r(j,i).normal      - out.meansBlock[0]),2); 
                out.sdBlock[1] += pow((r(j,i).alb1        - out.meansBlock[1]),2); 
                out.sdBlock[2] += pow((r(j,i).alb2        - out.meansBlock[2]),2); 
                out.sdBlock[3] += pow((r(j,i).worldPos    - out.meansBlock[3]),2); 
                out.sdBlock[4] += pow((r(j,i).directLight - out.meansBlock[4]),2); 
            }
        }
        out.sdBlock[0] = sqrt(out.sdBlock[0] / 49.0f);
        out.sdBlock[1] = sqrt(out.sdBlock[1] / 49.0f);
        out.sdBlock[2] = sqrt(out.sdBlock[2] / 49.0f);
        out.sdBlock[3] = sqrt(out.sdBlock[3] / 49.0f);
        out.sdBlock[4] = sqrt(out.sdBlock[4] / 49.0f);
        // K Gradients (3x3 Block)
        // K Mean Deviations (3x3 block)
        for ( j = -1; j <= 1; j++) { 
            for ( i = -1; i <= 1; i++) { 
                linearInd = (j+1)*3 + i + 1;
                // Sobel operator
                GxSum[0] += Gx[linearInd]*r(j,i).normal;
                GySum[0] += Gy[linearInd]*r(j,i).normal;
                GxSum[1] += Gx[linearInd]*r(j,i).alb1;
                GySum[1] += Gy[linearInd]*r(j,i).alb1;
                GxSum[2] += Gx[linearInd]*r(j,i).alb2;
                GySum[2] += Gy[linearInd]*r(j,i).alb2;
                GxSum[3] += Gx[linearInd]*r(j,i).worldPos;
                GySum[3] += Gy[linearInd]*r(j,i).worldPos;
                GxSum[4] += Gx[linearInd]*r(j,i).directLight;
                GySum[4] += Gy[linearInd]*r(j,i).directLight;

                // Mean Abs Diff
                out.meanDeviation[0] += fabs(r(j,i).normal      - meansForMD[0])/ 9.0f; 
                out.meanDeviation[1] += fabs(r(j,i).alb1        - meansForMD[1])/ 9.0f; 
                out.meanDeviation[2] += fabs(r(j,i).alb2        - meansForMD[2])/ 9.0f; 
                out.meanDeviation[3] += fabs(r(j,i).worldPos    - meansForMD[3])/ 9.0f; 
                out.meanDeviation[4] += fabs(r(j,i).directLight - meansForMD[4])/ 9.0f; 

                // Collect MAD Values
                valuesForMAD[0][linearInd] = r(j,i).normal     ;
                valuesForMAD[1][linearInd] = r(j,i).alb1       ;
                valuesForMAD[2][linearInd] = r(j,i).alb2       ;
                valuesForMAD[3][linearInd] = r(j,i).worldPos   ;
                valuesForMAD[4][linearInd] = r(j,i).directLight;
            }
        }
        out.gradients[0] = sqrt(GxSum[0]*GxSum[0] + GySum[0]*GySum[0]);
        out.gradients[1] = sqrt(GxSum[1]*GxSum[1] + GySum[1]*GySum[1]);
        out.gradients[2] = sqrt(GxSum[2]*GxSum[2] + GySum[2]*GySum[2]);
        out.gradients[3] = sqrt(GxSum[3]*GxSum[3] + GySum[3]*GySum[3]);
        out.gradients[4] = sqrt(GxSum[4]*GxSum[4] + GySum[4]*GySum[4]);
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
            out.MAD[feature] = medianGetter[0];
        }
        // L
        out.L = 1.0f/sConstants.samples;

    }
    
    // Variances
    {
        int node, weight, numNodes;

        // Layer 1 - 2
        for (node=0; node<10; node++) {
            out.l2[node] = 0.0f;

            // Go through all secondary features
            {
                out.l2[node] += out.meansSingle[0]   * sConstants.onetwo[36*node + 0]; 
                out.l2[node] += out.meansSingle[1]   * sConstants.onetwo[36*node + 1]; 
                out.l2[node] += out.meansSingle[2]   * sConstants.onetwo[36*node + 2]; 
                out.l2[node] += out.meansSingle[3]   * sConstants.onetwo[36*node + 3]; 
                out.l2[node] += out.meansSingle[4]   * sConstants.onetwo[36*node + 4]; 
                out.l2[node] += out.sdSingle[0]      * sConstants.onetwo[36*node + 5];
                out.l2[node] += out.sdSingle[1]      * sConstants.onetwo[36*node + 6];
                out.l2[node] += out.sdSingle[2]      * sConstants.onetwo[36*node + 7];
                out.l2[node] += out.sdSingle[3]      * sConstants.onetwo[36*node + 8];
                out.l2[node] += out.sdSingle[4]      * sConstants.onetwo[36*node + 9];
                out.l2[node] += out.meansBlock[0]    * sConstants.onetwo[36*node + 10];
                out.l2[node] += out.meansBlock[1]    * sConstants.onetwo[36*node + 11];
                out.l2[node] += out.meansBlock[2]    * sConstants.onetwo[36*node + 12];
                out.l2[node] += out.meansBlock[3]    * sConstants.onetwo[36*node + 13];
                out.l2[node] += out.meansBlock[4]    * sConstants.onetwo[36*node + 14];
                out.l2[node] += out.sdBlock[0]       * sConstants.onetwo[36*node + 15];
                out.l2[node] += out.sdBlock[1]       * sConstants.onetwo[36*node + 16];
                out.l2[node] += out.sdBlock[2]       * sConstants.onetwo[36*node + 17];
                out.l2[node] += out.sdBlock[3]       * sConstants.onetwo[36*node + 18];
                out.l2[node] += out.sdBlock[4]       * sConstants.onetwo[36*node + 19];
                out.l2[node] += out.gradients[0]     * sConstants.onetwo[36*node + 20];
                out.l2[node] += out.gradients[1]     * sConstants.onetwo[36*node + 21];
                out.l2[node] += out.gradients[2]     * sConstants.onetwo[36*node + 22];
                out.l2[node] += out.gradients[3]     * sConstants.onetwo[36*node + 23];
                out.l2[node] += out.gradients[4]     * sConstants.onetwo[36*node + 24];
                out.l2[node] += out.meanDeviation[0] * sConstants.onetwo[36*node + 25];
                out.l2[node] += out.meanDeviation[1] * sConstants.onetwo[36*node + 26];
                out.l2[node] += out.meanDeviation[2] * sConstants.onetwo[36*node + 27];
                out.l2[node] += out.meanDeviation[3] * sConstants.onetwo[36*node + 28];
                out.l2[node] += out.meanDeviation[4] * sConstants.onetwo[36*node + 29];
                out.l2[node] += out.MAD[0]           * sConstants.onetwo[36*node + 30];
                out.l2[node] += out.MAD[1]           * sConstants.onetwo[36*node + 31];
                out.l2[node] += out.MAD[2]           * sConstants.onetwo[36*node + 32];
                out.l2[node] += out.MAD[3]           * sConstants.onetwo[36*node + 33];
                out.l2[node] += out.MAD[4]           * sConstants.onetwo[36*node + 34];
                out.l2[node] += out.L                * sConstants.onetwo[36*node + 35];
            }
            
            out.l2[node] = 1.0f/(1.0f + exp(-out.l2[node]));
        }

        // Layer 2 - 3
        for (node=0; node<10; node++) {
            out.l3[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                out.l3[node] += out.l2[weight]*sConstants.twothree[10*node + weight];
            out.l3[node] = 1.0f/(1.0f + exp(-out.l3[node]));
        }

        // Layer 3 - 4
        for (node=0; node<7; node++) {
            out.variances[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                out.variances[node] += out.l3[weight]*sConstants.threefour[10*node + weight];
            out.variances[node] = log(1.0f + exp(out.variances[node]));
        }

    }

    return out;    
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE ForPropOut CPU(skepu::Region2D<ForPropIn> r, SkePUFPConstants sConstants)
{


    // Secondary Features
    ForPropOut out;
    {
        // - K Means of single pixel, K std deviations of single pixel, K Means of 7x7 block, K std deviations of 7x7 block (20 total)
        // - Magnitude of gradients of K features of single pixel (sobel operator) (5 total)
        // - Sum of the abs difference between K of each pixel in 3x3 block and the mean of that 3x3 block (5 total)
        // - MAD of K features, so median of values minus median value, in NxN block (5 total)
        // - 1/totalSamples (1 total)

        int ind, c = 0;
        int ijInd = 0;
        int jFixed, iFixed = 0;
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
        float medianGetter[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        int i, j;

        float meansForMD[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float GxSum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float GySum[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; 

        for (c=0;c<10;c++) {
            out.l2[c] = 0.0f;
            out.l3[c] = 0.0f;
        }     
        for (c=0;c<7;c++) 
            out.variances[c] = 0.0f;
        for (c=0; c<5; c++) {
            out.meansSingle[c] = 0.0f;
            out.sdSingle[c] = 0.0f;
            out.meansBlock[c] = 0.0f;
            out.sdBlock[c] = 0.0f;
            out.gradients[c] = 0.0f;
            out.meanDeviation[c] = 0.0f;
            out.MAD[c] = 0.0f; 
        }
        out.L = 0.0f;

        // K single Means
        out.meansSingle[0] = r(0,0).normal;
        out.meansSingle[1] = r(0,0).alb1;
        out.meansSingle[2] = r(0,0).alb2;
        out.meansSingle[3] = r(0,0).worldPos;
        out.meansSingle[4] = r(0,0).directLight;

        // K single std devs
		out.sdSingle[0] = r(0,0).stdDev[1];
		out.sdSingle[1] = r(0,0).stdDev[2];
		out.sdSingle[2] = r(0,0).stdDev[3];
		out.sdSingle[3] = r(0,0).stdDev[4];
        out.sdSingle[4] = r(0,0).stdDev[5];

        // K block means
        for ( j = -3; j <= 3; j++) { 
            for ( i = -3; i <= 3; i++) { 
                out.meansBlock[0] += r(j, i).normal      / 49.0f; 
                out.meansBlock[1] += r(j, i).alb1        / 49.0f; 
                out.meansBlock[2] += r(j, i).alb2        / 49.0f; 
                out.meansBlock[3] += r(j, i).worldPos    / 49.0f; 
                out.meansBlock[4] += r(j, i).directLight / 49.0f; 
                // 3x3 means for mean deviation
                if (abs(j) <= 1 && abs(i) <= 1) {
                    meansForMD[0] += r(j, i).normal      / 9.0f; 
                    meansForMD[1] += r(j, i).alb1        / 9.0f; 
                    meansForMD[2] += r(j, i).alb2        / 9.0f; 
                    meansForMD[3] += r(j, i).worldPos    / 9.0f; 
                    meansForMD[4] += r(j, i).directLight / 9.0f; 
                }
            }
        }
        // K block std dev (std dev of col to mean col, or avg of std dev of each ? )
        for ( j = -3; j <= 3; j++) { 
            for ( i = -3; i <= 3; i++) { 
                out.sdBlock[0] += pow((r(j,i).normal      - out.meansBlock[0]),2); 
                out.sdBlock[1] += pow((r(j,i).alb1        - out.meansBlock[1]),2); 
                out.sdBlock[2] += pow((r(j,i).alb2        - out.meansBlock[2]),2); 
                out.sdBlock[3] += pow((r(j,i).worldPos    - out.meansBlock[3]),2); 
                out.sdBlock[4] += pow((r(j,i).directLight - out.meansBlock[4]),2); 
            }
        }
        out.sdBlock[0] = sqrt(out.sdBlock[0] / 49.0f);
        out.sdBlock[1] = sqrt(out.sdBlock[1] / 49.0f);
        out.sdBlock[2] = sqrt(out.sdBlock[2] / 49.0f);
        out.sdBlock[3] = sqrt(out.sdBlock[3] / 49.0f);
        out.sdBlock[4] = sqrt(out.sdBlock[4] / 49.0f);
        // K Gradients (3x3 Block)
        // K Mean Deviations (3x3 block)
        for ( j = -1; j <= 1; j++) { 
            for ( i = -1; i <= 1; i++) { 
                linearInd = (j+1)*3 + i + 1;
                // Sobel operator
                GxSum[0] += Gx[linearInd]*r(j,i).normal;
                GySum[0] += Gy[linearInd]*r(j,i).normal;
                GxSum[1] += Gx[linearInd]*r(j,i).alb1;
                GySum[1] += Gy[linearInd]*r(j,i).alb1;
                GxSum[2] += Gx[linearInd]*r(j,i).alb2;
                GySum[2] += Gy[linearInd]*r(j,i).alb2;
                GxSum[3] += Gx[linearInd]*r(j,i).worldPos;
                GySum[3] += Gy[linearInd]*r(j,i).worldPos;
                GxSum[4] += Gx[linearInd]*r(j,i).directLight;
                GySum[4] += Gy[linearInd]*r(j,i).directLight;

                // Mean Abs Diff
                out.meanDeviation[0] += fabs(r(j,i).normal      - meansForMD[0])/ 9.0f; 
                out.meanDeviation[1] += fabs(r(j,i).alb1        - meansForMD[1])/ 9.0f; 
                out.meanDeviation[2] += fabs(r(j,i).alb2        - meansForMD[2])/ 9.0f; 
                out.meanDeviation[3] += fabs(r(j,i).worldPos    - meansForMD[3])/ 9.0f; 
                out.meanDeviation[4] += fabs(r(j,i).directLight - meansForMD[4])/ 9.0f; 

                // Collect MAD Values
                valuesForMAD[0][linearInd] = r(j,i).normal     ;
                valuesForMAD[1][linearInd] = r(j,i).alb1       ;
                valuesForMAD[2][linearInd] = r(j,i).alb2       ;
                valuesForMAD[3][linearInd] = r(j,i).worldPos   ;
                valuesForMAD[4][linearInd] = r(j,i).directLight;
            }
        }
        out.gradients[0] = sqrt(GxSum[0]*GxSum[0] + GySum[0]*GySum[0]);
        out.gradients[1] = sqrt(GxSum[1]*GxSum[1] + GySum[1]*GySum[1]);
        out.gradients[2] = sqrt(GxSum[2]*GxSum[2] + GySum[2]*GySum[2]);
        out.gradients[3] = sqrt(GxSum[3]*GxSum[3] + GySum[3]*GySum[3]);
        out.gradients[4] = sqrt(GxSum[4]*GxSum[4] + GySum[4]*GySum[4]);
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
            out.MAD[feature] = medianGetter[0];
        }
        // L
        out.L = 1.0f/sConstants.samples;

    }
    
    // Variances
    {
        int node, weight, numNodes;

        // Layer 1 - 2
        for (node=0; node<10; node++) {
            out.l2[node] = 0.0f;

            // Go through all secondary features
            {
                out.l2[node] += out.meansSingle[0]   * sConstants.onetwo[36*node + 0]; 
                out.l2[node] += out.meansSingle[1]   * sConstants.onetwo[36*node + 1]; 
                out.l2[node] += out.meansSingle[2]   * sConstants.onetwo[36*node + 2]; 
                out.l2[node] += out.meansSingle[3]   * sConstants.onetwo[36*node + 3]; 
                out.l2[node] += out.meansSingle[4]   * sConstants.onetwo[36*node + 4]; 
                out.l2[node] += out.sdSingle[0]      * sConstants.onetwo[36*node + 5];
                out.l2[node] += out.sdSingle[1]      * sConstants.onetwo[36*node + 6];
                out.l2[node] += out.sdSingle[2]      * sConstants.onetwo[36*node + 7];
                out.l2[node] += out.sdSingle[3]      * sConstants.onetwo[36*node + 8];
                out.l2[node] += out.sdSingle[4]      * sConstants.onetwo[36*node + 9];
                out.l2[node] += out.meansBlock[0]    * sConstants.onetwo[36*node + 10];
                out.l2[node] += out.meansBlock[1]    * sConstants.onetwo[36*node + 11];
                out.l2[node] += out.meansBlock[2]    * sConstants.onetwo[36*node + 12];
                out.l2[node] += out.meansBlock[3]    * sConstants.onetwo[36*node + 13];
                out.l2[node] += out.meansBlock[4]    * sConstants.onetwo[36*node + 14];
                out.l2[node] += out.sdBlock[0]       * sConstants.onetwo[36*node + 15];
                out.l2[node] += out.sdBlock[1]       * sConstants.onetwo[36*node + 16];
                out.l2[node] += out.sdBlock[2]       * sConstants.onetwo[36*node + 17];
                out.l2[node] += out.sdBlock[3]       * sConstants.onetwo[36*node + 18];
                out.l2[node] += out.sdBlock[4]       * sConstants.onetwo[36*node + 19];
                out.l2[node] += out.gradients[0]     * sConstants.onetwo[36*node + 20];
                out.l2[node] += out.gradients[1]     * sConstants.onetwo[36*node + 21];
                out.l2[node] += out.gradients[2]     * sConstants.onetwo[36*node + 22];
                out.l2[node] += out.gradients[3]     * sConstants.onetwo[36*node + 23];
                out.l2[node] += out.gradients[4]     * sConstants.onetwo[36*node + 24];
                out.l2[node] += out.meanDeviation[0] * sConstants.onetwo[36*node + 25];
                out.l2[node] += out.meanDeviation[1] * sConstants.onetwo[36*node + 26];
                out.l2[node] += out.meanDeviation[2] * sConstants.onetwo[36*node + 27];
                out.l2[node] += out.meanDeviation[3] * sConstants.onetwo[36*node + 28];
                out.l2[node] += out.meanDeviation[4] * sConstants.onetwo[36*node + 29];
                out.l2[node] += out.MAD[0]           * sConstants.onetwo[36*node + 30];
                out.l2[node] += out.MAD[1]           * sConstants.onetwo[36*node + 31];
                out.l2[node] += out.MAD[2]           * sConstants.onetwo[36*node + 32];
                out.l2[node] += out.MAD[3]           * sConstants.onetwo[36*node + 33];
                out.l2[node] += out.MAD[4]           * sConstants.onetwo[36*node + 34];
                out.l2[node] += out.L                * sConstants.onetwo[36*node + 35];
            }
            
            out.l2[node] = 1.0f/(1.0f + exp(-out.l2[node]));
        }

        // Layer 2 - 3
        for (node=0; node<10; node++) {
            out.l3[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                out.l3[node] += out.l2[weight]*sConstants.twothree[10*node + weight];
            out.l3[node] = 1.0f/(1.0f + exp(-out.l3[node]));
        }

        // Layer 3 - 4
        for (node=0; node<7; node++) {
            out.variances[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                out.variances[node] += out.l3[weight]*sConstants.threefour[10*node + weight];
            out.variances[node] = log(1.0f + exp(out.variances[node]));
        }

    }

    return out;    
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "SkePUDenoiserNN_Overlap2DKernel_SkePUFPFunc.cu"
void DenoiserNN::SkePUForwardProp() {

    auto in = skepu::Matrix<ForPropIn>(yRes, xRes);
    auto out = skepu::Matrix<ForPropOut>(yRes, xRes);
    SkePUFPConstants sConstants;

    delete sFeatures;
    sFeatures = new SecondaryFeatures[yRes*xRes];

    // Set sConstants
    int w;
    sConstants.samples = sampleCount;
    for (w=0;w<360;w++)
        sConstants.onetwo[w] = onetwo[w];
    for (w=0;w<100;w++)
        sConstants.twothree[w] = twothree[w];
    for (w=0;w<70;w++)
        sConstants.threefour[w] = threefour[w];

    // Set Inputs
    for (int ind = 0; ind < xRes*yRes; ind++) {

        in[ind].normal      = (normal[ind].x     + normal[ind].y   + normal[ind].z)   / (3.0f * sampleCount);
        in[ind].alb1        = (albedo1[ind].x    + albedo1[ind].y  + albedo1[ind].z)  / (3.0f * sampleCount);
        in[ind].alb2        = (albedo2[ind].x    + albedo2[ind].y  + albedo2[ind].z)  / (3.0f * sampleCount);
        in[ind].worldPos    = (worldPos[ind].x   + worldPos[ind].y + worldPos[ind].z) / (3.0f * sampleCount);
        in[ind].directLight = directLight[ind].x / sampleCount;

        in[ind].stdDev[0] = denoisingInf[ind].stdDev[0];
        in[ind].stdDev[1] = denoisingInf[ind].stdDev[1];
        in[ind].stdDev[2] = denoisingInf[ind].stdDev[2];
        in[ind].stdDev[3] = denoisingInf[ind].stdDev[3];
        in[ind].stdDev[4] = denoisingInf[ind].stdDev[4];
        in[ind].stdDev[5] = denoisingInf[ind].stdDev[5];
    }

    auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(denoisingSkePUBackend)};
	spec.activateBackend();
    skepu::backend::MapOverlap2D<skepu_userfunction_skepu_skel_2convol_SkePUFPFunc, decltype(&SkePUDenoiserNN_Overlap2DKernel_SkePUFPFunc_conv_cuda_2D_kernel), void> convol(SkePUDenoiserNN_Overlap2DKernel_SkePUFPFunc_conv_cuda_2D_kernel);
	convol.setBackend(spec);
    convol.setOverlap(3);
    convol.setEdgeMode(skepu::Edge::Duplicate);

    convol(out, in, sConstants);
    out.updateHost();

    int ind, v;
    ForPropOut ret;
    DenoisingInf* info;
    SecondaryFeatures* s;
    for (int j = 0; j < yRes; j++) {
        for (int i = 0; i < xRes; i++) {
            
            ind = j*xRes + i;

            ret = out(j, i);
            info = &denoisingInf[ind];
            s = &sFeatures[ind];

            for (v = 0; v < 10; v++) {
                layerTwoValues[10*ind + v] = ret.l2[v];
                layerThreeValues[10*ind + v] = ret.l3[v];
            }
            for (v = 0; v < 7;  v++)
                info->variances[v] = ret.variances[v];
            for (v = 0; v < 5; v++) {
                s->meansSingle[v]   = ret.meansSingle[v];
                s->sdSingle[v]      = ret.sdSingle[v];
                s->meansBlock[v]    = ret.meansBlock[v];
                s->sdBlock[v]       = ret.sdBlock[v];
                s->gradients[v]     = ret.gradients[v];
                s->meanDeviation[v] = ret.meanDeviation[v];
                s->MAD[v]           = ret.MAD[v];
            }
            s->L = ret.L;

        }
    }
    
}

// Back Prop Functions
vec3 DenoiserNN::CPUFilterDerivative(int j, int i, int var) {

    // Contribution to the filtered colour of the filter paramater

    int tW = yRes;
    int tH = xRes;
    int N = denoisingN;
    int iIndex = j*tW + i;
    int jIndex;
    int dVal;
    DenoisingInf info, infoj;
    vec3 vecSum, fDeriv;
    info = denoisingInf[iIndex];
    float weightOverParam = 0.0f;

    float dVals[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dValMult = 1.0f;


    vec3  jCol     , iCol      ;
    vec3  jNormal  , iNormal   ;
    vec3  jAlbedo1 , iAlbedo1  ;
    vec3  jAlbedo2 , iAlbedo2  ;
    vec3  jWorldPos, iWorldPos ;
    float jDirectLight, iDirectLight;

    float paramDiffs[7];

    // For all pixels j around i
    for (int j1 = -N; j1 <= N; j1++) {
        for (int i1 = -N; i1 <= N; i1++) {

            // if (j1==j && i1==i)
            //     continue;

            jIndex = j+j1<0 ? 0 : (j+j1>=tH ? tH-1 : j+j1);
            jIndex *= tW;
            jIndex += i+i1<0 ? 0 : (i+i1>=tW ? tW-1 : i+i1);

            infoj = denoisingInf[jIndex];

            vecSum = (preScreen[jIndex] - denoisedCol[iIndex]) / info.wcSum;

            iCol =         preScreen[iIndex]     / sampleCount;
            iNormal =      normal[iIndex]        / sampleCount;
            iAlbedo1 =     albedo1[iIndex]       / sampleCount;
            iAlbedo2 =     albedo2[iIndex]       / sampleCount;
            iWorldPos =    worldPos[iIndex]      / sampleCount;
            iDirectLight = directLight[iIndex].x / sampleCount;


            jCol =         preScreen[jIndex]     / sampleCount;
            jNormal =      normal[jIndex]        / sampleCount;
            jAlbedo1 =     albedo1[jIndex]       / sampleCount;
            jAlbedo2 =     albedo2[jIndex]       / sampleCount;
            jWorldPos =    worldPos[jIndex]      / sampleCount;
            jDirectLight = directLight[jIndex].x / sampleCount;

            // Index d value
            paramDiffs[0] = pow(j,2)+pow(i,2);
            dVals[0] = paramDiffs[0] / (2.0f * info.variances[0] + 0.000001f);
            // Colour d value
            paramDiffs[1] = pow(iCol[0] - jCol[0],2)+pow(iCol[1] - jCol[1],2) + pow(iCol[2] - jCol[2],2);
            dVals[1] = paramDiffs[1] / (2.0f * info.variances[1] * (info.stdDev[0] + infoj.stdDev[0])  + 0.000001f);
            // Normal d value
            paramDiffs[2] = pow(iNormal[0] - jNormal[0],2)+pow(iNormal[1] - jNormal[1],2) + pow(iNormal[2] - jNormal[2],2);
            dVals[2] = paramDiffs[2] / (2.0f * info.variances[2] * info.stdDev[1]  + 0.000001f);
            // Alb1 d value
            paramDiffs[3] = pow(iAlbedo1[0] - jAlbedo1[0],2)+pow(iAlbedo1[1] - jAlbedo1[1],2) + pow(iAlbedo1[2] - jAlbedo1[2],2);
            dVals[3] = paramDiffs[3] / (2.0f * info.variances[3] * info.stdDev[2]  + 0.000001f);
            // Alb2 d value
            paramDiffs[4] = pow(iAlbedo2[0] - jAlbedo2[0],2)+pow(iAlbedo2[1] - jAlbedo2[1],2) + pow(iAlbedo2[2] - jAlbedo2[2],2);
            dVals[4] = paramDiffs[4] / (2.0f * info.variances[4] * info.stdDev[3]  + 0.000001f);
            // worldPos d value
            paramDiffs[5] = pow(iWorldPos[0] - jWorldPos[0],2)+pow(iWorldPos[1] - jWorldPos[1],2) + pow(iWorldPos[2] - jWorldPos[2],2);
            dVals[5] = paramDiffs[5] / (2.0f * info.variances[5] * info.stdDev[4]  + 0.000001f);
            // directLight d value
            paramDiffs[6] = pow(iDirectLight - jDirectLight,2);
            dVals[6] = paramDiffs[6] / (2.0f * info.variances[6] * info.stdDev[5] + 0.000001f);


            dValMult = 1.0f;
            for (int dVal=0;dVal<7;dVal++) {
                dValMult *= exp(-dVals[dVal]) + 0.000001f;;
            }

            weightOverParam = dValMult * paramDiffs[var] / pow(info.variances[var],3);

            fDeriv += vecSum*weightOverParam;

        }      
    } 

    return fDeriv;   

}
void DenoiserNN::CPUBackProp() {
    int pixels = xRes*yRes;
    int pixel, var, w;
    float paramOverWeight;
    vec3 errorOverColour, colourOverParam;
    vec3 tCol;

    // All pixels
    for (int j=0;j<yRes;j++) {
        for (int i=0; i<xRes; i++) {

            pixel = j*xRes + i;
            tCol = targetCol[pixel];

            // Derivative One: samples * (cFiltered - cTarget)/(cTarget*cTarget)
            errorOverColour = vec3();
            errorOverColour.x = sampleCount * (denoisedCol[pixel].x - tCol.x) / (tCol.x*tCol.x+0.0001f);
            errorOverColour.y = sampleCount * (denoisedCol[pixel].y - tCol.y) / (tCol.y*tCol.y+0.0001f);
            errorOverColour.z = sampleCount * (denoisedCol[pixel].z - tCol.z) / (tCol.z*tCol.z+0.0001f);

            // Filter Paramaters (index, col, K)
            for (var=0;var<7;var++) {

                // Derivative Two: cross-bilateral filter derivative
                colourOverParam = CPUFilterDerivative(j, i, var);

                // Weights One
                for (w=0;w<360;w++){
                    // Derivative Three: filter/weight = secondary feature input at weight index
                    paramOverWeight = sFeatures[pixel](w % 36);
                    onetwo[w] += learningRate*errorOverColour.dot(colourOverParam)*paramOverWeight;
                }
                // Weights Two
                for (w=0;w<100;w++){
                    // Derivative Three: filter/weight = second layer value at weight index
                    paramOverWeight = layerTwoValues[10*pixel + w % 10];
                    twothree[w] += learningRate*errorOverColour.dot(colourOverParam)*paramOverWeight;
                }
                // Weights Three
                for (w=0;w<70;w++){
                    // Derivative Three: filter/weight = third layer value at weight index
                    paramOverWeight = layerThreeValues[10*pixel + w % 10];
                    threefour[w] += learningRate*errorOverColour.dot(colourOverParam)*paramOverWeight;
                }
            }
        }
    }
}
vec3 DenoiserNN::OMPFilterDerivative(int j, int i, int var) {

    // Contribution to the filtered colour of the filter paramater

    int N = denoisingN;
    int iIndex = j*xRes + i;
    DenoisingInf info = denoisingInf[iIndex];
    vec3 fDeriv;

    vec3 iCol =         preScreen[iIndex]     / sampleCount;
    vec3 iNormal =      normal[iIndex]        / sampleCount;
    vec3 iAlbedo1 =     albedo1[iIndex]       / sampleCount;
    vec3 iAlbedo2 =     albedo2[iIndex]       / sampleCount;
    vec3 iWorldPos =    worldPos[iIndex]      / sampleCount;
    float iDirectLight = directLight[iIndex].x / sampleCount;

    float paramDiffs[7];

    // For all pixels j around i
    #pragma omp parallel for
    for (int j1 = -N; j1 <= N; j1++) {
        #pragma omp parallel for
        for (int i1 = -N; i1 <= N; i1++) {

            // if (j1==j && i1==i)
            //     continue;

            float dVals[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            int jIndex = j+j1<0 ? 0 : (j+j1>=yRes ? yRes-1 : j+j1);
            jIndex *= xRes;
            jIndex += i+i1<0 ? 0 : (i+i1>=xRes ? xRes-1 : i+i1);

            DenoisingInf infoj = denoisingInf[jIndex];

            vec3 vecSum = (preScreen[jIndex] - denoisedCol[iIndex]) / info.wcSum;


            vec3 jCol =         preScreen[jIndex]     / sampleCount;
            vec3 jNormal =      normal[jIndex]        / sampleCount;
            vec3 jAlbedo1 =     albedo1[jIndex]       / sampleCount;
            vec3 jAlbedo2 =     albedo2[jIndex]       / sampleCount;
            vec3 jWorldPos =    worldPos[jIndex]      / sampleCount;
            float jDirectLight = directLight[jIndex].x / sampleCount;

            // Index d value
            paramDiffs[0] = pow(j,2)+pow(i,2);
            dVals[0] = paramDiffs[0] / (2.0f * info.variances[0] + 0.000001f);
            // Colour d value
            paramDiffs[1] = pow(iCol[0] - jCol[0],2)+pow(iCol[1] - jCol[1],2) + pow(iCol[2] - jCol[2],2);
            dVals[1] = paramDiffs[1] / (2.0f * info.variances[1] * (info.stdDev[0] + infoj.stdDev[0])  + 0.000001f);
            // Normal d value
            paramDiffs[2] = pow(iNormal[0] - jNormal[0],2)+pow(iNormal[1] - jNormal[1],2) + pow(iNormal[2] - jNormal[2],2);
            dVals[2] = paramDiffs[2] / (2.0f * info.variances[2] * info.stdDev[1]  + 0.000001f);
            // Alb1 d value
            paramDiffs[3] = pow(iAlbedo1[0] - jAlbedo1[0],2)+pow(iAlbedo1[1] - jAlbedo1[1],2) + pow(iAlbedo1[2] - jAlbedo1[2],2);
            dVals[3] = paramDiffs[3] / (2.0f * info.variances[3] * info.stdDev[2]  + 0.000001f);
            // Alb2 d value
            paramDiffs[4] = pow(iAlbedo2[0] - jAlbedo2[0],2)+pow(iAlbedo2[1] - jAlbedo2[1],2) + pow(iAlbedo2[2] - jAlbedo2[2],2);
            dVals[4] = paramDiffs[4] / (2.0f * info.variances[4] * info.stdDev[3]  + 0.000001f);
            // worldPos d value
            paramDiffs[5] = pow(iWorldPos[0] - jWorldPos[0],2)+pow(iWorldPos[1] - jWorldPos[1],2) + pow(iWorldPos[2] - jWorldPos[2],2);
            dVals[5] = paramDiffs[5] / (2.0f * info.variances[5] * info.stdDev[4]  + 0.000001f);
            // directLight d value
            paramDiffs[6] = pow(iDirectLight - jDirectLight,2);
            dVals[6] = paramDiffs[6] / (2.0f * info.variances[6] * info.stdDev[5] + 0.000001f);


            float dValMult = 1.0f;
            for (int dVal=0;dVal<7;dVal++) {
                dValMult *= exp(-dVals[dVal]) + 0.000001f;;
            }

            float weightOverParam = dValMult * paramDiffs[var] / pow(info.variances[var],3);

            fDeriv += vecSum*weightOverParam;

        }      
    } 

    return fDeriv;   

}
void DenoiserNN::OMPBackProp() {

    int pixels = xRes*yRes;
    
    #pragma omp parallel for
    for (int j=0;j<yRes;j++) {
        #pragma omp parallel for
        for (int i=0; i<xRes; i++) {

            vec3 errorOverColour, colourOverParam;
            float paramOverWeight;

            int pixel = j*xRes + i;
            vec3 tCol = targetCol[pixel];

            // Derivative One: samples * (cFiltered - cTarget)/(cTarget*cTarget)
            errorOverColour = vec3();
            errorOverColour.x = sampleCount * (denoisedCol[pixel].x - tCol.x) / (tCol.x*tCol.x+0.0001f);
            errorOverColour.y = sampleCount * (denoisedCol[pixel].y - tCol.y) / (tCol.y*tCol.y+0.0001f);
            errorOverColour.z = sampleCount * (denoisedCol[pixel].z - tCol.z) / (tCol.z*tCol.z+0.0001f);

            // Filter Paramaters (index, col, K)
            int w;
            for (int var=0;var<7;var++) {

                // Derivative Two: cross-bilateral filter derivative
                colourOverParam = OMPFilterDerivative(j, i, var);

                // Weights One
                for (w=0;w<360;w++){
                    // Derivative Three: filter/weight = secondary feature input at weight index
                    paramOverWeight = sFeatures[pixel](w % 36);
                    onetwo[w] += learningRate*errorOverColour.dot(colourOverParam)*paramOverWeight;
                }
                // Weights Two
                for (w=0;w<100;w++){
                    // Derivative Three: filter/weight = second layer value at weight index
                    paramOverWeight = layerTwoValues[10*pixel + w % 10];
                    twothree[w] += learningRate*errorOverColour.dot(colourOverParam)*paramOverWeight;
                }
                // Weights Three
                for (w=0;w<70;w++){
                    // Derivative Three: filter/weight = third layer value at weight index
                    paramOverWeight = layerThreeValues[10*pixel + w % 10];
                    threefour[w] += learningRate*errorOverColour.dot(colourOverParam)*paramOverWeight;
                }
            }
        }
    }
}
static FilterDerivOut SkePUFDFunc(skepu::Region2D<FilterDerivIn> r, int samples) {
    // Contribution to the filtered colour of the filter paramater

    int dVal, var, c;
    float vecSum[3];
    float weightOverParam = 0.0f;

    FilterDerivOut fDeriv;

    float dVals[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dValMult = 1.0f;

    float iCol[3]   ;
    float iNormal[3]  ;
    float iAlbedo1[3] ;
    float iAlbedo2[3] ;
    float iWorldPos[3];
    float iDirectLight;
    for (c =0; c<3;c++) {
        iCol[c]      = r(0,0).preScreen[c] / samples;
        iNormal[c]   = r(0,0).normal[c]    / samples;
        iAlbedo1[c]  = r(0,0).alb1[c]      / samples;
        iAlbedo2[c]  = r(0,0).alb2[c]      / samples;
        iWorldPos[c] = r(0,0).worldPos[c]  / samples;
    }
    iDirectLight = r(0,0).directLight / samples;

    float  jCol[3]      ;
    float  jNormal[3]   ;
    float  jAlbedo1[3]  ;
    float  jAlbedo2[3]  ;
    float  jWorldPos[3] ;
    float jDirectLight;

    for (var=0; var<7; var++) {
        fDeriv.paramXYZ[var][0] = 0.0f;
        fDeriv.paramXYZ[var][1] = 0.0f;
        fDeriv.paramXYZ[var][2] = 0.0f;
    }

    float paramDiffs[7];

    // For all pixels j around i
    for (int j = -r.oj; j <= r.oj; j++) {
        for (int i = -r.oi; i <= r.oi; i++) {

            vecSum[0] = (r(j, i).preScreen[0] - r(j, i).denoisedCol[0]) / r(j, i).wcSum;
            vecSum[1] = (r(j, i).preScreen[1] - r(j, i).denoisedCol[1]) / r(j, i).wcSum;
            vecSum[2] = (r(j, i).preScreen[2] - r(j, i).denoisedCol[2]) / r(j, i).wcSum;

            for (c =0; c<3;c++) {
                jCol[c]      = r(j,i).preScreen[c] / samples;
                jNormal[c]   = r(j,i).normal[c]    / samples;
                jAlbedo1[c]  = r(j,i).alb1[c]      / samples;
                jAlbedo2[c]  = r(j,i).alb2[c]      / samples;
                jWorldPos[c] = r(j,i).worldPos[c]  / samples;
            }
            jDirectLight = r(j,i).directLight / samples;


            // Index d value
            paramDiffs[0] = pow(j,2)+pow(i,2);
            dVals[0] = paramDiffs[0] / (2.0f * r(0,0).variances[0] + 0.000001f);
            // Colour d value
            paramDiffs[0] = pow(iCol[0] - jCol[0],2)+pow(iCol[1] - jCol[1],2) + pow(iCol[2] - jCol[2],2);
            dVals[1] = paramDiffs[1] / (2.0f * r(0,0).variances[1] * (r(0,0).stdDev[0] + r(j,i).stdDev[0])  + 0.000001f);
            // Normal d value
            paramDiffs[2] = pow(iNormal[0] - jNormal[0],2)+pow(iNormal[1] - jNormal[1],2) + pow(iNormal[2] - jNormal[2],2);
            dVals[2] = paramDiffs[2] / (2.0f * r(0,0).variances[2] * r(0,0).stdDev[1]  + 0.000001f);
            // Alb1 d value
            paramDiffs[3] = pow(iAlbedo1[0] - jAlbedo1[0],2)+pow(iAlbedo1[1] - jAlbedo1[1],2) + pow(iAlbedo1[2] - jAlbedo1[2],2);
            dVals[3] = paramDiffs[3] / (2.0f * r(0,0).variances[3] * r(0,0).stdDev[2]  + 0.000001f);
            // Alb2 d value
            paramDiffs[4] = pow(iAlbedo2[0] - jAlbedo2[0],2)+pow(iAlbedo2[1] - jAlbedo2[1],2) + pow(iAlbedo2[2] - jAlbedo2[2],2);
            dVals[4] = paramDiffs[4] / (2.0f * r(0,0).variances[4] * r(0,0).stdDev[3]  + 0.000001f);
            // worldPos d value
            paramDiffs[5] = pow(iWorldPos[0] - jWorldPos[0],2)+pow(iWorldPos[1] - jWorldPos[1],2) + pow(iWorldPos[2] - jWorldPos[2],2);
            dVals[5] = paramDiffs[5] / (2.0f * r(0,0).variances[5] * r(0,0).stdDev[4]  + 0.000001f);
            // directLight d value
            paramDiffs[6] = pow(iDirectLight - jDirectLight,2);
            dVals[6] = paramDiffs[6] / (2.0f * r(0,0).variances[6] * r(0,0).stdDev[5] + 0.000001f);

            dValMult = 1.0f;
            for (dVal=0;dVal<7;dVal++) {
                dValMult *= exp(-dVals[dVal]) + 0.000001f;;
            }

            for (var=0; var<7; var++) {
                weightOverParam = dValMult * paramDiffs[var] / pow(r(0,0).variances[var],3);
                fDeriv.paramXYZ[var][0] += vecSum[0]*weightOverParam;
                fDeriv.paramXYZ[var][1] += vecSum[1]*weightOverParam;
                fDeriv.paramXYZ[var][2] += vecSum[2]*weightOverParam;
            }

        }      
    } 

    return fDeriv;
}
static SkePUBPOut SkePUBPFunc(SkePUBPIn f, int samples, float learningRate) {

            float paramOverWeight, dot;
            float errorOverColour[3];
            float colourOverParam[3];

            SkePUBPOut out;

            // Derivative One: samples * (cFiltered - cTarget)/(cTarget*cTarget)
            errorOverColour[0] = samples * (f.denoisedCol[0] - f.targetCol[0]) / (f.targetCol[0]*f.targetCol[0]+0.0001f);
            errorOverColour[1] = samples * (f.denoisedCol[1] - f.targetCol[1]) / (f.targetCol[1]*f.targetCol[1]+0.0001f);
            errorOverColour[2] = samples * (f.denoisedCol[2] - f.targetCol[2]) / (f.targetCol[2]*f.targetCol[2]+0.0001f);

            // Filter Paramaters (index, col, K)
            int w;
            for (int var=0;var<7;var++) {

                // Derivative Two: cross-bilateral filter derivative
                colourOverParam[0] = f.deriv.paramXYZ[var][0];
                colourOverParam[1] = f.deriv.paramXYZ[var][1];
                colourOverParam[2] = f.deriv.paramXYZ[var][2];

                // Dot Product
                dot = errorOverColour[0]*colourOverParam[0] + 
                      errorOverColour[1]*colourOverParam[1] + 
                      errorOverColour[2]*colourOverParam[2];

                // Weights One
                for (w=0;w<360;w++){
                    if (var==0)
                        out.onetwo[w] = 0.0f;
                    // Derivative Three: filter/weight = secondary feature input at weight index
                    paramOverWeight = f.s[w % 36];
                    out.onetwo[w] += learningRate*dot*paramOverWeight;
                }
                // Weights Two
                for (w=0;w<100;w++){
                    if (var==0)
                        out.twothree[w] = 0.0f;
                    // Derivative Three: filter/weight = second layer value at weight index
                    paramOverWeight = f.l2[w % 10];
                    out.twothree[w] += learningRate*dot*paramOverWeight;
                }
                // Weights Three
                for (w=0;w<70;w++){
                    if (var==0)
                        out.threefour[w] = 0.0f;
                    // Derivative Three: filter/weight = third layer value at weight index
                    paramOverWeight = f.l3[w % 10];
                    out.threefour[w] += learningRate*dot*paramOverWeight;
                }
            }

        return out;
}

struct skepu_userfunction_skepu_skel_0map_SkePUBPFunc
{
constexpr static size_t totalArity = 3;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
constexpr static bool usesPRNG = 0;
constexpr static size_t randomCount = SKEPU_NO_RANDOM;
using IndexType = void;
using ElwiseArgs = std::tuple<SkePUBPIn>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<int, float>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = SkePUBPOut;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ SkePUBPOut CU(SkePUBPIn f, int samples, float learningRate)
{

            float paramOverWeight, dot;
            float errorOverColour[3];
            float colourOverParam[3];

            SkePUBPOut out;

            // Derivative One: samples * (cFiltered - cTarget)/(cTarget*cTarget)
            errorOverColour[0] = samples * (f.denoisedCol[0] - f.targetCol[0]) / (f.targetCol[0]*f.targetCol[0]+0.0001f);
            errorOverColour[1] = samples * (f.denoisedCol[1] - f.targetCol[1]) / (f.targetCol[1]*f.targetCol[1]+0.0001f);
            errorOverColour[2] = samples * (f.denoisedCol[2] - f.targetCol[2]) / (f.targetCol[2]*f.targetCol[2]+0.0001f);

            // Filter Paramaters (index, col, K)
            int w;
            for (int var=0;var<7;var++) {

                // Derivative Two: cross-bilateral filter derivative
                colourOverParam[0] = f.deriv.paramXYZ[var][0];
                colourOverParam[1] = f.deriv.paramXYZ[var][1];
                colourOverParam[2] = f.deriv.paramXYZ[var][2];

                // Dot Product
                dot = errorOverColour[0]*colourOverParam[0] + 
                      errorOverColour[1]*colourOverParam[1] + 
                      errorOverColour[2]*colourOverParam[2];

                // Weights One
                for (w=0;w<360;w++){
                    if (var==0)
                        out.onetwo[w] = 0.0f;
                    // Derivative Three: filter/weight = secondary feature input at weight index
                    paramOverWeight = f.s[w % 36];
                    out.onetwo[w] += learningRate*dot*paramOverWeight;
                }
                // Weights Two
                for (w=0;w<100;w++){
                    if (var==0)
                        out.twothree[w] = 0.0f;
                    // Derivative Three: filter/weight = second layer value at weight index
                    paramOverWeight = f.l2[w % 10];
                    out.twothree[w] += learningRate*dot*paramOverWeight;
                }
                // Weights Three
                for (w=0;w<70;w++){
                    if (var==0)
                        out.threefour[w] = 0.0f;
                    // Derivative Three: filter/weight = third layer value at weight index
                    paramOverWeight = f.l3[w % 10];
                    out.threefour[w] += learningRate*dot*paramOverWeight;
                }
            }

        return out;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE SkePUBPOut OMP(SkePUBPIn f, int samples, float learningRate)
{

            float paramOverWeight, dot;
            float errorOverColour[3];
            float colourOverParam[3];

            SkePUBPOut out;

            // Derivative One: samples * (cFiltered - cTarget)/(cTarget*cTarget)
            errorOverColour[0] = samples * (f.denoisedCol[0] - f.targetCol[0]) / (f.targetCol[0]*f.targetCol[0]+0.0001f);
            errorOverColour[1] = samples * (f.denoisedCol[1] - f.targetCol[1]) / (f.targetCol[1]*f.targetCol[1]+0.0001f);
            errorOverColour[2] = samples * (f.denoisedCol[2] - f.targetCol[2]) / (f.targetCol[2]*f.targetCol[2]+0.0001f);

            // Filter Paramaters (index, col, K)
            int w;
            for (int var=0;var<7;var++) {

                // Derivative Two: cross-bilateral filter derivative
                colourOverParam[0] = f.deriv.paramXYZ[var][0];
                colourOverParam[1] = f.deriv.paramXYZ[var][1];
                colourOverParam[2] = f.deriv.paramXYZ[var][2];

                // Dot Product
                dot = errorOverColour[0]*colourOverParam[0] + 
                      errorOverColour[1]*colourOverParam[1] + 
                      errorOverColour[2]*colourOverParam[2];

                // Weights One
                for (w=0;w<360;w++){
                    if (var==0)
                        out.onetwo[w] = 0.0f;
                    // Derivative Three: filter/weight = secondary feature input at weight index
                    paramOverWeight = f.s[w % 36];
                    out.onetwo[w] += learningRate*dot*paramOverWeight;
                }
                // Weights Two
                for (w=0;w<100;w++){
                    if (var==0)
                        out.twothree[w] = 0.0f;
                    // Derivative Three: filter/weight = second layer value at weight index
                    paramOverWeight = f.l2[w % 10];
                    out.twothree[w] += learningRate*dot*paramOverWeight;
                }
                // Weights Three
                for (w=0;w<70;w++){
                    if (var==0)
                        out.threefour[w] = 0.0f;
                    // Derivative Three: filter/weight = third layer value at weight index
                    paramOverWeight = f.l3[w % 10];
                    out.threefour[w] += learningRate*dot*paramOverWeight;
                }
            }

        return out;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE SkePUBPOut CPU(SkePUBPIn f, int samples, float learningRate)
{

            float paramOverWeight, dot;
            float errorOverColour[3];
            float colourOverParam[3];

            SkePUBPOut out;

            // Derivative One: samples * (cFiltered - cTarget)/(cTarget*cTarget)
            errorOverColour[0] = samples * (f.denoisedCol[0] - f.targetCol[0]) / (f.targetCol[0]*f.targetCol[0]+0.0001f);
            errorOverColour[1] = samples * (f.denoisedCol[1] - f.targetCol[1]) / (f.targetCol[1]*f.targetCol[1]+0.0001f);
            errorOverColour[2] = samples * (f.denoisedCol[2] - f.targetCol[2]) / (f.targetCol[2]*f.targetCol[2]+0.0001f);

            // Filter Paramaters (index, col, K)
            int w;
            for (int var=0;var<7;var++) {

                // Derivative Two: cross-bilateral filter derivative
                colourOverParam[0] = f.deriv.paramXYZ[var][0];
                colourOverParam[1] = f.deriv.paramXYZ[var][1];
                colourOverParam[2] = f.deriv.paramXYZ[var][2];

                // Dot Product
                dot = errorOverColour[0]*colourOverParam[0] + 
                      errorOverColour[1]*colourOverParam[1] + 
                      errorOverColour[2]*colourOverParam[2];

                // Weights One
                for (w=0;w<360;w++){
                    if (var==0)
                        out.onetwo[w] = 0.0f;
                    // Derivative Three: filter/weight = secondary feature input at weight index
                    paramOverWeight = f.s[w % 36];
                    out.onetwo[w] += learningRate*dot*paramOverWeight;
                }
                // Weights Two
                for (w=0;w<100;w++){
                    if (var==0)
                        out.twothree[w] = 0.0f;
                    // Derivative Three: filter/weight = second layer value at weight index
                    paramOverWeight = f.l2[w % 10];
                    out.twothree[w] += learningRate*dot*paramOverWeight;
                }
                // Weights Three
                for (w=0;w<70;w++){
                    if (var==0)
                        out.threefour[w] = 0.0f;
                    // Derivative Three: filter/weight = third layer value at weight index
                    paramOverWeight = f.l3[w % 10];
                    out.threefour[w] += learningRate*dot*paramOverWeight;
                }
            }

        return out;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "skepu_skel_0_SkePUDenoiserNN_MapKernel_SkePUBPFunc.cu"

struct skepu_userfunction_skepu_skel_1convol_SkePUFDFunc
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
constexpr static bool usesPRNG = 0;
constexpr static size_t randomCount = SKEPU_NO_RANDOM;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<int>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = FilterDerivOut;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ FilterDerivOut CU(skepu::Region2D<FilterDerivIn> r, int samples)
{
    // Contribution to the filtered colour of the filter paramater

    int dVal, var, c;
    float vecSum[3];
    float weightOverParam = 0.0f;

    FilterDerivOut fDeriv;

    float dVals[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dValMult = 1.0f;

    float iCol[3]   ;
    float iNormal[3]  ;
    float iAlbedo1[3] ;
    float iAlbedo2[3] ;
    float iWorldPos[3];
    float iDirectLight;
    for (c =0; c<3;c++) {
        iCol[c]      = r(0,0).preScreen[c] / samples;
        iNormal[c]   = r(0,0).normal[c]    / samples;
        iAlbedo1[c]  = r(0,0).alb1[c]      / samples;
        iAlbedo2[c]  = r(0,0).alb2[c]      / samples;
        iWorldPos[c] = r(0,0).worldPos[c]  / samples;
    }
    iDirectLight = r(0,0).directLight / samples;

    float  jCol[3]      ;
    float  jNormal[3]   ;
    float  jAlbedo1[3]  ;
    float  jAlbedo2[3]  ;
    float  jWorldPos[3] ;
    float jDirectLight;

    for (var=0; var<7; var++) {
        fDeriv.paramXYZ[var][0] = 0.0f;
        fDeriv.paramXYZ[var][1] = 0.0f;
        fDeriv.paramXYZ[var][2] = 0.0f;
    }

    float paramDiffs[7];

    // For all pixels j around i
    for (int j = -r.oj; j <= r.oj; j++) {
        for (int i = -r.oi; i <= r.oi; i++) {

            vecSum[0] = (r(j, i).preScreen[0] - r(j, i).denoisedCol[0]) / r(j, i).wcSum;
            vecSum[1] = (r(j, i).preScreen[1] - r(j, i).denoisedCol[1]) / r(j, i).wcSum;
            vecSum[2] = (r(j, i).preScreen[2] - r(j, i).denoisedCol[2]) / r(j, i).wcSum;

            for (c =0; c<3;c++) {
                jCol[c]      = r(j,i).preScreen[c] / samples;
                jNormal[c]   = r(j,i).normal[c]    / samples;
                jAlbedo1[c]  = r(j,i).alb1[c]      / samples;
                jAlbedo2[c]  = r(j,i).alb2[c]      / samples;
                jWorldPos[c] = r(j,i).worldPos[c]  / samples;
            }
            jDirectLight = r(j,i).directLight / samples;


            // Index d value
            paramDiffs[0] = pow(j,2)+pow(i,2);
            dVals[0] = paramDiffs[0] / (2.0f * r(0,0).variances[0] + 0.000001f);
            // Colour d value
            paramDiffs[0] = pow(iCol[0] - jCol[0],2)+pow(iCol[1] - jCol[1],2) + pow(iCol[2] - jCol[2],2);
            dVals[1] = paramDiffs[1] / (2.0f * r(0,0).variances[1] * (r(0,0).stdDev[0] + r(j,i).stdDev[0])  + 0.000001f);
            // Normal d value
            paramDiffs[2] = pow(iNormal[0] - jNormal[0],2)+pow(iNormal[1] - jNormal[1],2) + pow(iNormal[2] - jNormal[2],2);
            dVals[2] = paramDiffs[2] / (2.0f * r(0,0).variances[2] * r(0,0).stdDev[1]  + 0.000001f);
            // Alb1 d value
            paramDiffs[3] = pow(iAlbedo1[0] - jAlbedo1[0],2)+pow(iAlbedo1[1] - jAlbedo1[1],2) + pow(iAlbedo1[2] - jAlbedo1[2],2);
            dVals[3] = paramDiffs[3] / (2.0f * r(0,0).variances[3] * r(0,0).stdDev[2]  + 0.000001f);
            // Alb2 d value
            paramDiffs[4] = pow(iAlbedo2[0] - jAlbedo2[0],2)+pow(iAlbedo2[1] - jAlbedo2[1],2) + pow(iAlbedo2[2] - jAlbedo2[2],2);
            dVals[4] = paramDiffs[4] / (2.0f * r(0,0).variances[4] * r(0,0).stdDev[3]  + 0.000001f);
            // worldPos d value
            paramDiffs[5] = pow(iWorldPos[0] - jWorldPos[0],2)+pow(iWorldPos[1] - jWorldPos[1],2) + pow(iWorldPos[2] - jWorldPos[2],2);
            dVals[5] = paramDiffs[5] / (2.0f * r(0,0).variances[5] * r(0,0).stdDev[4]  + 0.000001f);
            // directLight d value
            paramDiffs[6] = pow(iDirectLight - jDirectLight,2);
            dVals[6] = paramDiffs[6] / (2.0f * r(0,0).variances[6] * r(0,0).stdDev[5] + 0.000001f);

            dValMult = 1.0f;
            for (dVal=0;dVal<7;dVal++) {
                dValMult *= exp(-dVals[dVal]) + 0.000001f;;
            }

            for (var=0; var<7; var++) {
                weightOverParam = dValMult * paramDiffs[var] / pow(r(0,0).variances[var],3);
                fDeriv.paramXYZ[var][0] += vecSum[0]*weightOverParam;
                fDeriv.paramXYZ[var][1] += vecSum[1]*weightOverParam;
                fDeriv.paramXYZ[var][2] += vecSum[2]*weightOverParam;
            }

        }      
    } 

    return fDeriv;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE FilterDerivOut OMP(skepu::Region2D<FilterDerivIn> r, int samples)
{
    // Contribution to the filtered colour of the filter paramater

    int dVal, var, c;
    float vecSum[3];
    float weightOverParam = 0.0f;

    FilterDerivOut fDeriv;

    float dVals[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dValMult = 1.0f;

    float iCol[3]   ;
    float iNormal[3]  ;
    float iAlbedo1[3] ;
    float iAlbedo2[3] ;
    float iWorldPos[3];
    float iDirectLight;
    for (c =0; c<3;c++) {
        iCol[c]      = r(0,0).preScreen[c] / samples;
        iNormal[c]   = r(0,0).normal[c]    / samples;
        iAlbedo1[c]  = r(0,0).alb1[c]      / samples;
        iAlbedo2[c]  = r(0,0).alb2[c]      / samples;
        iWorldPos[c] = r(0,0).worldPos[c]  / samples;
    }
    iDirectLight = r(0,0).directLight / samples;

    float  jCol[3]      ;
    float  jNormal[3]   ;
    float  jAlbedo1[3]  ;
    float  jAlbedo2[3]  ;
    float  jWorldPos[3] ;
    float jDirectLight;

    for (var=0; var<7; var++) {
        fDeriv.paramXYZ[var][0] = 0.0f;
        fDeriv.paramXYZ[var][1] = 0.0f;
        fDeriv.paramXYZ[var][2] = 0.0f;
    }

    float paramDiffs[7];

    // For all pixels j around i
    for (int j = -r.oj; j <= r.oj; j++) {
        for (int i = -r.oi; i <= r.oi; i++) {

            vecSum[0] = (r(j, i).preScreen[0] - r(j, i).denoisedCol[0]) / r(j, i).wcSum;
            vecSum[1] = (r(j, i).preScreen[1] - r(j, i).denoisedCol[1]) / r(j, i).wcSum;
            vecSum[2] = (r(j, i).preScreen[2] - r(j, i).denoisedCol[2]) / r(j, i).wcSum;

            for (c =0; c<3;c++) {
                jCol[c]      = r(j,i).preScreen[c] / samples;
                jNormal[c]   = r(j,i).normal[c]    / samples;
                jAlbedo1[c]  = r(j,i).alb1[c]      / samples;
                jAlbedo2[c]  = r(j,i).alb2[c]      / samples;
                jWorldPos[c] = r(j,i).worldPos[c]  / samples;
            }
            jDirectLight = r(j,i).directLight / samples;


            // Index d value
            paramDiffs[0] = pow(j,2)+pow(i,2);
            dVals[0] = paramDiffs[0] / (2.0f * r(0,0).variances[0] + 0.000001f);
            // Colour d value
            paramDiffs[0] = pow(iCol[0] - jCol[0],2)+pow(iCol[1] - jCol[1],2) + pow(iCol[2] - jCol[2],2);
            dVals[1] = paramDiffs[1] / (2.0f * r(0,0).variances[1] * (r(0,0).stdDev[0] + r(j,i).stdDev[0])  + 0.000001f);
            // Normal d value
            paramDiffs[2] = pow(iNormal[0] - jNormal[0],2)+pow(iNormal[1] - jNormal[1],2) + pow(iNormal[2] - jNormal[2],2);
            dVals[2] = paramDiffs[2] / (2.0f * r(0,0).variances[2] * r(0,0).stdDev[1]  + 0.000001f);
            // Alb1 d value
            paramDiffs[3] = pow(iAlbedo1[0] - jAlbedo1[0],2)+pow(iAlbedo1[1] - jAlbedo1[1],2) + pow(iAlbedo1[2] - jAlbedo1[2],2);
            dVals[3] = paramDiffs[3] / (2.0f * r(0,0).variances[3] * r(0,0).stdDev[2]  + 0.000001f);
            // Alb2 d value
            paramDiffs[4] = pow(iAlbedo2[0] - jAlbedo2[0],2)+pow(iAlbedo2[1] - jAlbedo2[1],2) + pow(iAlbedo2[2] - jAlbedo2[2],2);
            dVals[4] = paramDiffs[4] / (2.0f * r(0,0).variances[4] * r(0,0).stdDev[3]  + 0.000001f);
            // worldPos d value
            paramDiffs[5] = pow(iWorldPos[0] - jWorldPos[0],2)+pow(iWorldPos[1] - jWorldPos[1],2) + pow(iWorldPos[2] - jWorldPos[2],2);
            dVals[5] = paramDiffs[5] / (2.0f * r(0,0).variances[5] * r(0,0).stdDev[4]  + 0.000001f);
            // directLight d value
            paramDiffs[6] = pow(iDirectLight - jDirectLight,2);
            dVals[6] = paramDiffs[6] / (2.0f * r(0,0).variances[6] * r(0,0).stdDev[5] + 0.000001f);

            dValMult = 1.0f;
            for (dVal=0;dVal<7;dVal++) {
                dValMult *= exp(-dVals[dVal]) + 0.000001f;;
            }

            for (var=0; var<7; var++) {
                weightOverParam = dValMult * paramDiffs[var] / pow(r(0,0).variances[var],3);
                fDeriv.paramXYZ[var][0] += vecSum[0]*weightOverParam;
                fDeriv.paramXYZ[var][1] += vecSum[1]*weightOverParam;
                fDeriv.paramXYZ[var][2] += vecSum[2]*weightOverParam;
            }

        }      
    } 

    return fDeriv;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE FilterDerivOut CPU(skepu::Region2D<FilterDerivIn> r, int samples)
{
    // Contribution to the filtered colour of the filter paramater

    int dVal, var, c;
    float vecSum[3];
    float weightOverParam = 0.0f;

    FilterDerivOut fDeriv;

    float dVals[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dValMult = 1.0f;

    float iCol[3]   ;
    float iNormal[3]  ;
    float iAlbedo1[3] ;
    float iAlbedo2[3] ;
    float iWorldPos[3];
    float iDirectLight;
    for (c =0; c<3;c++) {
        iCol[c]      = r(0,0).preScreen[c] / samples;
        iNormal[c]   = r(0,0).normal[c]    / samples;
        iAlbedo1[c]  = r(0,0).alb1[c]      / samples;
        iAlbedo2[c]  = r(0,0).alb2[c]      / samples;
        iWorldPos[c] = r(0,0).worldPos[c]  / samples;
    }
    iDirectLight = r(0,0).directLight / samples;

    float  jCol[3]      ;
    float  jNormal[3]   ;
    float  jAlbedo1[3]  ;
    float  jAlbedo2[3]  ;
    float  jWorldPos[3] ;
    float jDirectLight;

    for (var=0; var<7; var++) {
        fDeriv.paramXYZ[var][0] = 0.0f;
        fDeriv.paramXYZ[var][1] = 0.0f;
        fDeriv.paramXYZ[var][2] = 0.0f;
    }

    float paramDiffs[7];

    // For all pixels j around i
    for (int j = -r.oj; j <= r.oj; j++) {
        for (int i = -r.oi; i <= r.oi; i++) {

            vecSum[0] = (r(j, i).preScreen[0] - r(j, i).denoisedCol[0]) / r(j, i).wcSum;
            vecSum[1] = (r(j, i).preScreen[1] - r(j, i).denoisedCol[1]) / r(j, i).wcSum;
            vecSum[2] = (r(j, i).preScreen[2] - r(j, i).denoisedCol[2]) / r(j, i).wcSum;

            for (c =0; c<3;c++) {
                jCol[c]      = r(j,i).preScreen[c] / samples;
                jNormal[c]   = r(j,i).normal[c]    / samples;
                jAlbedo1[c]  = r(j,i).alb1[c]      / samples;
                jAlbedo2[c]  = r(j,i).alb2[c]      / samples;
                jWorldPos[c] = r(j,i).worldPos[c]  / samples;
            }
            jDirectLight = r(j,i).directLight / samples;


            // Index d value
            paramDiffs[0] = pow(j,2)+pow(i,2);
            dVals[0] = paramDiffs[0] / (2.0f * r(0,0).variances[0] + 0.000001f);
            // Colour d value
            paramDiffs[0] = pow(iCol[0] - jCol[0],2)+pow(iCol[1] - jCol[1],2) + pow(iCol[2] - jCol[2],2);
            dVals[1] = paramDiffs[1] / (2.0f * r(0,0).variances[1] * (r(0,0).stdDev[0] + r(j,i).stdDev[0])  + 0.000001f);
            // Normal d value
            paramDiffs[2] = pow(iNormal[0] - jNormal[0],2)+pow(iNormal[1] - jNormal[1],2) + pow(iNormal[2] - jNormal[2],2);
            dVals[2] = paramDiffs[2] / (2.0f * r(0,0).variances[2] * r(0,0).stdDev[1]  + 0.000001f);
            // Alb1 d value
            paramDiffs[3] = pow(iAlbedo1[0] - jAlbedo1[0],2)+pow(iAlbedo1[1] - jAlbedo1[1],2) + pow(iAlbedo1[2] - jAlbedo1[2],2);
            dVals[3] = paramDiffs[3] / (2.0f * r(0,0).variances[3] * r(0,0).stdDev[2]  + 0.000001f);
            // Alb2 d value
            paramDiffs[4] = pow(iAlbedo2[0] - jAlbedo2[0],2)+pow(iAlbedo2[1] - jAlbedo2[1],2) + pow(iAlbedo2[2] - jAlbedo2[2],2);
            dVals[4] = paramDiffs[4] / (2.0f * r(0,0).variances[4] * r(0,0).stdDev[3]  + 0.000001f);
            // worldPos d value
            paramDiffs[5] = pow(iWorldPos[0] - jWorldPos[0],2)+pow(iWorldPos[1] - jWorldPos[1],2) + pow(iWorldPos[2] - jWorldPos[2],2);
            dVals[5] = paramDiffs[5] / (2.0f * r(0,0).variances[5] * r(0,0).stdDev[4]  + 0.000001f);
            // directLight d value
            paramDiffs[6] = pow(iDirectLight - jDirectLight,2);
            dVals[6] = paramDiffs[6] / (2.0f * r(0,0).variances[6] * r(0,0).stdDev[5] + 0.000001f);

            dValMult = 1.0f;
            for (dVal=0;dVal<7;dVal++) {
                dValMult *= exp(-dVals[dVal]) + 0.000001f;;
            }

            for (var=0; var<7; var++) {
                weightOverParam = dValMult * paramDiffs[var] / pow(r(0,0).variances[var],3);
                fDeriv.paramXYZ[var][0] += vecSum[0]*weightOverParam;
                fDeriv.paramXYZ[var][1] += vecSum[1]*weightOverParam;
                fDeriv.paramXYZ[var][2] += vecSum[2]*weightOverParam;
            }

        }      
    } 

    return fDeriv;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "SkePUDenoiserNN_Overlap2DKernel_SkePUFDFunc.cu"
void DenoiserNN::SkePUBackProp() {

    // Calc Filter Derivs with Map Overlap
    auto fIn  = skepu::Matrix<FilterDerivIn>(yRes, xRes);
    auto fOut = skepu::Matrix<FilterDerivOut>(yRes, xRes);
    int c;
    for (int ind = 0; ind < xRes*yRes; ind++) {

        for (c=0; c<3; c++) {
            fIn[ind].preScreen[c]    = preScreen[ind][c];
            fIn[ind].normal[c]       = normal[ind][c];
            fIn[ind].alb1[c]         = albedo1[ind][c];
            fIn[ind].alb2[c]         = albedo2[ind][c];
            fIn[ind].worldPos[c]     = worldPos[ind][c];
            fIn[ind].denoisedCol[c]  = denoisedCol[ind][c];
        }
        fIn[ind].directLight = directLight[ind].x;
        fIn[ind].wcSum = denoisingInf[ind].wcSum;

        for (c=0; c<7; c++)
            fIn[ind].variances[c] = denoisingInf[ind].variances[c];
        for (c=0; c<6; c++)
            fIn[ind].stdDev[c] = denoisingInf[ind].stdDev[c];
    }

    auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(denoisingSkePUBackend)};
	spec.activateBackend();
    skepu::backend::MapOverlap2D<skepu_userfunction_skepu_skel_1convol_SkePUFDFunc, decltype(&SkePUDenoiserNN_Overlap2DKernel_SkePUFDFunc_conv_cuda_2D_kernel), void> convol(SkePUDenoiserNN_Overlap2DKernel_SkePUFDFunc_conv_cuda_2D_kernel);
	convol.setBackend(spec);
    convol.setOverlap(denoisingN);
    convol.setEdgeMode(skepu::Edge::Duplicate);

    convol(fOut, fIn, sampleCount);
    //fOut.updateHost();

    //Calc New Weights with Map, then loop through weights to add

    auto bpIn  = skepu::Matrix<SkePUBPIn>(yRes, xRes);
    auto bpOut = skepu::Matrix<SkePUBPOut>(yRes, xRes);
    for (int ind = 0; ind < xRes*yRes; ind++) {

        for (c=0; c<3; c++) {
            bpIn[ind].targetCol[c]    = targetCol[ind][c];
            bpIn[ind].denoisedCol[c]  = denoisedCol[ind][c];
        }
        bpIn[ind].deriv = fOut[ind];

        for (c=0; c<36; c++)
             bpIn[ind].s[c] = sFeatures[ind](c);
        for (c=0; c<10; c++) {
            bpIn[ind].l2[c] = layerTwoValues[10*ind + c];
            bpIn[ind].l3[c] = layerThreeValues[10*ind + c];
        }
    }

    spec = skepu::BackendSpec{skepu::Backend::typeFromString(denoisingSkePUBackend)};
	spec.activateBackend();
    skepu::backend::Map<1, skepu_userfunction_skepu_skel_0map_SkePUBPFunc, decltype(&skepu_skel_0_SkePUDenoiserNN_MapKernel_SkePUBPFunc), void> map(skepu_skel_0_SkePUDenoiserNN_MapKernel_SkePUBPFunc);
	map.setBackend(spec);

    map(bpOut, bpIn, sampleCount, learningRate);
    //bpOut.updateHost();

    for (int ind = 0; ind < xRes*yRes; ind++) {
        for (c=0; c<360; c++)
             onetwo[c] += bpOut[ind].onetwo[c];
        for (c=0; c<100; c++) 
            twothree[c] += bpOut[ind].twothree[c];
        for (c=0; c<70; c++) 
            threefour[c] += bpOut[ind].threefour[c];
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

void DenoiserNN::InitTraining() {

    training = true;

    // Get Target
	denoiser.saveTargetCol();
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
			    renderer.Render();

            // Denoise (and forward prop)
            denoiser.denoise();

			// Get Error Value
			GenRelMSE();

            // Append Training File
            AppendTrainingFile();

			// Back Prop
            BackProp();

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
	const uint64_t s0 = constants.GloRandS[0];
	uint64_t s1 = constants.GloRandS[1];
	const uint64_t result_plus = rotl(s0 + s1, R) + s0;
	s1 ^= s0;
	constants.GloRandS[0] = rotl(s0, A) ^ s1 ^ (s1 << B);
	constants.GloRandS[1] = rotl(s1, C);
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
