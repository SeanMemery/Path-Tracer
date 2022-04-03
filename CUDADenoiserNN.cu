#include "CUDAHeader.h"
#include "DenoiserNN.h"

__global__
void CUDAForwardPropFunc(ForPropIn* in, ForPropOut* out, FPConstants* sConstants) {
    // Secondary Features
    uint pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint pj = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (pi >= sConstants->RESH || pj >= sConstants->RESV)
        return;

    int pixel = pj*sConstants->RESH + pi;

    ForPropOut* ret = &out[pixel];
    {
        // - K Means of single pixel, K std deviations of single pixel, K Means of 7x7 block, K std deviations of 7x7 block (20 total)
        // - Magnitude of gradients of K features of single pixel (sobel operator) (5 total)
        // - Sum of the abs difference between K of each pixel in 3x3 block and the mean of that 3x3 block (5 total)
        // - MAD of K features, so median of values minus median value, in NxN block (5 total)
        // - 1/totalSamples (1 total)

        int ind, c = 0;
        int _index = 0;
        int _i, _j = 0;
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
            ret->l2[c] = 0.0f;
            ret->l3[c] = 0.0f;
        }     
        for (c=0;c<7;c++) 
            ret->variances[c] = 0.0f;
        for (c=0; c<5; c++) {
            ret->meansSingle[c] = 0.0f;
            ret->sdSingle[c] = 0.0f;
            ret->meansBlock[c] = 0.0f;
            ret->sdBlock[c] = 0.0f;
            ret->gradients[c] = 0.0f;
            ret->meanDeviation[c] = 0.0f;
            ret->MAD[c] = 0.0f; 
        }
        ret->L = 0.0f;

        // K single Means
        ret->meansSingle[0] = in[pixel].normal;
        ret->meansSingle[1] = in[pixel].alb1;
        ret->meansSingle[2] = in[pixel].alb2;
        ret->meansSingle[3] = in[pixel].worldPos;
        ret->meansSingle[4] = in[pixel].directLight;

        // K single std devs
		ret->sdSingle[0] = in[pixel].stdDev[1];
		ret->sdSingle[1] = in[pixel].stdDev[2];
		ret->sdSingle[2] = in[pixel].stdDev[3];
		ret->sdSingle[3] = in[pixel].stdDev[4];
        ret->sdSingle[4] = in[pixel].stdDev[5];

        // K block means
        for ( j = -3; j <= 3; j++) { 
            _j = pj + j < 0 ? 0 : (pj + j >= sConstants->RESV ? sConstants->RESV-1 : pj + j);
            for ( i = -3; i <= 3; i++) { 
                _i = pi + i < 0 ? 0 : (pi + i >= sConstants->RESH ? sConstants->RESH-1 : pi + i);
                _index = _j*sConstants->RESH + _i;
                ret->meansBlock[0] += in[_index].normal      / 49.0f; 
                ret->meansBlock[1] += in[_index].alb1        / 49.0f; 
                ret->meansBlock[2] += in[_index].alb2        / 49.0f; 
                ret->meansBlock[3] += in[_index].worldPos    / 49.0f; 
                ret->meansBlock[4] += in[_index].directLight / 49.0f; 
                // 3x3 means for mean deviation
                if (abs(j) <= 1 && abs(i) <= 1) {
                    meansForMD[0] += in[_index].normal      / 9.0f; 
                    meansForMD[1] += in[_index].alb1        / 9.0f; 
                    meansForMD[2] += in[_index].alb2        / 9.0f; 
                    meansForMD[3] += in[_index].worldPos    / 9.0f; 
                    meansForMD[4] += in[_index].directLight / 9.0f; 
                }
            }
        }
        // K block std dev (std dev of col to mean col, or avg of std dev of each ? )
        for ( j = -3; j <= 3; j++) { 
            _j = pj + j < 0 ? 0 : (pj + j >= sConstants->RESV ? sConstants->RESV-1 : pj + j);
            for ( i = -3; i <= 3; i++) { 
                _i = pi + i < 0 ? 0 : (pi + i >= sConstants->RESH ? sConstants->RESH-1 : pi + i);
                _index = _j*sConstants->RESH + _i;
                ret->sdBlock[0] += powf((in[_index].normal      - ret->meansBlock[0]),2); 
                ret->sdBlock[1] += powf((in[_index].alb1        - ret->meansBlock[1]),2); 
                ret->sdBlock[2] += powf((in[_index].alb2        - ret->meansBlock[2]),2); 
                ret->sdBlock[3] += powf((in[_index].worldPos    - ret->meansBlock[3]),2); 
                ret->sdBlock[4] += powf((in[_index].directLight - ret->meansBlock[4]),2); 
            }
        }
        ret->sdBlock[0] = sqrt(ret->sdBlock[0] / 49.0f);
        ret->sdBlock[1] = sqrt(ret->sdBlock[1] / 49.0f);
        ret->sdBlock[2] = sqrt(ret->sdBlock[2] / 49.0f);
        ret->sdBlock[3] = sqrt(ret->sdBlock[3] / 49.0f);
        ret->sdBlock[4] = sqrt(ret->sdBlock[4] / 49.0f);
        // K Gradients (3x3 Block)
        // K Mean Deviations (3x3 block)
        for ( j = -1; j <= 1; j++) { 
            _j = pj + j < 0 ? 0 : (pj + j >= sConstants->RESV ? sConstants->RESV-1 : pj + j);
            for ( i = -1; i <= 1; i++) { 
                _i = pi + i < 0 ? 0 : (pi + i >= sConstants->RESH ? sConstants->RESH-1 : pi + i);
                _index = _j*sConstants->RESH + _i;
                linearInd = (j+1)*3 + i + 1;
                // Sobel operator
                GxSum[0] += Gx[linearInd]*in[_index].normal;
                GySum[0] += Gy[linearInd]*in[_index].normal;
                GxSum[1] += Gx[linearInd]*in[_index].alb1;
                GySum[1] += Gy[linearInd]*in[_index].alb1;
                GxSum[2] += Gx[linearInd]*in[_index].alb2;
                GySum[2] += Gy[linearInd]*in[_index].alb2;
                GxSum[3] += Gx[linearInd]*in[_index].worldPos;
                GySum[3] += Gy[linearInd]*in[_index].worldPos;
                GxSum[4] += Gx[linearInd]*in[_index].directLight;
                GySum[4] += Gy[linearInd]*in[_index].directLight;

                // Mean Abs Diff
                ret->meanDeviation[0] += fabs(in[_index].normal      - meansForMD[0])/ 9.0f; 
                ret->meanDeviation[1] += fabs(in[_index].alb1        - meansForMD[1])/ 9.0f; 
                ret->meanDeviation[2] += fabs(in[_index].alb2        - meansForMD[2])/ 9.0f; 
                ret->meanDeviation[3] += fabs(in[_index].worldPos    - meansForMD[3])/ 9.0f; 
                ret->meanDeviation[4] += fabs(in[_index].directLight - meansForMD[4])/ 9.0f; 

                // Collect MAD Values
                valuesForMAD[0][linearInd] = in[_index].normal     ;
                valuesForMAD[1][linearInd] = in[_index].alb1       ;
                valuesForMAD[2][linearInd] = in[_index].alb2       ;
                valuesForMAD[3][linearInd] = in[_index].worldPos   ;
                valuesForMAD[4][linearInd] = in[_index].directLight;
            }
        }
        ret->gradients[0] = sqrt(GxSum[0]*GxSum[0] + GySum[0]*GySum[0]);
        ret->gradients[1] = sqrt(GxSum[1]*GxSum[1] + GySum[1]*GySum[1]);
        ret->gradients[2] = sqrt(GxSum[2]*GxSum[2] + GySum[2]*GySum[2]);
        ret->gradients[3] = sqrt(GxSum[3]*GxSum[3] + GySum[3]*GySum[3]);
        ret->gradients[4] = sqrt(GxSum[4]*GxSum[4] + GySum[4]*GySum[4]);
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
            ret->MAD[feature] = medianGetter[0];
        }
        // L
        ret->L = 1.0f/sConstants->samples;
    }

    // Variances
    {
        int node, weight, numNodes;

        // Layer 1 - 2
        for (node=0; node<10; node++) {
            ret->l2[node] = 0.0f;

            // Go through all secondary features
            {
                ret->l2[node] += ret->meansSingle[0]   * sConstants->onetwo[36*node + 0]; 
                ret->l2[node] += ret->meansSingle[1]   * sConstants->onetwo[36*node + 1]; 
                ret->l2[node] += ret->meansSingle[2]   * sConstants->onetwo[36*node + 2]; 
                ret->l2[node] += ret->meansSingle[3]   * sConstants->onetwo[36*node + 3]; 
                ret->l2[node] += ret->meansSingle[4]   * sConstants->onetwo[36*node + 4]; 
                ret->l2[node] += ret->sdSingle[0]      * sConstants->onetwo[36*node + 5];
                ret->l2[node] += ret->sdSingle[1]      * sConstants->onetwo[36*node + 6];
                ret->l2[node] += ret->sdSingle[2]      * sConstants->onetwo[36*node + 7];
                ret->l2[node] += ret->sdSingle[3]      * sConstants->onetwo[36*node + 8];
                ret->l2[node] += ret->sdSingle[4]      * sConstants->onetwo[36*node + 9];
                ret->l2[node] += ret->meansBlock[0]    * sConstants->onetwo[36*node + 10];
                ret->l2[node] += ret->meansBlock[1]    * sConstants->onetwo[36*node + 11];
                ret->l2[node] += ret->meansBlock[2]    * sConstants->onetwo[36*node + 12];
                ret->l2[node] += ret->meansBlock[3]    * sConstants->onetwo[36*node + 13];
                ret->l2[node] += ret->meansBlock[4]    * sConstants->onetwo[36*node + 14];
                ret->l2[node] += ret->sdBlock[0]       * sConstants->onetwo[36*node + 15];
                ret->l2[node] += ret->sdBlock[1]       * sConstants->onetwo[36*node + 16];
                ret->l2[node] += ret->sdBlock[2]       * sConstants->onetwo[36*node + 17];
                ret->l2[node] += ret->sdBlock[3]       * sConstants->onetwo[36*node + 18];
                ret->l2[node] += ret->sdBlock[4]       * sConstants->onetwo[36*node + 19];
                ret->l2[node] += ret->gradients[0]     * sConstants->onetwo[36*node + 20];
                ret->l2[node] += ret->gradients[1]     * sConstants->onetwo[36*node + 21];
                ret->l2[node] += ret->gradients[2]     * sConstants->onetwo[36*node + 22];
                ret->l2[node] += ret->gradients[3]     * sConstants->onetwo[36*node + 23];
                ret->l2[node] += ret->gradients[4]     * sConstants->onetwo[36*node + 24];
                ret->l2[node] += ret->meanDeviation[0] * sConstants->onetwo[36*node + 25];
                ret->l2[node] += ret->meanDeviation[1] * sConstants->onetwo[36*node + 26];
                ret->l2[node] += ret->meanDeviation[2] * sConstants->onetwo[36*node + 27];
                ret->l2[node] += ret->meanDeviation[3] * sConstants->onetwo[36*node + 28];
                ret->l2[node] += ret->meanDeviation[4] * sConstants->onetwo[36*node + 29];
                ret->l2[node] += ret->MAD[0]           * sConstants->onetwo[36*node + 30];
                ret->l2[node] += ret->MAD[1]           * sConstants->onetwo[36*node + 31];
                ret->l2[node] += ret->MAD[2]           * sConstants->onetwo[36*node + 32];
                ret->l2[node] += ret->MAD[3]           * sConstants->onetwo[36*node + 33];
                ret->l2[node] += ret->MAD[4]           * sConstants->onetwo[36*node + 34];
                ret->l2[node] += ret->L                * sConstants->onetwo[36*node + 35];
            }
            
            ret->l2[node] = 1.0f/(1.0f + expf(-ret->l2[node]));
        }

        // Layer 2 - 3
        for (node=0; node<10; node++) {
            ret->l3[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                ret->l3[node] += ret->l2[weight]*sConstants->twothree[10*node + weight];
            ret->l3[node] = 1.0f/(1.0f + expf(-ret->l3[node]));
        }

        // Layer 3 - 4
        for (node=0; node<7; node++) {
            ret->variances[node] = 0.0f;
            for (weight=0; weight<10; weight++)
                ret->variances[node] += ret->l3[weight]*sConstants->threefour[10*node + weight];
            ret->variances[node] = log(1.0f + expf(ret->variances[node]));
        }

    }
}

void CUDADenoiserNN::ForwardProp() {

    int numPixels = xRes*yRes;
    delete denoiserNN.sFeatures;
    denoiserNN.sFeatures = new SecondaryFeatures[numPixels];

    // Set sConstants
    int w;
    CUDAFPConstants->samples = sampleCount;
    for (w=0;w<360;w++)
        CUDAFPConstants->onetwo[w] = denoiserNN.onetwo[w];
    for (w=0;w<100;w++)
        CUDAFPConstants->twothree[w] = denoiserNN.twothree[w];
    for (w=0;w<70;w++)
        CUDAFPConstants->threefour[w] = denoiserNN.threefour[w];
    CUDAFPConstants->RESH = xRes;
    CUDAFPConstants->RESV = yRes;

    // Set Inputs
    for (int ind = 0; ind < xRes*yRes; ind++) {

        CUDAFPIn[ind].normal      = (normal[ind].x     + normal[ind].y   + normal[ind].z)   / (3.0f * sampleCount);
        CUDAFPIn[ind].alb1        = (albedo1[ind].x    + albedo1[ind].y  + albedo1[ind].z)  / (3.0f * sampleCount);
        CUDAFPIn[ind].alb2        = (albedo2[ind].x    + albedo2[ind].y  + albedo2[ind].z)  / (3.0f * sampleCount);
        CUDAFPIn[ind].worldPos    = (worldPos[ind].x   + worldPos[ind].y + worldPos[ind].z) / (3.0f * sampleCount);
        CUDAFPIn[ind].directLight = directLight[ind].x / sampleCount;

        CUDAFPIn[ind].stdDev[0] = denoisingInf[ind].stdDev[0];
        CUDAFPIn[ind].stdDev[1] = denoisingInf[ind].stdDev[1];
        CUDAFPIn[ind].stdDev[2] = denoisingInf[ind].stdDev[2];
        CUDAFPIn[ind].stdDev[3] = denoisingInf[ind].stdDev[3];
        CUDAFPIn[ind].stdDev[4] = denoisingInf[ind].stdDev[4];
        CUDAFPIn[ind].stdDev[5] = denoisingInf[ind].stdDev[5];
    }

    dim3 numBlocks(xRes/rootThreadsPerBlock + 1, 
                yRes/rootThreadsPerBlock + 1); 

    CUDAForwardPropFunc<<<numBlocks,dim3(rootThreadsPerBlock, rootThreadsPerBlock)>>>(CUDAFPIn, CUDAFPOut, CUDAFPConstants );

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    int ind, v;
    ForPropOut ret;
    DenoisingInf* info;
    SecondaryFeatures* s;
    for (ind =0; ind < numPixels; ind++) {

        ret = CUDAFPOut[ind];
        info = &denoisingInf[ind];
        s = &denoiserNN.sFeatures[ind];

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

__global__
void CUDAFilterDerivFunc(FilterDerivIn* in, FilterDerivOut* out, BPConstants* sConstants) {
    uint pi = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint pj = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (pi >= sConstants->RESH || pj >= sConstants->RESV)
        return;

    int pixel = pj*sConstants->RESH + pi;

    FilterDerivOut* fDeriv = &out[pixel];

    // Contribution to the filtered colour of the filter paramater

    int dVal, var, c;
    int samples = sConstants->samples;
    float vecSum[3];
    float weightOverParam = 0.0f;

    float dVals[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dValMult = 1.0f;

    float iCol[3]   ;
    float iNormal[3]  ;
    float iAlbedo1[3] ;
    float iAlbedo2[3] ;
    float iWorldPos[3];
    float iDirectLight;
    for (c =0; c<3;c++) {
        iCol[c]      = in[pixel].preScreen[c] / samples;
        iNormal[c]   = in[pixel].normal[c]    / samples;
        iAlbedo1[c]  = in[pixel].alb1[c]      / samples;
        iAlbedo2[c]  = in[pixel].alb2[c]      / samples;
        iWorldPos[c] = in[pixel].worldPos[c]  / samples;
    }
    iDirectLight = in[pixel].directLight / samples;

    float  jCol[3]      ;
    float  jNormal[3]   ;
    float  jAlbedo1[3]  ;
    float  jAlbedo2[3]  ;
    float  jWorldPos[3] ;
    float jDirectLight;

    for (var=0; var<7; var++) {
        fDeriv->paramXYZ[var][0] = 0.0f;
        fDeriv->paramXYZ[var][1] = 0.0f;
        fDeriv->paramXYZ[var][2] = 0.0f;
    }

    float paramDiffs[7];

    // For all pixels j around i
    int _j, _i, _index;
    for (int j = -sConstants->denoisingN; j <= sConstants->denoisingN; j++) {
        _j = pj + j < 0 ? 0 : (pj + j >= sConstants->RESV ? sConstants->RESV-1 : pj + j);
        for (int i = -sConstants->denoisingN; i <= sConstants->denoisingN; i++) {
            _i = pi + i < 0 ? 0 : (pi + i >= sConstants->RESH ? sConstants->RESH-1 : pi + i);
            _index = _j*sConstants->RESH + _i;

            vecSum[0] = (in[_index].preScreen[0] - in[_index].denoisedCol[0]) / in[_index].wcSum;
            vecSum[1] = (in[_index].preScreen[1] - in[_index].denoisedCol[1]) / in[_index].wcSum;
            vecSum[2] = (in[_index].preScreen[2] - in[_index].denoisedCol[2]) / in[_index].wcSum;

            for (c =0; c<3;c++) {
                jCol[c]      = in[_index].preScreen[c] / samples;
                jNormal[c]   = in[_index].normal[c]    / samples;
                jAlbedo1[c]  = in[_index].alb1[c]      / samples;
                jAlbedo2[c]  = in[_index].alb2[c]      / samples;
                jWorldPos[c] = in[_index].worldPos[c]  / samples;
            }
            jDirectLight = in[_index].directLight / samples;

            // Index d value
            paramDiffs[0] = powf(j,2)+powf(i,2);
            dVals[0] = paramDiffs[0] / (2.0f * in[pixel].variances[0] + 0.000001f);
            // Colour d value
            paramDiffs[1] = powf(iCol[0] - jCol[0],2)+powf(iCol[1] - jCol[1],2) + powf(iCol[2] - jCol[2],2);
            dVals[1] = paramDiffs[1] / (2.0f * in[pixel].variances[1] * (in[pixel].stdDev[0] + in[_index].stdDev[0])  + 0.000001f);
            // Normal d value
            paramDiffs[2] = powf(iNormal[0] - jNormal[0],2)+powf(iNormal[1] - jNormal[1],2) + powf(iNormal[2] - jNormal[2],2);
            dVals[2] = paramDiffs[2] / (2.0f * in[pixel].variances[2] * in[pixel].stdDev[1]  + 0.000001f);
            // Alb1 d value
            paramDiffs[3] = powf(iAlbedo1[0] - jAlbedo1[0],2)+powf(iAlbedo1[1] - jAlbedo1[1],2) + powf(iAlbedo1[2] - jAlbedo1[2],2);
            dVals[3] = paramDiffs[3] / (2.0f * in[pixel].variances[3] * in[pixel].stdDev[2]  + 0.000001f);
            // Alb2 d value
            paramDiffs[4] = powf(iAlbedo2[0] - jAlbedo2[0],2)+powf(iAlbedo2[1] - jAlbedo2[1],2) + powf(iAlbedo2[2] - jAlbedo2[2],2);
            dVals[4] = paramDiffs[4] / (2.0f * in[pixel].variances[4] * in[pixel].stdDev[3]  + 0.000001f);
            // worldPos d value
            paramDiffs[5] = powf(iWorldPos[0] - jWorldPos[0],2)+powf(iWorldPos[1] - jWorldPos[1],2) + powf(iWorldPos[2] - jWorldPos[2],2);
            dVals[5] = paramDiffs[5] / (2.0f * in[pixel].variances[5] * in[pixel].stdDev[4]  + 0.000001f);
            // directLight d value
            paramDiffs[6] = powf(iDirectLight - jDirectLight,2);
            dVals[6] = paramDiffs[6] / (2.0f * in[pixel].variances[6] * in[pixel].stdDev[5] + 0.000001f);

            dValMult = 1.0f;
            for (dVal=0;dVal<7;dVal++) {
                dValMult *= exp(-dVals[dVal]) + 0.000001f;
            }

            for (var=0; var<7; var++) {
                weightOverParam = dValMult * paramDiffs[var] / powf(in[pixel].variances[var],3);
                fDeriv->paramXYZ[var][0] += vecSum[0]*weightOverParam;
                fDeriv->paramXYZ[var][1] += vecSum[1]*weightOverParam;
                fDeriv->paramXYZ[var][2] += vecSum[2]*weightOverParam;
            }

        }      
    } 
}

__global__
void CUDABackPropFunc(SkePUBPIn* in, SkePUBPOut* out, BPConstants* sConstants) {
            uint pi = (blockIdx.x * blockDim.x) + threadIdx.x;
            uint pj = (blockIdx.y * blockDim.y) + threadIdx.y;

            if (pi >= sConstants->RESH || pj >= sConstants->RESV)
                return;

            int pixel = pj*sConstants->RESH + pi;


            float paramOverWeight, dot;
            float errorOverColour[3];
            float colourOverParam[3];

            SkePUBPOut* ret = &out[pixel];

            // Derivative One: samples * (cFiltered - cTarget)/(cTarget*cTarget)
            errorOverColour[0] = sConstants->samples * (in[pixel].denoisedCol[0] - in[pixel].targetCol[0]) / (in[pixel].targetCol[0]*in[pixel].targetCol[0]+0.0001f);
            errorOverColour[1] = sConstants->samples * (in[pixel].denoisedCol[1] - in[pixel].targetCol[1]) / (in[pixel].targetCol[1]*in[pixel].targetCol[1]+0.0001f);
            errorOverColour[2] = sConstants->samples * (in[pixel].denoisedCol[2] - in[pixel].targetCol[2]) / (in[pixel].targetCol[2]*in[pixel].targetCol[2]+0.0001f);

            // Filter Paramaters (index, col, K)
            int w;
            for (int var=0;var<7;var++) {

                // Derivative Two: cross-bilateral filter derivative
                colourOverParam[0] = in[pixel].deriv.paramXYZ[var][0];
                colourOverParam[1] = in[pixel].deriv.paramXYZ[var][1];
                colourOverParam[2] = in[pixel].deriv.paramXYZ[var][2];

                // Dot Product
                dot = errorOverColour[0]*colourOverParam[0] + 
                      errorOverColour[1]*colourOverParam[1] + 
                      errorOverColour[2]*colourOverParam[2];

                // Weights One
                for (w=0;w<360;w++){
                    if (var==0)
                        ret->onetwo[w] = 0.0f;
                    // Derivative Three: filter/weight = secondary feature input at weight index
                    paramOverWeight = in[pixel].s[w % 36];
                    ret->onetwo[w] += sConstants->learningRate*dot*paramOverWeight;
                }
                // Weights Two
                for (w=0;w<100;w++){
                    if (var==0)
                        ret->twothree[w] = 0.0f;
                    // Derivative Three: filter/weight = second layer value at weight index
                    paramOverWeight = in[pixel].l2[w % 10];
                    ret->twothree[w] += sConstants->learningRate*dot*paramOverWeight;
                }
                // Weights Three
                for (w=0;w<70;w++){
                    if (var==0)
                        ret->threefour[w] = 0.0f;
                    // Derivative Three: filter/weight = third layer value at weight index
                    paramOverWeight = in[pixel].l3[w % 10];
                    ret->threefour[w] += sConstants->learningRate*dot*paramOverWeight;
                }
            }
}

void CUDADenoiserNN::BackProp() {
    int numPixels = xRes*yRes;

    dim3 numBlocks(xRes/rootThreadsPerBlock + 1, 
                yRes/rootThreadsPerBlock + 1);

    CUDAConstants->RESH = xRes;
    CUDAConstants->RESV = yRes;
    CUDAConstants->samples = sampleCount;
    CUDAConstants->learningRate = denoiserNN.learningRate;
    CUDAConstants->denoisingN = denoisingN; 
    
    // Calc Filter Derivs with Map Overlap

    int c;
    for (int ind = 0; ind < numPixels; ind++) {

        for (c=0; c<3; c++) {
            CUDAFIn[ind].preScreen[c]    = preScreen[ind][c];
            CUDAFIn[ind].normal[c]       = normal[ind][c];
            CUDAFIn[ind].alb1[c]         = albedo1[ind][c];
            CUDAFIn[ind].alb2[c]         = albedo2[ind][c];
            CUDAFIn[ind].worldPos[c]     = worldPos[ind][c];
            CUDAFIn[ind].denoisedCol[c]  = denoisedCol[ind][c];
        }
        CUDAFIn[ind].directLight = directLight[ind].x;
        CUDAFIn[ind].wcSum = denoisingInf[ind].wcSum;

        for (c=0; c<7; c++)
            CUDAFIn[ind].variances[c] = denoisingInf[ind].variances[c];
        for (c=0; c<6; c++)
            CUDAFIn[ind].stdDev[c] = denoisingInf[ind].stdDev[c];
    }

    CUDAFilterDerivFunc<<<numBlocks,dim3(rootThreadsPerBlock, rootThreadsPerBlock)>>>(CUDAFIn, CUDAFOut, CUDAConstants);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    for (int ind = 0; ind < numPixels; ind++) {

        for (c=0; c<3; c++) {
            CUDAIn[ind].targetCol[c]    = targetCol[ind][c];
            CUDAIn[ind].denoisedCol[c]  = denoisedCol[ind][c];
        }
        CUDAIn[ind].deriv = CUDAFOut[ind];

        for (c=0; c<36; c++)
            CUDAIn[ind].s[c] = denoiserNN.sFeatures[ind](c);
        for (c=0; c<10; c++) {
            CUDAIn[ind].l2[c] = layerTwoValues[10*ind + c];
            CUDAIn[ind].l3[c] = layerThreeValues[10*ind + c];
        }
    }

    CUDABackPropFunc<<<numBlocks,dim3(rootThreadsPerBlock, rootThreadsPerBlock)>>>(CUDAIn, CUDAOut, CUDAConstants);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    for (int ind = 0; ind < numPixels; ind++) {
        for (c=0; c<360; c++)
            denoiserNN.onetwo[c] += CUDAOut[ind].onetwo[c];
        for (c=0; c<100; c++) 
            denoiserNN.twothree[c] += CUDAOut[ind].twothree[c];
        for (c=0; c<70; c++) 
            denoiserNN.threefour[c] += CUDAOut[ind].threefour[c];
    }
}

void CUDADenoiserNN::InitBuffers() {
    int numPixels = xRes*yRes;

    // Forward Prop
    cudaMallocManaged(&CUDAFPIn,  numPixels*sizeof(ForPropIn));
    cudaMallocManaged(&CUDAFPOut, numPixels*sizeof(ForPropOut));
    cudaMallocManaged(&CUDAFPConstants, sizeof(FPConstants));

    // Back Prop
    cudaMallocManaged(&CUDAConstants,  sizeof(BPConstants));

    cudaMallocManaged(&CUDAFIn,  numPixels*sizeof(FilterDerivIn));
    cudaMallocManaged(&CUDAFOut,  numPixels*sizeof(FilterDerivOut));

    cudaMallocManaged(&CUDAIn,  numPixels*sizeof(SkePUBPIn));
    cudaMallocManaged(&CUDAOut,  numPixels*sizeof(SkePUBPOut));
}

void CUDADenoiserNN::FreeBuffers() {

    // Free fp memory
    cudaFree(CUDAFPConstants);
    cudaFree(CUDAFPIn);
    cudaFree(CUDAFPOut);

    // Free bp memory
    cudaFree(CUDAFIn);
    cudaFree(CUDAFOut);
    cudaFree(CUDAConstants);
    cudaFree(CUDAIn);
    cudaFree(CUDAOut);

}