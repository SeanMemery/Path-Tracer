#include "CUDAHeader.h"


__global__
void CUDADenoiseFunc(GPUInf* in, FilterVals* out, CUDADenoiseConstants* constants) {
        uint pi = (blockIdx.x * blockDim.x) + threadIdx.x;
        uint pj = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (pi >= constants->RESH || pj >= constants->RESV)
            return;

        int pIndex = pj*constants->RESH + pi;

        // sum up total weights and sum of individual weights times col of pixel
        float3 wSum {0,0,0};
        float wcSum = 0;

        // Center Pixel Specific

            // Standard Deviations
            float col1StdDev =        in[pIndex].stdDevs[0];
            float normalStdDev =      in[pIndex].stdDevs[1];
            float albedo1StdDev =     in[pIndex].stdDevs[2];
            float albedo2StdDev =     in[pIndex].stdDevs[3];
            float worldPosStdDev =    in[pIndex].stdDevs[4];
            float directLightStdDev = in[pIndex].stdDevs[5];

            // Variances
            float indexVariance =       in[pIndex].variances[0];
            float colVariance =         in[pIndex].variances[1];
            float normalVariance =      in[pIndex].variances[2];
            float albedo1Variance =     in[pIndex].variances[3];
            float albedo2Variance =     in[pIndex].variances[4];
            float worldPosVariance =    in[pIndex].variances[5];
            float directLightVariance = in[pIndex].variances[6];


            float pCol[3] =      {in[pIndex].col[0],      in[pIndex].col[1],      in[pIndex].col[2]};
            float pNormal[3] =   {in[pIndex].normal[0],   in[pIndex].normal[1],   in[pIndex].normal[2]};
            float pAlbedo1[3] =  {in[pIndex].albedo1[0],  in[pIndex].albedo1[1],  in[pIndex].albedo1[2]};
            float pAlbedo2[3] =  {in[pIndex].albedo2[0],  in[pIndex].albedo2[1],  in[pIndex].albedo2[2]};
            float pWorldPos[3] = {in[pIndex].worldPos[0], in[pIndex].worldPos[1], in[pIndex].worldPos[2]};
            float pDirectLight =  in[pIndex].directLight;  

        // Center Pixel Specific

        // ij Pixel Specific 

            float ijCol[3]      = {0.0f, 0.0f, 0.0f};
            float ijNormal[3]   = {0.0f, 0.0f, 0.0f};
            float ijAlbedo1[3]  = {0.0f, 0.0f, 0.0f};
            float ijAlbedo2[3]  = {0.0f, 0.0f, 0.0f};
            float ijWorldPos[3] = {0.0f, 0.0f, 0.0f};
            float ijDirectLight = 0.0f;

            float indexVal =       1.0f;      
            float colVal =         1.0f;
            float normalVal =      1.0f;
            float albedo1Val =     1.0f;
            float albedo2Val =     1.0f;
            float worldPosVal =    1.0f;
            float directLightVal = 1.0f;
            float weight         = 1.0f;
            float col2StdDev     = 1.0f;

        // ij Pixel Specific 

        int _j, _i, _index;

        // Weight Sum
        for (int j = -constants->denoisingN; j <= constants->denoisingN; j++) {
            _j = pj + j < 0 ? 0 : (pj + j >= constants->RESV ? constants->RESV-1 : pj + j);
            for (int i = -constants->denoisingN; i <= constants->denoisingN; i++) {
                _i = pi + i < 0 ? 0 : (pi + i >= constants->RESH ? constants->RESH-1 : pi + i);
                _index = _j*constants->RESH + _i;

                ijCol[0] =      in[_index].col[0];      ijCol[1] =      in[_index].col[1];      ijCol[2] =      in[_index].col[2];     
                ijNormal[0] =   in[_index].normal[0];   ijNormal[1] =   in[_index].normal[1];   ijNormal[2] =   in[_index].normal[2];  
                ijAlbedo1[0] =  in[_index].albedo1[0];  ijAlbedo1[1] =  in[_index].albedo1[1];  ijAlbedo1[2] =  in[_index].albedo1[2]; 
                ijAlbedo2[0] =  in[_index].albedo2[0];  ijAlbedo2[1] =  in[_index].albedo2[1];  ijAlbedo2[2] =  in[_index].albedo2[2]; 
                ijWorldPos[0] = in[_index].worldPos[0]; ijWorldPos[1] = in[_index].worldPos[1]; ijWorldPos[2] = in[_index].worldPos[2];
                ijDirectLight = in[_index].directLight;

                col2StdDev = in[_index].stdDevs[0] + 0.000001f;

                // INDEX
                indexVal = (j*j + i*i)/(2.0f * indexVariance);  
                indexVal = expf(-indexVal);
                // COLOUR
                colVal = (powf(ijCol[0]-pCol[0],2) + powf(ijCol[1]-pCol[1],2) + powf(ijCol[2]-pCol[2],2))/((col1StdDev + col2StdDev) * 2.0f * colVariance + 0.000001f); 
                colVal = expf(-colVal);
                // NORMAL
                normalVal = (powf(ijNormal[0]-pNormal[0],2) + powf(ijNormal[1]-pNormal[1],2) + powf(ijNormal[2]-pNormal[2],2))/(normalStdDev * 2.0f * normalVariance + 0.000001f); 
                normalVal = expf(-normalVal);
                // ALBEDO1
                albedo1Val = (powf(ijAlbedo1[0]-pAlbedo1[0],2) + powf(ijAlbedo1[1]-pAlbedo1[1],2) + powf(ijAlbedo1[2]-pAlbedo1[2],2))/(albedo1StdDev * 2.0f * albedo1Variance + 0.000001f); 
                albedo1Val = expf(-albedo1Val);
                // ALBEDO2
                albedo2Val = (powf(ijAlbedo2[0]-pAlbedo2[0],2) + powf(ijAlbedo2[1]-pAlbedo2[1],2) + powf(ijAlbedo2[2]-pAlbedo2[2],2))/(albedo2StdDev * 2.0f * albedo2Variance + 0.000001f); 
                albedo2Val = expf(-albedo2Val);
                // WORLD POS
                worldPosVal = (powf(ijWorldPos[0]-pWorldPos[0],2) + powf(ijWorldPos[1]-pWorldPos[1],2) + powf(ijWorldPos[2]-pWorldPos[2],2))/( 2.0f * worldPosStdDev *worldPosVariance + 0.000001f); 
                worldPosVal = expf(-worldPosVal);
                //DIRECT LIGHT
                directLightVal = powf(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance + 0.000001f); 
                directLightVal = expf(-directLightVal);

                weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                wcSum += weight;

                wSum.x += ijCol[0]*weight;
                wSum.y += ijCol[1]*weight;
                wSum.z += ijCol[2]*weight;
            }
        }
        FilterVals* ret = &out[pIndex];
        ret->x = wSum.x / wcSum;
        ret->y = wSum.y / wcSum;
        ret->z = wSum.z / wcSum;
        ret->wcSum = wcSum;
}


void CUDADenoiser::denoise() {
        int numPixels = xRes*yRes;

        CUDAConstants->RESH = xRes;
        CUDAConstants->RESV = yRes;
        CUDAConstants->denoisingN = denoisingN;

        int v, sd, c;
        for (int ind = 0; ind < numPixels; ind++) {

            GPUInf tempInf;

            // Final Col
            tempInf.col[0] = preScreen[ind].x / sampleCount;
            tempInf.col[1] = preScreen[ind].y / sampleCount;
            tempInf.col[2] = preScreen[ind].z / sampleCount; 
            // Normal
            tempInf.normal[0] = normal[ind].x / sampleCount;
            tempInf.normal[1] = normal[ind].y / sampleCount;
            tempInf.normal[2] = normal[ind].z / sampleCount;
            // Albedo1
            tempInf.albedo1[0] = albedo1[ind].x / sampleCount;
            tempInf.albedo1[1] = albedo1[ind].y / sampleCount;
            tempInf.albedo1[2] = albedo1[ind].z / sampleCount;
            // Albedo2
            tempInf.albedo2[0] = albedo2[ind].x / sampleCount;
            tempInf.albedo2[1] = albedo2[ind].y / sampleCount;
            tempInf.albedo2[2] = albedo2[ind].z / sampleCount;
            // World Pos
            tempInf.worldPos[0] = worldPos[ind].x / sampleCount;
            tempInf.worldPos[1] = worldPos[ind].y / sampleCount;
            tempInf.worldPos[2] = worldPos[ind].z / sampleCount;
            // Direct Light
            tempInf.directLight = directLight[ind].x / sampleCount;
            // Std Devs
            for (sd=0; sd < 6; sd++)
                tempInf.stdDevs[sd] = denoisingInf[ind].stdDev[sd];
            // Variances
            for (v=0; v < 7; v++)
                tempInf.variances[v] = denoisingInf[ind].variances[v];

            CUDAIn[ind] = tempInf;
        }

        dim3 numBlocks(xRes/rootThreadsPerBlock + 1, 
            yRes/rootThreadsPerBlock + 1); 

        CUDADenoiseFunc<<<numBlocks,dim3(rootThreadsPerBlock, rootThreadsPerBlock)>>>(CUDAIn, CUDAOut, CUDAConstants);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        FilterVals res;
        for (int ind = 0; ind < numPixels; ind++) {
            res = CUDAOut[ind];
            denoisedCol[ind] = vec3(res.x, res.y, res.z);
            denoisingInf[ind].wcSum = res.wcSum;
        }
}

void CUDADenoiser::InitBuffers() {
    cudaMallocManaged(&CUDAIn,  xRes*yRes*sizeof(GPUInf));
    cudaMallocManaged(&CUDAOut, xRes*yRes*sizeof(FilterVals));
    cudaMallocManaged(&CUDAConstants, sizeof(CUDADenoiseConstants));
}

void CUDADenoiser::FreeBuffers() {
    // Free memory
    cudaFree(CUDAIn);
    cudaFree(CUDAConstants);
    cudaFree(CUDAOut);
}
