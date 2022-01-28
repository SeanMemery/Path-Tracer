#define SKEPU_PRECOMPILED 1
#define SKEPU_OPENMP 1
#define SKEPU_CUDA 1
#include "Denoiser.h"

    void Denoiser::denoise() {

        auto denoiseTimer = clock_::now();

        if (!denoiserNN.LoadWeights())
		    denoiserNN.RandomizeWeights();

        denoiserNN.GeneratePFeatures();
		denoiserNN.GenerateSFeatures();
		denoiserNN.ComputeVariances();

        switch(currentRenderer) {
            case 0:
                CPUDenoise();
                break;
            case 1:
                OMPDenoise();
                break;
            case 2:
                CUDADenoise();
                break;
            case 3:
                OpenGLDenoise();
                break;
            case 4:
                SkePUDenoise();
                break;
            case 5:
                SkePUDenoise();
                break;
            case 6:
                SkePUDenoise();
                break;
        }

        denoiseTime = std::chrono::duration_cast<milli_second_>(clock_::now() - denoiseTimer).count();
    }

    FilterVals Denoiser::SkePUFilter(skepu::Region2D<GPUInf> r) {

        // sum up total weights and sum of individual weights times col of pixel
        float wSum[3] = {0,0,0};
        float wcSum = 0;

        // Center Pixel Specific

            // Standard Deviations
            float colStdDev =         r(0,0).stdDevs[0] + 0.000001f;
            float normalStdDev =      r(0,0).stdDevs[1] + 0.000001f;
            float albedo1StdDev =     r(0,0).stdDevs[2] + 0.000001f;
            float albedo2StdDev =     r(0,0).stdDevs[3] + 0.000001f;
            float worldPosStdDev =    r(0,0).stdDevs[4] + 0.000001f;
            float directLightStdDev = r(0,0).stdDevs[5] + 0.000001f;

            // Variances
            float indexVariance =       r(0,0).variances[0];
            float colVariance =         r(0,0).variances[1];
            float normalVariance =      r(0,0).variances[2];
            float albedo1Variance =     r(0,0).variances[3];
            float albedo2Variance =     r(0,0).variances[4];
            float worldPosVariance =    r(0,0).variances[5];
            float directLightVariance = r(0,0).variances[6];


            float pCol[3] =      {r(0,0).col[0],      r(0,0).col[1],      r(0,0).col[2]};
            float pNormal[3] =   {r(0,0).normal[0],   r(0,0).normal[1],   r(0,0).normal[2]};
            float pAlbedo1[3] =  {r(0,0).albedo1[0],  r(0,0).albedo1[1],  r(0,0).albedo1[2]};
            float pAlbedo2[3] =  {r(0,0).albedo2[0],  r(0,0).albedo2[1],  r(0,0).albedo2[2]};
            float pWorldPos[3] = {r(0,0).worldPos[0], r(0,0).worldPos[1], r(0,0).worldPos[2]};
            float pDirectLight =  r(0,0).directLight;  

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

        // ij Pixel Specific 

        // Weight Sum
        for (int j = -r.oj; j <= r.oj; j++) {
            for (int i = -r.oi; i <= r.oi; i++) {

                ijCol[0] =      r(j, i).col[0];      ijCol[1] =      r(j, i).col[1];      ijCol[2] =      r(j, i).col[2];     
                ijNormal[0] =   r(j, i).normal[0];   ijNormal[1] =   r(j, i).normal[1];   ijNormal[2] =   r(j, i).normal[2];  
                ijAlbedo1[0] =  r(j, i).albedo1[0];  ijAlbedo1[1] =  r(j, i).albedo1[1];  ijAlbedo1[2] =  r(j, i).albedo1[2]; 
                ijAlbedo2[0] =  r(j, i).albedo2[0];  ijAlbedo2[1] =  r(j, i).albedo2[1];  ijAlbedo2[2] =  r(j, i).albedo2[2]; 
                ijWorldPos[0] = r(j, i).worldPos[0]; ijWorldPos[1] = r(j, i).worldPos[1]; ijWorldPos[2] = r(j, i).worldPos[2];
                ijDirectLight = r(j, i).directLight;

                // INDEX
                indexVal = (j*j + i*i)/(2.0f * indexVariance);  
                indexVal = exp(-indexVal);
                // COLOUR
                colVal = (pow(ijCol[0]-pCol[0],2) + pow(ijCol[1]-pCol[1],2) + pow(ijCol[2]-pCol[2],2))/(colStdDev * 2.0f * colVariance); 
                colVal = exp(-colVal);
                // NORMAL
                normalVal = (pow(ijNormal[0]-pNormal[0],2) + pow(ijNormal[1]-pNormal[1],2) + pow(ijNormal[2]-pNormal[2],2))/(normalStdDev * 2.0f * normalVariance); 
                normalVal = exp(-normalVal);
                // ALBEDO1
                albedo1Val = (pow(ijAlbedo1[0]-pAlbedo1[0],2) + pow(ijAlbedo1[1]-pAlbedo1[1],2) + pow(ijAlbedo1[2]-pAlbedo1[2],2))/(albedo1StdDev * 2.0f * albedo1Variance); 
                albedo1Val = exp(-albedo1Val);
                // ALBEDO2
                albedo2Val = (pow(ijAlbedo2[0]-pAlbedo2[0],2) + pow(ijAlbedo2[1]-pAlbedo2[1],2) + pow(ijAlbedo2[2]-pAlbedo2[2],2))/(albedo2StdDev * 2.0f * albedo2Variance); 
                albedo2Val = exp(-albedo2Val);
                // WORLD POS
                worldPosVal = (pow(ijWorldPos[0]-pWorldPos[0],2) + pow(ijWorldPos[1]-pWorldPos[1],2) + pow(ijWorldPos[2]-pWorldPos[2],2))/(worldPosStdDev * 2.0f * worldPosVariance); 
                worldPosVal = exp(-worldPosVal);
                //DIRECT LIGHT
                directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance); 
                directLightVal = exp(-directLightVal);

                weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                wcSum += weight;

                wSum[0] += r(j, i).col[0]*weight;
                wSum[1] += r(j, i).col[1]*weight;
                wSum[2] += r(j, i).col[2]*weight;
            }
        }

        FilterVals ret;
        ret.x = wSum[0] / wcSum;
        ret.y = wSum[1] / wcSum;
        ret.z = wSum[2] / wcSum;
        ret.wcSum = wcSum;
        return ret;
    }

    float Denoiser::getMSE() {
        float MSE = 0.0f;
        for (int j = 0; j < yRes; j++) {
            for (int i = 0; i < xRes; i++) {
                int index = xRes*j + i;
                auto diff = targetCol[index] - denoisedCol[index];
                auto tempMSE = diff.dot(diff);
                tempMSE = tempMSE==tempMSE ? tempMSE : 0.0f;
                MSE += tempMSE;
            }
        }
        return MSE;
    }

    void Denoiser::saveTargetCol() {
        for (int j = 0; j < yRes; j++) {
            for (int i = 0; i < xRes; i++) {
                int index = xRes*j + i;
                targetCol[index] = preScreen[index]/sampleCount;
            }
        }
    }

    
struct skepu_userfunction_skepu_skel_0convol_SkePUFilter
{
constexpr static size_t totalArity = 1;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
constexpr static bool usesPRNG = 0;
constexpr static size_t randomCount = SKEPU_NO_RANDOM;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = FilterVals;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ FilterVals CU(skepu::Region2D<GPUInf> r)
{

        // sum up total weights and sum of individual weights times col of pixel
        float wSum[3] = {0,0,0};
        float wcSum = 0;

        // Center Pixel Specific

            // Standard Deviations
            float colStdDev =         r(0,0).stdDevs[0] + 0.000001f;
            float normalStdDev =      r(0,0).stdDevs[1] + 0.000001f;
            float albedo1StdDev =     r(0,0).stdDevs[2] + 0.000001f;
            float albedo2StdDev =     r(0,0).stdDevs[3] + 0.000001f;
            float worldPosStdDev =    r(0,0).stdDevs[4] + 0.000001f;
            float directLightStdDev = r(0,0).stdDevs[5] + 0.000001f;

            // Variances
            float indexVariance =       r(0,0).variances[0];
            float colVariance =         r(0,0).variances[1];
            float normalVariance =      r(0,0).variances[2];
            float albedo1Variance =     r(0,0).variances[3];
            float albedo2Variance =     r(0,0).variances[4];
            float worldPosVariance =    r(0,0).variances[5];
            float directLightVariance = r(0,0).variances[6];


            float pCol[3] =      {r(0,0).col[0],      r(0,0).col[1],      r(0,0).col[2]};
            float pNormal[3] =   {r(0,0).normal[0],   r(0,0).normal[1],   r(0,0).normal[2]};
            float pAlbedo1[3] =  {r(0,0).albedo1[0],  r(0,0).albedo1[1],  r(0,0).albedo1[2]};
            float pAlbedo2[3] =  {r(0,0).albedo2[0],  r(0,0).albedo2[1],  r(0,0).albedo2[2]};
            float pWorldPos[3] = {r(0,0).worldPos[0], r(0,0).worldPos[1], r(0,0).worldPos[2]};
            float pDirectLight =  r(0,0).directLight;  

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

        // ij Pixel Specific 

        // Weight Sum
        for (int j = -r.oj; j <= r.oj; j++) {
            for (int i = -r.oi; i <= r.oi; i++) {

                ijCol[0] =      r(j, i).col[0];      ijCol[1] =      r(j, i).col[1];      ijCol[2] =      r(j, i).col[2];     
                ijNormal[0] =   r(j, i).normal[0];   ijNormal[1] =   r(j, i).normal[1];   ijNormal[2] =   r(j, i).normal[2];  
                ijAlbedo1[0] =  r(j, i).albedo1[0];  ijAlbedo1[1] =  r(j, i).albedo1[1];  ijAlbedo1[2] =  r(j, i).albedo1[2]; 
                ijAlbedo2[0] =  r(j, i).albedo2[0];  ijAlbedo2[1] =  r(j, i).albedo2[1];  ijAlbedo2[2] =  r(j, i).albedo2[2]; 
                ijWorldPos[0] = r(j, i).worldPos[0]; ijWorldPos[1] = r(j, i).worldPos[1]; ijWorldPos[2] = r(j, i).worldPos[2];
                ijDirectLight = r(j, i).directLight;

                // INDEX
                indexVal = (j*j + i*i)/(2.0f * indexVariance);  
                indexVal = exp(-indexVal);
                // COLOUR
                colVal = (pow(ijCol[0]-pCol[0],2) + pow(ijCol[1]-pCol[1],2) + pow(ijCol[2]-pCol[2],2))/(colStdDev * 2.0f * colVariance); 
                colVal = exp(-colVal);
                // NORMAL
                normalVal = (pow(ijNormal[0]-pNormal[0],2) + pow(ijNormal[1]-pNormal[1],2) + pow(ijNormal[2]-pNormal[2],2))/(normalStdDev * 2.0f * normalVariance); 
                normalVal = exp(-normalVal);
                // ALBEDO1
                albedo1Val = (pow(ijAlbedo1[0]-pAlbedo1[0],2) + pow(ijAlbedo1[1]-pAlbedo1[1],2) + pow(ijAlbedo1[2]-pAlbedo1[2],2))/(albedo1StdDev * 2.0f * albedo1Variance); 
                albedo1Val = exp(-albedo1Val);
                // ALBEDO2
                albedo2Val = (pow(ijAlbedo2[0]-pAlbedo2[0],2) + pow(ijAlbedo2[1]-pAlbedo2[1],2) + pow(ijAlbedo2[2]-pAlbedo2[2],2))/(albedo2StdDev * 2.0f * albedo2Variance); 
                albedo2Val = exp(-albedo2Val);
                // WORLD POS
                worldPosVal = (pow(ijWorldPos[0]-pWorldPos[0],2) + pow(ijWorldPos[1]-pWorldPos[1],2) + pow(ijWorldPos[2]-pWorldPos[2],2))/(worldPosStdDev * 2.0f * worldPosVariance); 
                worldPosVal = exp(-worldPosVal);
                //DIRECT LIGHT
                directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance); 
                directLightVal = exp(-directLightVal);

                weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                wcSum += weight;

                wSum[0] += r(j, i).col[0]*weight;
                wSum[1] += r(j, i).col[1]*weight;
                wSum[2] += r(j, i).col[2]*weight;
            }
        }

        FilterVals ret;
        ret.x = wSum[0] / wcSum;
        ret.y = wSum[1] / wcSum;
        ret.z = wSum[2] / wcSum;
        ret.wcSum = wcSum;
        return ret;
   
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE FilterVals OMP(skepu::Region2D<GPUInf> r)
{

        // sum up total weights and sum of individual weights times col of pixel
        float wSum[3] = {0,0,0};
        float wcSum = 0;

        // Center Pixel Specific

            // Standard Deviations
            float colStdDev =         r(0,0).stdDevs[0] + 0.000001f;
            float normalStdDev =      r(0,0).stdDevs[1] + 0.000001f;
            float albedo1StdDev =     r(0,0).stdDevs[2] + 0.000001f;
            float albedo2StdDev =     r(0,0).stdDevs[3] + 0.000001f;
            float worldPosStdDev =    r(0,0).stdDevs[4] + 0.000001f;
            float directLightStdDev = r(0,0).stdDevs[5] + 0.000001f;

            // Variances
            float indexVariance =       r(0,0).variances[0];
            float colVariance =         r(0,0).variances[1];
            float normalVariance =      r(0,0).variances[2];
            float albedo1Variance =     r(0,0).variances[3];
            float albedo2Variance =     r(0,0).variances[4];
            float worldPosVariance =    r(0,0).variances[5];
            float directLightVariance = r(0,0).variances[6];


            float pCol[3] =      {r(0,0).col[0],      r(0,0).col[1],      r(0,0).col[2]};
            float pNormal[3] =   {r(0,0).normal[0],   r(0,0).normal[1],   r(0,0).normal[2]};
            float pAlbedo1[3] =  {r(0,0).albedo1[0],  r(0,0).albedo1[1],  r(0,0).albedo1[2]};
            float pAlbedo2[3] =  {r(0,0).albedo2[0],  r(0,0).albedo2[1],  r(0,0).albedo2[2]};
            float pWorldPos[3] = {r(0,0).worldPos[0], r(0,0).worldPos[1], r(0,0).worldPos[2]};
            float pDirectLight =  r(0,0).directLight;  

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

        // ij Pixel Specific 

        // Weight Sum
        for (int j = -r.oj; j <= r.oj; j++) {
            for (int i = -r.oi; i <= r.oi; i++) {

                ijCol[0] =      r(j, i).col[0];      ijCol[1] =      r(j, i).col[1];      ijCol[2] =      r(j, i).col[2];     
                ijNormal[0] =   r(j, i).normal[0];   ijNormal[1] =   r(j, i).normal[1];   ijNormal[2] =   r(j, i).normal[2];  
                ijAlbedo1[0] =  r(j, i).albedo1[0];  ijAlbedo1[1] =  r(j, i).albedo1[1];  ijAlbedo1[2] =  r(j, i).albedo1[2]; 
                ijAlbedo2[0] =  r(j, i).albedo2[0];  ijAlbedo2[1] =  r(j, i).albedo2[1];  ijAlbedo2[2] =  r(j, i).albedo2[2]; 
                ijWorldPos[0] = r(j, i).worldPos[0]; ijWorldPos[1] = r(j, i).worldPos[1]; ijWorldPos[2] = r(j, i).worldPos[2];
                ijDirectLight = r(j, i).directLight;

                // INDEX
                indexVal = (j*j + i*i)/(2.0f * indexVariance);  
                indexVal = exp(-indexVal);
                // COLOUR
                colVal = (pow(ijCol[0]-pCol[0],2) + pow(ijCol[1]-pCol[1],2) + pow(ijCol[2]-pCol[2],2))/(colStdDev * 2.0f * colVariance); 
                colVal = exp(-colVal);
                // NORMAL
                normalVal = (pow(ijNormal[0]-pNormal[0],2) + pow(ijNormal[1]-pNormal[1],2) + pow(ijNormal[2]-pNormal[2],2))/(normalStdDev * 2.0f * normalVariance); 
                normalVal = exp(-normalVal);
                // ALBEDO1
                albedo1Val = (pow(ijAlbedo1[0]-pAlbedo1[0],2) + pow(ijAlbedo1[1]-pAlbedo1[1],2) + pow(ijAlbedo1[2]-pAlbedo1[2],2))/(albedo1StdDev * 2.0f * albedo1Variance); 
                albedo1Val = exp(-albedo1Val);
                // ALBEDO2
                albedo2Val = (pow(ijAlbedo2[0]-pAlbedo2[0],2) + pow(ijAlbedo2[1]-pAlbedo2[1],2) + pow(ijAlbedo2[2]-pAlbedo2[2],2))/(albedo2StdDev * 2.0f * albedo2Variance); 
                albedo2Val = exp(-albedo2Val);
                // WORLD POS
                worldPosVal = (pow(ijWorldPos[0]-pWorldPos[0],2) + pow(ijWorldPos[1]-pWorldPos[1],2) + pow(ijWorldPos[2]-pWorldPos[2],2))/(worldPosStdDev * 2.0f * worldPosVariance); 
                worldPosVal = exp(-worldPosVal);
                //DIRECT LIGHT
                directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance); 
                directLightVal = exp(-directLightVal);

                weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                wcSum += weight;

                wSum[0] += r(j, i).col[0]*weight;
                wSum[1] += r(j, i).col[1]*weight;
                wSum[2] += r(j, i).col[2]*weight;
            }
        }

        FilterVals ret;
        ret.x = wSum[0] / wcSum;
        ret.y = wSum[1] / wcSum;
        ret.z = wSum[2] / wcSum;
        ret.wcSum = wcSum;
        return ret;
   
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE FilterVals CPU(skepu::Region2D<GPUInf> r)
{

        // sum up total weights and sum of individual weights times col of pixel
        float wSum[3] = {0,0,0};
        float wcSum = 0;

        // Center Pixel Specific

            // Standard Deviations
            float colStdDev =         r(0,0).stdDevs[0] + 0.000001f;
            float normalStdDev =      r(0,0).stdDevs[1] + 0.000001f;
            float albedo1StdDev =     r(0,0).stdDevs[2] + 0.000001f;
            float albedo2StdDev =     r(0,0).stdDevs[3] + 0.000001f;
            float worldPosStdDev =    r(0,0).stdDevs[4] + 0.000001f;
            float directLightStdDev = r(0,0).stdDevs[5] + 0.000001f;

            // Variances
            float indexVariance =       r(0,0).variances[0];
            float colVariance =         r(0,0).variances[1];
            float normalVariance =      r(0,0).variances[2];
            float albedo1Variance =     r(0,0).variances[3];
            float albedo2Variance =     r(0,0).variances[4];
            float worldPosVariance =    r(0,0).variances[5];
            float directLightVariance = r(0,0).variances[6];


            float pCol[3] =      {r(0,0).col[0],      r(0,0).col[1],      r(0,0).col[2]};
            float pNormal[3] =   {r(0,0).normal[0],   r(0,0).normal[1],   r(0,0).normal[2]};
            float pAlbedo1[3] =  {r(0,0).albedo1[0],  r(0,0).albedo1[1],  r(0,0).albedo1[2]};
            float pAlbedo2[3] =  {r(0,0).albedo2[0],  r(0,0).albedo2[1],  r(0,0).albedo2[2]};
            float pWorldPos[3] = {r(0,0).worldPos[0], r(0,0).worldPos[1], r(0,0).worldPos[2]};
            float pDirectLight =  r(0,0).directLight;  

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

        // ij Pixel Specific 

        // Weight Sum
        for (int j = -r.oj; j <= r.oj; j++) {
            for (int i = -r.oi; i <= r.oi; i++) {

                ijCol[0] =      r(j, i).col[0];      ijCol[1] =      r(j, i).col[1];      ijCol[2] =      r(j, i).col[2];     
                ijNormal[0] =   r(j, i).normal[0];   ijNormal[1] =   r(j, i).normal[1];   ijNormal[2] =   r(j, i).normal[2];  
                ijAlbedo1[0] =  r(j, i).albedo1[0];  ijAlbedo1[1] =  r(j, i).albedo1[1];  ijAlbedo1[2] =  r(j, i).albedo1[2]; 
                ijAlbedo2[0] =  r(j, i).albedo2[0];  ijAlbedo2[1] =  r(j, i).albedo2[1];  ijAlbedo2[2] =  r(j, i).albedo2[2]; 
                ijWorldPos[0] = r(j, i).worldPos[0]; ijWorldPos[1] = r(j, i).worldPos[1]; ijWorldPos[2] = r(j, i).worldPos[2];
                ijDirectLight = r(j, i).directLight;

                // INDEX
                indexVal = (j*j + i*i)/(2.0f * indexVariance);  
                indexVal = exp(-indexVal);
                // COLOUR
                colVal = (pow(ijCol[0]-pCol[0],2) + pow(ijCol[1]-pCol[1],2) + pow(ijCol[2]-pCol[2],2))/(colStdDev * 2.0f * colVariance); 
                colVal = exp(-colVal);
                // NORMAL
                normalVal = (pow(ijNormal[0]-pNormal[0],2) + pow(ijNormal[1]-pNormal[1],2) + pow(ijNormal[2]-pNormal[2],2))/(normalStdDev * 2.0f * normalVariance); 
                normalVal = exp(-normalVal);
                // ALBEDO1
                albedo1Val = (pow(ijAlbedo1[0]-pAlbedo1[0],2) + pow(ijAlbedo1[1]-pAlbedo1[1],2) + pow(ijAlbedo1[2]-pAlbedo1[2],2))/(albedo1StdDev * 2.0f * albedo1Variance); 
                albedo1Val = exp(-albedo1Val);
                // ALBEDO2
                albedo2Val = (pow(ijAlbedo2[0]-pAlbedo2[0],2) + pow(ijAlbedo2[1]-pAlbedo2[1],2) + pow(ijAlbedo2[2]-pAlbedo2[2],2))/(albedo2StdDev * 2.0f * albedo2Variance); 
                albedo2Val = exp(-albedo2Val);
                // WORLD POS
                worldPosVal = (pow(ijWorldPos[0]-pWorldPos[0],2) + pow(ijWorldPos[1]-pWorldPos[1],2) + pow(ijWorldPos[2]-pWorldPos[2],2))/(worldPosStdDev * 2.0f * worldPosVariance); 
                worldPosVal = exp(-worldPosVal);
                //DIRECT LIGHT
                directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance); 
                directLightVal = exp(-directLightVal);

                weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                wcSum += weight;

                wSum[0] += r(j, i).col[0]*weight;
                wSum[1] += r(j, i).col[1]*weight;
                wSum[2] += r(j, i).col[2]*weight;
            }
        }

        FilterVals ret;
        ret.x = wSum[0] / wcSum;
        ret.y = wSum[1] / wcSum;
        ret.z = wSum[2] / wcSum;
        ret.wcSum = wcSum;
        return ret;
   
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "SkePUDenoiser_Overlap2DKernel_SkePUFilter.cu"
void Denoiser::SkePUDenoise() {

        // 1. Load saved NN weights 
        // 2. Calculate standard deviations of primary features (sqrt of sum of differences to mean)
        // 3. Loop through all pixels in Matrix of primary features (i.e. each element of matrix is struct with multiple primary features)
        // 4. Get primary feature weights from NN output at current pixel 
        // 5. Apply cross-bilateral filter to pixel, taking into account surrounding pixels

        // ----- 5 ----- (with manual variance values)

        // Create SkePU Matrix of GPUInf
        auto mat = skepu::Matrix<GPUInf>(yRes, xRes);
        int v, sd, c;
        for (int ind = 0; ind < xRes*yRes; ind++) {

            // Final Col
            mat[ind].col[0] = preScreen[ind].x / sampleCount;
            mat[ind].col[1] = preScreen[ind].y / sampleCount;
            mat[ind].col[2] = preScreen[ind].z / sampleCount; 
            // Normal
            mat[ind].normal[0] = normal[ind].x / sampleCount;
            mat[ind].normal[1] = normal[ind].y / sampleCount;
            mat[ind].normal[2] = normal[ind].z / sampleCount;
            // Albedo1
            mat[ind].albedo1[0] = albedo1[ind].x / sampleCount;
            mat[ind].albedo1[1] = albedo1[ind].y / sampleCount;
            mat[ind].albedo1[2] = albedo1[ind].z / sampleCount;
            // Albedo2
            mat[ind].albedo2[0] = albedo2[ind].x / sampleCount;
            mat[ind].albedo2[1] = albedo2[ind].y / sampleCount;
            mat[ind].albedo2[2] = albedo2[ind].z / sampleCount;
            // World Pos
            mat[ind].worldPos[0] = worldPos[ind].x / sampleCount;
            mat[ind].worldPos[1] = worldPos[ind].y / sampleCount;
            mat[ind].worldPos[2] = worldPos[ind].z / sampleCount;
            // Direct Light
            mat[ind].directLight = directLight[ind].x / sampleCount;
            // Std Devs
            for (sd=0; sd < 6; sd++) {
                mat[ind].stdDevs[sd] = 0.0f;
                for (c=0; c<3; c++) 
					mat[ind].stdDevs[sd] += sqrt(denoisingInf[ind].stdDevVecs[sd][c]/sampleCount);
            }
            // Variances
            for (v=0; v < 7; v++)
                mat[ind].variances[v] = denoisingInf[ind].variances[v];
        }

        auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(skepuBackend)};
	    spec.activateBackend();
        skepu::Matrix<FilterVals> result(yRes, xRes);
        skepu::backend::MapOverlap2D<skepu_userfunction_skepu_skel_0convol_SkePUFilter, decltype(&SkePUDenoiser_Overlap2DKernel_SkePUFilter_conv_cuda_2D_kernel), void> convol(SkePUDenoiser_Overlap2DKernel_SkePUFilter_conv_cuda_2D_kernel);
	    convol.setBackend(spec);
        convol.setOverlap(denoisingN);
        convol.setEdgeMode(skepu::Edge::Duplicate);

        convol(result, mat);
        FilterVals res;
        for (int ind = 0; ind < xRes*yRes; ind++) {
            res = result[ind];
            denoisedCol[ind] = vec3(res.x, res.y, res.z);
            denoisingInf[ind].wcSum = res.wcSum;
        }
    }

    void Denoiser::CPUDenoise(){

    }
    void Denoiser::OMPDenoise(){
        int tH = yRes;
        int tW = xRes;
        int N = denoisingN;

        //////////////////// Denoising Alg ////////////////////
        #pragma omp parallel for
        for (int jMain = 0; jMain < tH; jMain++) {
            for (int iMain = 0; iMain < tW; iMain++) {

                int index = jMain*tW + iMain;

                float wcSum = 0.0f;
                auto wSum = vec3(0,0,0);

                // Values
                vec3 pCol =          preScreen[index] / sampleCount;
                vec3 pNormal =       normal[index] / sampleCount;
                vec3 pAlbedo1 =      albedo1[index] / sampleCount;
                vec3 pAlbedo2 =      albedo2[index] / sampleCount;
                vec3 pWorldPos =     worldPos[index] / sampleCount;
                float pDirectLight = directLight[index].x / sampleCount;

                // Standard Deviations
                int c;
                float colStdDev = 0.000001f;
                float normalStdDev = 0.000001f;
                float albedo1StdDev = 0.000001f;
                float albedo2StdDev = 0.000001f;
                float worldPosStdDev = 0.000001f;
                float directLightStdDev = 0.000001f;
                for (c=0; c<3; c++) {
				    colStdDev         += sqrt(denoisingInf[index].stdDevVecs[0][c]/sampleCount);
				    normalStdDev      += sqrt(denoisingInf[index].stdDevVecs[1][c]/sampleCount);
				    albedo1StdDev     += sqrt(denoisingInf[index].stdDevVecs[2][c]/sampleCount);
				    albedo2StdDev     += sqrt(denoisingInf[index].stdDevVecs[3][c]/sampleCount);
				    worldPosStdDev    += sqrt(denoisingInf[index].stdDevVecs[4][c]/sampleCount);
                }
                directLightStdDev += sqrt(denoisingInf[index].stdDevVecs[5][0]/sampleCount);


                // Variances
                float indexVariance =       denoisingInf[index].variances[0];
                float colVariance =         denoisingInf[index].variances[1];
                float normalVariance =      denoisingInf[index].variances[2];
                float albedo1Variance =     denoisingInf[index].variances[3];
                float albedo2Variance =     denoisingInf[index].variances[4];
                float worldPosVariance =    denoisingInf[index].variances[5];
                float directLightVariance = denoisingInf[index].variances[6];

			    #pragma omp parallel for
                for (int j = -N; j <= N; j++) {
                    int jFixed = jMain + j < 0 ? 0 : (jMain + j >= tH ? tH-1 : jMain + j);
                    for (int i = -N; i <= N; i++) {

                        int iFixed = iMain + i < 0 ? 0 : (iMain + i >= tW ? tW-1 : iMain + i);
                        int ijIndex = jFixed*tW + iFixed;

                        vec3 ijCol = preScreen[index] / sampleCount;
                        vec3 ijNormal = normal[index] / sampleCount;
                        vec3 ijAlbedo1 = albedo1[index] / sampleCount;
                        vec3 ijAlbedo2 = albedo2[index] / sampleCount;
                        vec3 ijWorldPos = worldPos[index] / sampleCount;
                        float ijDirectLight = directLight[index].x / sampleCount;

                        // INDEX
                        float indexVal = (pow(j,2) + pow(i,2))/(2.0f * indexVariance); 
                        indexVal = exp(-indexVal);

                        // COLOUR
                        float colVal = (pow(ijCol.x-pCol.x,2) + pow(ijCol.y-pCol.y,2) + pow(ijCol.z-pCol.z,2))/(colStdDev * 2.0f * colVariance); 
                        colVal = exp(-colVal);

                        // NORMAL
                        float normalVal = (pow(ijNormal.x-pNormal.x,2) + pow(ijNormal.y-pNormal.y,2) + pow(ijNormal.z-pNormal.z,2))/(normalStdDev * 2.0f * normalVariance); 
                        normalVal = exp(-normalVal);

                        // ALBEDO1
                        float albedo1Val = (pow(ijAlbedo1.x-pAlbedo1.x,2) + pow(ijAlbedo1.y-pAlbedo1.y,2) + pow(ijAlbedo1.z-pAlbedo1.z,2))/(albedo1StdDev * 2.0f * albedo1Variance); 
                        albedo1Val = exp(-albedo1Val);

                        // ALBEDO2
                        float albedo2Val = (pow(ijAlbedo2.x-pAlbedo2.x,2) + pow(ijAlbedo2.y-pAlbedo2.y,2) + pow(ijAlbedo2.z-pAlbedo2.z,2))/(albedo2StdDev * 2.0f * albedo2Variance); 
                        albedo2Val = exp(-albedo2Val);

                        // WORDLD POS
                        float worldPosVal = (pow(ijWorldPos.x-pWorldPos.x,2) + pow(ijWorldPos.y-pWorldPos.y,2) + pow(ijWorldPos.z-pWorldPos.z,2))/(worldPosStdDev * 2.0f * worldPosVariance); 
                        worldPosVal = exp(-worldPosVal);

                        // DIRECT LIGHT
                        float directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance); 
                        directLightVal = exp(-directLightVal);

                        float weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                        wSum += preScreen[ijIndex]*weight/sampleCount;
                        wcSum += weight;
                    }
                }

                denoisedCol[index] = wSum / wcSum;
                denoisingInf[index].wcSum = wcSum;
            }
        }   
    }
    void Denoiser::CUDADenoise(){

    }
    void Denoiser::OpenGLDenoise(){

    }
