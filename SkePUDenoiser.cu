#define SKEPU_PRECOMPILED 1
#define SKEPU_OPENMP 1
#define SKEPU_CUDA 1
#include "Denoiser.h"

    void Denoiser::denoise() {

        auto denoiseTimer = clock_::now();

        denoiserNN.ForwardProp();

        switch(denoisingBackend) {
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
                if (skipCudaDenoise)
                    denoisingSkePUBackend = "openmp";
                SkePUDenoise();
                if (skipCudaDenoise)
                    denoisingSkePUBackend = "cuda";
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
            float col1StdDev =        r(0,0).stdDevs[0];
            float normalStdDev =      r(0,0).stdDevs[1];
            float albedo1StdDev =     r(0,0).stdDevs[2];
            float albedo2StdDev =     r(0,0).stdDevs[3];
            float worldPosStdDev =    r(0,0).stdDevs[4];
            float directLightStdDev = r(0,0).stdDevs[5];

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
            float col2StdDev     = 1.0f;

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

                col2StdDev = r(j, i).stdDevs[0] + 0.000001f;

                // INDEX
                indexVal = (j*j + i*i)/(2.0f * indexVariance);  
                indexVal = exp(-indexVal);
                // COLOUR
                colVal = (pow(ijCol[0]-pCol[0],2) + pow(ijCol[1]-pCol[1],2) + pow(ijCol[2]-pCol[2],2))/((col1StdDev + col2StdDev) * 2.0f * colVariance + 0.000001f); 
                colVal = exp(-colVal);
                // NORMAL
                normalVal = (pow(ijNormal[0]-pNormal[0],2) + pow(ijNormal[1]-pNormal[1],2) + pow(ijNormal[2]-pNormal[2],2))/(normalStdDev * 2.0f * normalVariance + 0.000001f); 
                normalVal = exp(-normalVal);
                // ALBEDO1
                albedo1Val = (pow(ijAlbedo1[0]-pAlbedo1[0],2) + pow(ijAlbedo1[1]-pAlbedo1[1],2) + pow(ijAlbedo1[2]-pAlbedo1[2],2))/(albedo1StdDev * 2.0f * albedo1Variance + 0.000001f); 
                albedo1Val = exp(-albedo1Val);
                // ALBEDO2
                albedo2Val = (pow(ijAlbedo2[0]-pAlbedo2[0],2) + pow(ijAlbedo2[1]-pAlbedo2[1],2) + pow(ijAlbedo2[2]-pAlbedo2[2],2))/(albedo2StdDev * 2.0f * albedo2Variance + 0.000001f); 
                albedo2Val = exp(-albedo2Val);
                // WORLD POS
                worldPosVal = (pow(ijWorldPos[0]-pWorldPos[0],2) + pow(ijWorldPos[1]-pWorldPos[1],2) + pow(ijWorldPos[2]-pWorldPos[2],2))/( 2.0f * worldPosStdDev *worldPosVariance + 0.000001f); 
                worldPosVal = exp(-worldPosVal);
                //DIRECT LIGHT
                directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance + 0.000001f); 
                directLightVal = exp(-directLightVal);

                weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                wcSum += weight;

                wSum[0] += ijCol[0]*weight;
                wSum[1] += ijCol[1]*weight;
                wSum[2] += ijCol[2]*weight;
            }
        }

        FilterVals ret;
        ret.x = wSum[0] / wcSum;
        ret.y = wSum[1] / wcSum;
        ret.z = wSum[2] / wcSum;
        ret.wcSum = wcSum;
        return ret;
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
            float col1StdDev =        r(0,0).stdDevs[0];
            float normalStdDev =      r(0,0).stdDevs[1];
            float albedo1StdDev =     r(0,0).stdDevs[2];
            float albedo2StdDev =     r(0,0).stdDevs[3];
            float worldPosStdDev =    r(0,0).stdDevs[4];
            float directLightStdDev = r(0,0).stdDevs[5];

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
            float col2StdDev     = 1.0f;

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

                col2StdDev = r(j, i).stdDevs[0] + 0.000001f;

                // INDEX
                indexVal = (j*j + i*i)/(2.0f * indexVariance);  
                indexVal = exp(-indexVal);
                // COLOUR
                colVal = (pow(ijCol[0]-pCol[0],2) + pow(ijCol[1]-pCol[1],2) + pow(ijCol[2]-pCol[2],2))/((col1StdDev + col2StdDev) * 2.0f * colVariance + 0.000001f); 
                colVal = exp(-colVal);
                // NORMAL
                normalVal = (pow(ijNormal[0]-pNormal[0],2) + pow(ijNormal[1]-pNormal[1],2) + pow(ijNormal[2]-pNormal[2],2))/(normalStdDev * 2.0f * normalVariance + 0.000001f); 
                normalVal = exp(-normalVal);
                // ALBEDO1
                albedo1Val = (pow(ijAlbedo1[0]-pAlbedo1[0],2) + pow(ijAlbedo1[1]-pAlbedo1[1],2) + pow(ijAlbedo1[2]-pAlbedo1[2],2))/(albedo1StdDev * 2.0f * albedo1Variance + 0.000001f); 
                albedo1Val = exp(-albedo1Val);
                // ALBEDO2
                albedo2Val = (pow(ijAlbedo2[0]-pAlbedo2[0],2) + pow(ijAlbedo2[1]-pAlbedo2[1],2) + pow(ijAlbedo2[2]-pAlbedo2[2],2))/(albedo2StdDev * 2.0f * albedo2Variance + 0.000001f); 
                albedo2Val = exp(-albedo2Val);
                // WORLD POS
                worldPosVal = (pow(ijWorldPos[0]-pWorldPos[0],2) + pow(ijWorldPos[1]-pWorldPos[1],2) + pow(ijWorldPos[2]-pWorldPos[2],2))/( 2.0f * worldPosStdDev *worldPosVariance + 0.000001f); 
                worldPosVal = exp(-worldPosVal);
                //DIRECT LIGHT
                directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance + 0.000001f); 
                directLightVal = exp(-directLightVal);

                weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                wcSum += weight;

                wSum[0] += ijCol[0]*weight;
                wSum[1] += ijCol[1]*weight;
                wSum[2] += ijCol[2]*weight;
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
            float col1StdDev =        r(0,0).stdDevs[0];
            float normalStdDev =      r(0,0).stdDevs[1];
            float albedo1StdDev =     r(0,0).stdDevs[2];
            float albedo2StdDev =     r(0,0).stdDevs[3];
            float worldPosStdDev =    r(0,0).stdDevs[4];
            float directLightStdDev = r(0,0).stdDevs[5];

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
            float col2StdDev     = 1.0f;

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

                col2StdDev = r(j, i).stdDevs[0] + 0.000001f;

                // INDEX
                indexVal = (j*j + i*i)/(2.0f * indexVariance);  
                indexVal = exp(-indexVal);
                // COLOUR
                colVal = (pow(ijCol[0]-pCol[0],2) + pow(ijCol[1]-pCol[1],2) + pow(ijCol[2]-pCol[2],2))/((col1StdDev + col2StdDev) * 2.0f * colVariance + 0.000001f); 
                colVal = exp(-colVal);
                // NORMAL
                normalVal = (pow(ijNormal[0]-pNormal[0],2) + pow(ijNormal[1]-pNormal[1],2) + pow(ijNormal[2]-pNormal[2],2))/(normalStdDev * 2.0f * normalVariance + 0.000001f); 
                normalVal = exp(-normalVal);
                // ALBEDO1
                albedo1Val = (pow(ijAlbedo1[0]-pAlbedo1[0],2) + pow(ijAlbedo1[1]-pAlbedo1[1],2) + pow(ijAlbedo1[2]-pAlbedo1[2],2))/(albedo1StdDev * 2.0f * albedo1Variance + 0.000001f); 
                albedo1Val = exp(-albedo1Val);
                // ALBEDO2
                albedo2Val = (pow(ijAlbedo2[0]-pAlbedo2[0],2) + pow(ijAlbedo2[1]-pAlbedo2[1],2) + pow(ijAlbedo2[2]-pAlbedo2[2],2))/(albedo2StdDev * 2.0f * albedo2Variance + 0.000001f); 
                albedo2Val = exp(-albedo2Val);
                // WORLD POS
                worldPosVal = (pow(ijWorldPos[0]-pWorldPos[0],2) + pow(ijWorldPos[1]-pWorldPos[1],2) + pow(ijWorldPos[2]-pWorldPos[2],2))/( 2.0f * worldPosStdDev *worldPosVariance + 0.000001f); 
                worldPosVal = exp(-worldPosVal);
                //DIRECT LIGHT
                directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance + 0.000001f); 
                directLightVal = exp(-directLightVal);

                weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                wcSum += weight;

                wSum[0] += ijCol[0]*weight;
                wSum[1] += ijCol[1]*weight;
                wSum[2] += ijCol[2]*weight;
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
            float col1StdDev =        r(0,0).stdDevs[0];
            float normalStdDev =      r(0,0).stdDevs[1];
            float albedo1StdDev =     r(0,0).stdDevs[2];
            float albedo2StdDev =     r(0,0).stdDevs[3];
            float worldPosStdDev =    r(0,0).stdDevs[4];
            float directLightStdDev = r(0,0).stdDevs[5];

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
            float col2StdDev     = 1.0f;

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

                col2StdDev = r(j, i).stdDevs[0] + 0.000001f;

                // INDEX
                indexVal = (j*j + i*i)/(2.0f * indexVariance);  
                indexVal = exp(-indexVal);
                // COLOUR
                colVal = (pow(ijCol[0]-pCol[0],2) + pow(ijCol[1]-pCol[1],2) + pow(ijCol[2]-pCol[2],2))/((col1StdDev + col2StdDev) * 2.0f * colVariance + 0.000001f); 
                colVal = exp(-colVal);
                // NORMAL
                normalVal = (pow(ijNormal[0]-pNormal[0],2) + pow(ijNormal[1]-pNormal[1],2) + pow(ijNormal[2]-pNormal[2],2))/(normalStdDev * 2.0f * normalVariance + 0.000001f); 
                normalVal = exp(-normalVal);
                // ALBEDO1
                albedo1Val = (pow(ijAlbedo1[0]-pAlbedo1[0],2) + pow(ijAlbedo1[1]-pAlbedo1[1],2) + pow(ijAlbedo1[2]-pAlbedo1[2],2))/(albedo1StdDev * 2.0f * albedo1Variance + 0.000001f); 
                albedo1Val = exp(-albedo1Val);
                // ALBEDO2
                albedo2Val = (pow(ijAlbedo2[0]-pAlbedo2[0],2) + pow(ijAlbedo2[1]-pAlbedo2[1],2) + pow(ijAlbedo2[2]-pAlbedo2[2],2))/(albedo2StdDev * 2.0f * albedo2Variance + 0.000001f); 
                albedo2Val = exp(-albedo2Val);
                // WORLD POS
                worldPosVal = (pow(ijWorldPos[0]-pWorldPos[0],2) + pow(ijWorldPos[1]-pWorldPos[1],2) + pow(ijWorldPos[2]-pWorldPos[2],2))/( 2.0f * worldPosStdDev *worldPosVariance + 0.000001f); 
                worldPosVal = exp(-worldPosVal);
                //DIRECT LIGHT
                directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance + 0.000001f); 
                directLightVal = exp(-directLightVal);

                weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                wcSum += weight;

                wSum[0] += ijCol[0]*weight;
                wSum[1] += ijCol[1]*weight;
                wSum[2] += ijCol[2]*weight;
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


        // Create SkePU Matrix of GPUInf
        auto mat = skepu::Matrix<GPUInf>(yRes, xRes);
        int v, sd, c;
        for (int ind = 0; ind < xRes*yRes; ind++) {

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

            mat[ind] = tempInf;
        }

        auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(denoisingSkePUBackend)};
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
        int N = denoisingN;

        int index, c, i, j, iFixed, jFixed, ijIndex;
        float wcSum;
        vec3 wSum;

        vec3 pCol;
        vec3 pNormal;
        vec3 pAlbedo1 ;
        vec3 pAlbedo2 ;
        vec3 pWorldPos;
        float pDirectLight;

        float col1StdDev, col2StdDev;
        float normalStdDev;
        float albedo1StdDev ;
        float albedo2StdDev ;
        float worldPosStdDev;
        float directLightStdDev;

        float indexVariance   ;
        float colVariance ;
        float normalVariance;
        float albedo1Variance  ;
        float albedo2Variance  ;
        float worldPosVariance ;
        float directLightVariance;

        vec3 ijCol;
        vec3 ijNormal ;
        vec3 ijAlbedo1;
        vec3 ijAlbedo2;
        vec3 ijWorldPos ;
        float ijDirectLight;

        float indexVal;
        float colVal;
        float normalVal;
        float albedo1Val;
        float albedo2Val;
        float worldPosVal;
        float directLightVal;
        float weight;

        //////////////////// Denoising Alg ////////////////////
        for (int jMain = 0; jMain < yRes; jMain++) {
            for (int iMain = 0; iMain < xRes; iMain++) {

                index = jMain*xRes + iMain;

                wcSum = 0.0f;
                wSum = vec3();

                // Values
                pCol =          preScreen[index]     / sampleCount;
                pNormal =       normal[index]        / sampleCount;
                pAlbedo1 =      albedo1[index]       / sampleCount;
                pAlbedo2 =      albedo2[index]       / sampleCount;
                pWorldPos =     worldPos[index]      / sampleCount;
                pDirectLight =  directLight[index].x / sampleCount;

                // Standard Deviations
			    col1StdDev        = denoisingInf[index].stdDev[0];
			    normalStdDev      = denoisingInf[index].stdDev[1];
			    albedo1StdDev     = denoisingInf[index].stdDev[2];
			    albedo2StdDev     = denoisingInf[index].stdDev[3];
			    worldPosStdDev    = denoisingInf[index].stdDev[4];
                directLightStdDev = denoisingInf[index].stdDev[5];

                // Variances
                indexVariance       = denoisingInf[index].variances[0];
                colVariance         = denoisingInf[index].variances[1];
                normalVariance      = denoisingInf[index].variances[2];
                albedo1Variance     = denoisingInf[index].variances[3];
                albedo2Variance     = denoisingInf[index].variances[4];
                worldPosVariance    = denoisingInf[index].variances[5];
                directLightVariance = denoisingInf[index].variances[6];

                for (j = -N; j <= N; j++) {
                    jFixed = jMain + j < 0 ? 0 : (jMain + j >= yRes ? yRes-1 : jMain + j);
                    for (i = -N; i <= N; i++) {

                        iFixed = iMain + i < 0 ? 0 : (iMain + i >= xRes ? xRes-1 : iMain + i);
                        ijIndex = jFixed*xRes + iFixed;

                        ijCol         = preScreen[ijIndex]     / sampleCount;
                        ijNormal      = normal[ijIndex]        / sampleCount;
                        ijAlbedo1     = albedo1[ijIndex]       / sampleCount;
                        ijAlbedo2     = albedo2[ijIndex]       / sampleCount;
                        ijWorldPos    = worldPos[ijIndex]      / sampleCount;
                        ijDirectLight = directLight[ijIndex].x / sampleCount;

                        col2StdDev = denoisingInf[ijIndex].stdDev[0];

                        // INDEX
                        indexVal = (pow(j,2) + pow(i,2))/(2.0f * indexVariance); 
                        indexVal = exp(-indexVal);

                        // COLOUR
                        colVal = (pow(ijCol.x-pCol.x,2) + pow(ijCol.y-pCol.y,2) + pow(ijCol.z-pCol.z,2))/((col1StdDev + col2StdDev) * 2.0f * colVariance + 0.000001f); 
                        colVal = exp(-colVal);

                        // NORMAL
                        normalVal = (pow(ijNormal.x-pNormal.x,2) + pow(ijNormal.y-pNormal.y,2) + pow(ijNormal.z-pNormal.z,2))/(normalStdDev * 2.0f * normalVariance + 0.000001f); 
                        normalVal = exp(-normalVal);

                        // ALBEDO1
                        albedo1Val = (pow(ijAlbedo1.x-pAlbedo1.x,2) + pow(ijAlbedo1.y-pAlbedo1.y,2) + pow(ijAlbedo1.z-pAlbedo1.z,2))/(albedo1StdDev * 2.0f * albedo1Variance + 0.000001f); 
                        albedo1Val = exp(-albedo1Val);

                        // ALBEDO2
                        albedo2Val = (pow(ijAlbedo2.x-pAlbedo2.x,2) + pow(ijAlbedo2.y-pAlbedo2.y,2) + pow(ijAlbedo2.z-pAlbedo2.z,2))/(albedo2StdDev * 2.0f * albedo2Variance + 0.000001f); 
                        albedo2Val = exp(-albedo2Val);

                        // WORDLD POS
                        worldPosVal = (pow(ijWorldPos.x-pWorldPos.x,2) + pow(ijWorldPos.y-pWorldPos.y,2) + pow(ijWorldPos.z-pWorldPos.z,2))/( 2.0f * worldPosStdDev *worldPosVariance + 0.000001f);
                        worldPosVal = exp(-worldPosVal);

                        // DIRECT LIGHT
                        directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance + 0.000001f); 
                        directLightVal = exp(-directLightVal);

                        weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                        wSum += ijCol*weight;
                        wcSum += weight;
                    }
                }

                denoisedCol[index] = wSum / wcSum;
                denoisingInf[index].wcSum = wcSum;
            }
        } 
    }
    void Denoiser::OMPDenoise(){
        int N = denoisingN;

        //////////////////// Denoising Alg ////////////////////
        #pragma omp parallel for
        for (int jMain = 0; jMain < yRes; jMain++) {
            #pragma omp parallel for
            for (int iMain = 0; iMain < xRes; iMain++) {

                int index = jMain*xRes + iMain;

                float wcSum = 0.0f;
                vec3 wSum = vec3();

                // Values
                vec3 pCol =          preScreen[index] / sampleCount;
                vec3 pNormal =       normal[index] / sampleCount;
                vec3 pAlbedo1 =      albedo1[index] / sampleCount;
                vec3 pAlbedo2 =      albedo2[index] / sampleCount;
                vec3 pWorldPos =     worldPos[index] / sampleCount;
                float pDirectLight = directLight[index].x / sampleCount;

                // Standard Deviations
			    float col1StdDev        = denoisingInf[index].stdDev[0];
			    float normalStdDev      = denoisingInf[index].stdDev[1];
			    float albedo1StdDev     = denoisingInf[index].stdDev[2];
			    float albedo2StdDev     = denoisingInf[index].stdDev[3];
			    float worldPosStdDev    = denoisingInf[index].stdDev[4];
                float directLightStdDev = denoisingInf[index].stdDev[5];


                // Variances
                float indexVariance =       denoisingInf[index].variances[0];
                float colVariance =         denoisingInf[index].variances[1];
                float normalVariance =      denoisingInf[index].variances[2];
                float albedo1Variance =     denoisingInf[index].variances[3];
                float albedo2Variance =     denoisingInf[index].variances[4];
                float worldPosVariance =    denoisingInf[index].variances[5];
                float directLightVariance = denoisingInf[index].variances[6];

                vec3 ijCol;
                vec3 ijNormal;
                vec3 ijAlbedo1 ;
                vec3 ijAlbedo2 ;
                vec3 ijWorldPos;
                float ijDirectLight;

                float indexVal;
                float colVal;
                float normalVal;
                float albedo1Val;
                float albedo2Val;
                float worldPosVal;
                float directLightVal;
                float weight;
                float col2StdDev;

                int j, i, jFixed, iFixed, ijIndex;

                for (j = -N; j <= N; j++) {
                    jFixed = jMain + j < 0 ? 0 : (jMain + j >= yRes ? yRes-1 : jMain + j);
                    for (i = -N; i <= N; i++) {

                        iFixed = iMain + i < 0 ? 0 : (iMain + i >= xRes ? xRes-1 : iMain + i);
                        ijIndex = jFixed*xRes + iFixed;

                        ijCol =         preScreen[ijIndex]     / sampleCount;
                        ijNormal =      normal[ijIndex]        / sampleCount;
                        ijAlbedo1 =     albedo1[ijIndex]       / sampleCount;
                        ijAlbedo2 =     albedo2[ijIndex]       / sampleCount;
                        ijWorldPos =    worldPos[ijIndex]      / sampleCount;
                        ijDirectLight = directLight[ijIndex].x / sampleCount;

                        col2StdDev = denoisingInf[ijIndex].stdDev[0] + 0.000001f;

                        // INDEX
                        indexVal = (pow(j,2) + pow(i,2))/(2.0f * indexVariance); 
                        indexVal = exp(-indexVal);

                        // COLOUR
                        colVal = (pow(ijCol.x-pCol.x,2) + pow(ijCol.y-pCol.y,2) + pow(ijCol.z-pCol.z,2))/((col1StdDev) * 2.0f * colVariance + 0.000001f); 
                        colVal = exp(-colVal);

                        // NORMAL
                        normalVal = (pow(ijNormal.x-pNormal.x,2) + pow(ijNormal.y-pNormal.y,2) + pow(ijNormal.z-pNormal.z,2))/(normalStdDev * 2.0f * normalVariance + 0.000001f); 
                        normalVal = exp(-normalVal);

                        // ALBEDO1
                        albedo1Val = (pow(ijAlbedo1.x-pAlbedo1.x,2) + pow(ijAlbedo1.y-pAlbedo1.y,2) + pow(ijAlbedo1.z-pAlbedo1.z,2))/(albedo1StdDev * 2.0f * albedo1Variance + 0.000001f); 
                        albedo1Val = exp(-albedo1Val);

                        // ALBEDO2
                        albedo2Val = (pow(ijAlbedo2.x-pAlbedo2.x,2) + pow(ijAlbedo2.y-pAlbedo2.y,2) + pow(ijAlbedo2.z-pAlbedo2.z,2))/(albedo2StdDev * 2.0f * albedo2Variance + 0.000001f); 
                        albedo2Val = exp(-albedo2Val);

                        // WORLD POS
                        worldPosVal = (pow(ijWorldPos.x-pWorldPos.x,2) + pow(ijWorldPos.y-pWorldPos.y,2) + pow(ijWorldPos.z-pWorldPos.z,2))/( worldPosStdDev * 2.0f * worldPosVariance + 0.000001f); 
                        worldPosVal = exp(-worldPosVal);

                        // DIRECT LIGHT
                        directLightVal = pow(ijDirectLight-pDirectLight,2)/(directLightStdDev * 2.0f * directLightVariance + 0.000001f); 
                        directLightVal = exp(-directLightVal);

                        weight = indexVal*colVal*normalVal*albedo1Val*albedo2Val*worldPosVal*directLightVal; 
                        wSum += ijCol*weight;
                        wcSum += weight;
                    }
                }

                denoisedCol[index] = wSum / wcSum;
                denoisingInf[index].wcSum = wcSum;
            }
        } 
    }
    void Denoiser::CUDADenoise(){
        CUDADenoiser::denoise();
    }
    void Denoiser::OpenGLDenoise(){

    }
