#include "CUDARender.h"
#include "Renderers.h"

    __device__
    float RandBetween(RandomSeeds& s, float min, float max) {
        uint64_t s0 = s.s1;
        uint64_t s1 = s.s2;
        uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
        double one_two = ((uint64_t)1 << 63) * (double)2.0;
        float rand = xorshiro / one_two;
        s1 ^= s0;
        s.s1 = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
        s.s2 = (s1 << 28) | (s1 >> 36);

        rand *= max - min;
        rand += min;
        return rand;
    }

    __device__
    float3 cross(float3 a, float3 b) {
        float3 ret;
        ret.x = a.y*b.z - a.z*b.y;
        ret.y = a.z*b.x - a.x*b.z;
        ret.z = a.x*b.y - a.y*b.x;
        return ret;
    }

    __device__
    float dot(float3 a, float3 b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    __device__
    float3 normalize(float3 v){
        float rSqrt = rsqrtf(dot(v,v));
        float3 ret = {v.x*rSqrt, v.y*rSqrt, v.z*rSqrt};
        return ret;
    }

    __device__
    float3 Intersect(float3 rayDir, float3 rayOrigin, Constants* constants, int& shapeHit, int& shapeTypeHit, bool& hitAnything) {
        float E = 0.001f;
        float t = INFINITY;
        float3 posHit;
        for (int ind = 0; ind < constants->numShapes; ind++) {
            int shapeType = (int)constants->shapes[ind][7];
            float tempT = INFINITY;
            // ----- intersect shapes -----
            // aabb
            if ( shapeType == 0) {
                int sign[3] = {rayDir.x < 0, rayDir.y < 0, rayDir.z < 0};
                float3 bounds[2];
                bounds[0].x = constants->shapes[ind][8];
                bounds[0].y = constants->shapes[ind][9];
                bounds[0].z = constants->shapes[ind][10];
                bounds[1].x = constants->shapes[ind][11];
                bounds[1].y = constants->shapes[ind][12];
                bounds[1].z = constants->shapes[ind][13];
                float tmin = (bounds[sign[0]].x - rayOrigin.x) / rayDir.x;
                float tmax = (bounds[1 - sign[0]].x - rayOrigin.x) / rayDir.x;
                float tymin = (bounds[sign[1]].y - rayOrigin.y) / rayDir.y;
                float tymax = (bounds[1 - sign[1]].y - rayOrigin.y) / rayDir.y;
                if ((tmin > tymax) || (tymin > tmax))
                    continue;
                if (tymin > tmin)
                    tmin = tymin;
                if (tymax < tmax)
                    tmax = tymax;
                float tzmin = (bounds[sign[2]].z - rayOrigin.z) / rayDir.z;
                float tzmax = (bounds[1 - sign[2]].z - rayOrigin.z) / rayDir.z;
                if ((tmin > tzmax) || (tzmin > tmax))
                    continue;
                if (tzmin > tmin)
                    tmin = tzmin;
                if (tzmax < tmax)
                    tmax = tzmax;
                // Check times are positive, but use E for floating point accuracy
                if (tmin > E)
                    tempT = tmin;
                else if (tmax > E)
                    tempT = tmax;
                else
                    continue;
            }
            // sphere
            else if (shapeType == 1) {
                float3 L;
                L.x = constants->shapes[ind][0] - rayOrigin.x;
                L.y = constants->shapes[ind][1] - rayOrigin.y;
                L.z = constants->shapes[ind][2] - rayOrigin.z;
                float tca = dot(L, rayDir);
                if (tca < E)
                    continue;
                float dsq = dot(L,L) - tca * tca;
                float radiusSq = constants->shapes[ind][8] * constants->shapes[ind][8];
                if (radiusSq - dsq < E)
                    continue;
                float thc = sqrtf(radiusSq - dsq);
                float t0 = tca - thc;
                float t1 = tca + thc;
                // Check times are positive, but use E for floating point accuracy
                if (t0 > E)
                    tempT = t0;
                else if (t1 > E)
                    tempT = t1;
                else 
                    continue;
            }
            if (tempT < t) {
                hitAnything = true;
                t = tempT;
                posHit.x = rayOrigin.x + rayDir.x*t;
                posHit.y = rayOrigin.y + rayDir.y*t;
                posHit.z = rayOrigin.z + rayDir.z*t;
                shapeHit = ind;
                shapeTypeHit = shapeType;
            }
        }
        return posHit;
    }
    
    __device__
    bool ShadowIntersect(float3 rayDir, float3 rayOrigin, float maxT, int impShape, Constants* constants) {
        float E = 0.001f;
        for (int ind = 0; ind < constants->numShapes; ind++) {
            if (ind == impShape)
                continue;
            int shapeType = (int)constants->shapes[ind][7];
            float tempT = INFINITY;
            // ----- intersect shapes -----
            // aabb
            if ( shapeType == 0) {
                int sign[3] = {rayDir.x < 0, rayDir.y < 0, rayDir.z < 0};
                float3 bounds[2];
                bounds[0].x = constants->shapes[ind][8];
                bounds[0].y = constants->shapes[ind][9];
                bounds[0].z = constants->shapes[ind][10];
                bounds[1].x = constants->shapes[ind][11];
                bounds[1].y = constants->shapes[ind][12];
                bounds[1].z = constants->shapes[ind][13];
                float tmin = (bounds[sign[0]].x - rayOrigin.x) / rayDir.x;
                float tmax = (bounds[1 - sign[0]].x - rayOrigin.x) / rayDir.x;
                float tymin = (bounds[sign[1]].y - rayOrigin.y) / rayDir.y;
                float tymax = (bounds[1 - sign[1]].y - rayOrigin.y) / rayDir.y;
                if ((tmin > tymax) || (tymin > tmax))
                    continue;
                if (tymin > tmin)
                    tmin = tymin;
                if (tymax < tmax)
                    tmax = tymax;
                float tzmin = (bounds[sign[2]].z - rayOrigin.z) / rayDir.z;
                float tzmax = (bounds[1 - sign[2]].z - rayOrigin.z) / rayDir.z;
                if ((tmin > tzmax) || (tzmin > tmax))
                    continue;
                if (tzmin > tmin)
                    tmin = tzmin;
                if (tzmax < tmax)
                    tmax = tzmax;
                // Check times are positive, but use E for floating point accuracy
                tempT = tmin > E ? tmin : (tmax > E ? tmax : INFINITY); 
            }
            // sphere
            else if (shapeType == 1) {
                float3 L;
                L.x = constants->shapes[ind][0] - rayOrigin.x;
                L.y = constants->shapes[ind][1] - rayOrigin.y;
                L.z = constants->shapes[ind][2] - rayOrigin.z;
                float tca = dot(L, rayDir);
                if (tca < E)
                    continue;
                float dsq = dot(L,L) - tca * tca;
                float radiusSq = constants->shapes[ind][8] * constants->shapes[ind][8];
                if (radiusSq - dsq < E)
                    continue;
                float thc = sqrtf(radiusSq - dsq);
                float t0 = tca - thc;
                float t1 = tca + thc;
                // Check times are positive, but use E for floating point accuracy
                tempT = t0 > E ? t0 : (t1 > E ? t1 : INFINITY); 
            }
            if (tempT < maxT) {
                return true;
            }
        } 
        return false;      
    }

    __global__
    void CUDARenderFunc(RandomSeeds* CUDASeeds, Constants* constants, ReturnStruct* CUDAReturn) {

        uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
        uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (i >= constants->RESH || j >= constants->RESV)
            return;

        int pixel = j*constants->RESH + i;

        // Ray
        float3 camPos = {constants->camPos[0], constants->camPos[1], constants->camPos[2]};
        float3 rayPos = camPos;

        // Random seeds
        RandomSeeds s = CUDASeeds[pixel]; 

        // Rand Samp
        float rSamps[2] = {0.0f, 0.0f};
        if (constants->randSamp>0.001f) {
            rSamps[0] = RandBetween(s, -1, 1) * constants->randSamp;
            rSamps[1] = RandBetween(s, -1, 1) * constants->randSamp;
        }

        // Pixel Coord
        float3 camForward = {constants->camForward[0], constants->camForward[1], constants->camForward[2]};
        float3 camRight = {constants->camRight[0], constants->camRight[1], constants->camRight[2]};

        float pY = -constants->maxAngleV + 2*constants->maxAngleV*((float)j/(float)constants->RESV);
        float pX = -constants->maxAngleH + 2*constants->maxAngleH*((float)i/(float)constants->RESH);

        float3 pix;
        pix.x = camPos.x + constants->camForward[0]*constants->focalLength + constants->camRight[0]*(pX+rSamps[0]) + constants->camUp[0]*(pY+rSamps[1]);
        pix.y = camPos.y + constants->camForward[1]*constants->focalLength + constants->camRight[1]*(pX+rSamps[0]) + constants->camUp[1]*(pY+rSamps[1]);
        pix.z = camPos.z + constants->camForward[2]*constants->focalLength + constants->camRight[2]*(pX+rSamps[0]) + constants->camUp[2]*(pY+rSamps[1]);

        float3 rayDir = {pix.x-camPos.x, pix.y-camPos.y, pix.z-camPos.z};
        rayDir = normalize(rayDir);

        // Background col
        float3 back_col = {constants->backgroundColour[0], constants->backgroundColour[1], constants->backgroundColour[2]};

        // Store ray collisions and reverse through them (last num is shape index)
        float4 rayPositions[12];
        float3 normals[12];
        float pdfVals[12];
        // Shadow Rays: counts succesful shadow rays i.e. direct lighting, done for each bounce to provide more info
        int shadowRays[12];
        for (int v=0; v<12; v++){
            pdfVals[v] = 1.0f / (4.0f*M_PI);
            shadowRays[v] = 0;
        }

        /*
            - loop through shapes
            - append to positions if hit shape, break if not
            - add shape index as 4th component 

            constants->shapes
            - pos: 0, 1, 2
            - albedo: 3, 4, 5
            - mat type: 6 (0 lambertian, 1 light, 2 metal, 3 dielectric)
            - shape type: 7 (0 AABB, 1 Sphere)
                - 0: min bound 8, 9, 10, max bound 11, 12, 13
                - 1: radius 8
            - blur: 14
            - r index: 15
        */

        int numShapeHit = 0;
        int numRays = 0;
        float3 dir = rayDir;
        for (int pos = 0; pos < constants->maxDepth; pos++) {
            numRays++;
            int shapeHit;
            float3 prevPos;
            if (pos > 0) {
                prevPos.x = rayPositions[pos-1].x;
                prevPos.y = rayPositions[pos-1].y;
                prevPos.z = rayPositions[pos-1].z;
            } else {
                prevPos = camPos;
            }
            float3 posHit {0.0f, 0.0f, 0.0f};
            bool hitAnything = false;
            int shapeTypeHit;
            // Collide with shapes, generating new dirs as needed (i.e. random or specular)
            {

                float E = 0.001f;

                // Find shape
                {
                    float t = INFINITY;
                    for (int ind = 0; ind < constants->numShapes; ind++) {
                        int shapeType = (int)constants->shapes[ind][7];
                        float tempT = INFINITY;
                        // ----- intersect shapes -----
                        // aabb
                        if ( shapeType == 0) {
                            int sign[3] = {dir.x < 0, dir.y < 0, dir.z < 0};
                            float3 bounds[2];
                            bounds[0].x = constants->shapes[ind][8];
                            bounds[0].y = constants->shapes[ind][9];
                            bounds[0].z = constants->shapes[ind][10];
                            bounds[1].x = constants->shapes[ind][11];
                            bounds[1].y = constants->shapes[ind][12];
                            bounds[1].z = constants->shapes[ind][13];
                            float tmin = (bounds[sign[0]].x - prevPos.x) / dir.x;
                            float tmax = (bounds[1 - sign[0]].x - prevPos.x) / dir.x;
                            float tymin = (bounds[sign[1]].y - prevPos.y) / dir.y;
                            float tymax = (bounds[1 - sign[1]].y - prevPos.y) / dir.y;
                            if ((tmin > tymax) || (tymin > tmax))
                                continue;
                            if (tymin > tmin)
                                tmin = tymin;
                            if (tymax < tmax)
                                tmax = tymax;
                            float tzmin = (bounds[sign[2]].z - prevPos.z) / dir.z;
                            float tzmax = (bounds[1 - sign[2]].z - prevPos.z) / dir.z;
                            if ((tmin > tzmax) || (tzmin > tmax))
                                continue;
                            if (tzmin > tmin)
                                tmin = tzmin;
                            if (tzmax < tmax)
                                tmax = tzmax;
                            // Check times are positive, but use E for floating point accuracy
                            if (tmin > E)
                                tempT = tmin;
                            else if (tmax > E)
                                tempT = tmax;
                            else
                                continue;
                        }
                        // sphere
                        else if (shapeType == 1) {
                            float3 L;
                            L.x = constants->shapes[ind][0] - prevPos.x;
                            L.y = constants->shapes[ind][1] - prevPos.y;
                            L.z = constants->shapes[ind][2] - prevPos.z;
                            float tca = dot(L, dir);
                            if (tca < E)
                                continue;
                            float dsq = dot(L,L) - tca * tca;
                            float radiusSq = constants->shapes[ind][8] * constants->shapes[ind][8];
                            if (radiusSq - dsq < E)
                                continue;
                            float thc = sqrtf(radiusSq - dsq);
                            float t0 = tca - thc;
                            float t1 = tca + thc;
                            // Check times are positive, but use E for floating point accuracy
                            if (t0 > E)
                                tempT = t0;
                            else if (t1 > E)
                                tempT = t1;
                            else 
                                continue;
                        }
                        if (tempT < t) {
                            hitAnything = true;
                            t = tempT;
                            posHit.x = prevPos.x + dir.x*t;
                            posHit.y = prevPos.y + dir.y*t;
                            posHit.z = prevPos.z + dir.z*t;
                            shapeHit = ind;
                            shapeTypeHit = shapeType;
                        }
                    }
                }

                if (hitAnything) {

                    // Get Normal
                    {
                        if (shapeTypeHit == 0) {
                            float3 bounds[2];
                            bounds[0].x = constants->shapes[shapeHit][8];
                            bounds[0].y = constants->shapes[shapeHit][9];
                            bounds[0].z = constants->shapes[shapeHit][10];
                            bounds[1].x = constants->shapes[shapeHit][11];
                            bounds[1].y = constants->shapes[shapeHit][12];
                            bounds[1].z = constants->shapes[shapeHit][13];
                            normals[pos].x = 0.0f;
                            normals[pos].y = 0.0f;
                            normals[pos].z = 0.0f;

                            // Flat 
                            if (fabs(bounds[0].x - bounds[1].x) < E) {
                                normals[pos].x = dir.x > 0 ? -1 : 1;
                            }
                            else if (fabs(bounds[0].y - bounds[1].y) < E) {
                                normals[pos].y = dir.y > 0 ? -1 : 1;
                            }
                            else if (fabs(bounds[0].z - bounds[1].z) < E) {
                                normals[pos].z = dir.z > 0 ? -1 : 1;
                            }
                            // Non Flat
                            else if (fabs(posHit.x - bounds[0].x) < E)
                                normals[pos].x = -1;
                            else if (fabs(posHit.x - bounds[1].x) < E)
                                normals[pos].x = 1;
                            else if (fabs(posHit.y - bounds[0].y) < E)
                                normals[pos].y = -1;
                            else if (fabs(posHit.y - bounds[1].y) < E)
                                normals[pos].y = 1;
                            else if (fabs(posHit.z - bounds[0].z) < E)
                                normals[pos].z = -1;
                            else if (fabs(posHit.z - bounds[1].z) < E)
                                normals[pos].z = 1;
                        }
                        else if (shapeTypeHit == 1) {
                            normals[pos].x = posHit.x - constants->shapes[shapeHit][0];
                            normals[pos].y = posHit.y - constants->shapes[shapeHit][1];
                            normals[pos].z = posHit.z - constants->shapes[shapeHit][2];
                            normals[pos] = normalize(normals[pos]);
                        }
                    }
                
                    // Gen new dirs and pdfs
                    {

                        // Random Dir Generation
                            float3 randDir {0.0f, 0.0f, 0.0f};
                            float rands[5];
                            {
                                // Rand vals
                                for (int n = 0; n < 5; n++) 
                                    rands[n] = RandBetween(s, 0, 1);

                                float3 axis[3];
                                axis[0] = {0.0f, 0.0f, 0.0f};
                                axis[1] = {0.0f, 0.0f, 0.0f};
                                axis[2] = {0.0f, 0.0f, 0.0f};
                                // 2
                                // axis[2] = normal
                                axis[2].x = normals[pos].x;
                                axis[2].y = normals[pos].y;
                                axis[2].z = normals[pos].z;
                                // 1
                                if (fabs(axis[2].x) > 0.9) {
                                    // axis[1] = cross(axis[2], [0,1,0])
                                    axis[1].x = -axis[2].z;
                                    axis[1].z =  axis[2].x; 
                                }
                                else {
                                    // axis[1] = cross(axis[2], [1,0,0])
                                    axis[1].y =  axis[2].z;
                                    axis[1].z = -axis[2].y;
                                }
                                axis[1] = normalize(axis[1]);
                                // 0
                                // axis[0] = cross(axis[2], axis[1])
                                axis[0] = cross(axis[2], axis[1]);
                                // rand dir
                                float phi = 2.0f * M_PI * rands[0];
                                float x = cosf(phi) * sqrtf(rands[1]);
                                float y = sinf(phi) * sqrtf(rands[1]);
                                float z = sqrtf(1 - rands[1]);

                                randDir.x = x * axis[0].x + y * axis[1].x + z * axis[2].x;
                                randDir.y = x * axis[0].y + y * axis[1].y + z * axis[2].y;
                                randDir.z = x * axis[0].z + y * axis[1].z + z * axis[2].z;
                            }
                        // Random Dir Generation

                        /*
                            Random chance to be normal random ray or importance sampled, then
                            pdfs of both are averaged

                            Need to generate both importance direction and random direction, use one 
                            on random, but average pdf val from both

                            if angle between point is too low (e.g. light and point both on ceiling) then dont importance sample!

                            if material is metal, get reflect ray

                            if material dielectric, choose whether to reflect or refract
                        */
                        if (constants->shapes[shapeHit][6] == 3) {
                            // Dielectric Material
                            shadowRays[pos] = 1;
                            float blur = constants->shapes[shapeHit][14];
                            float RI = 1.0 / constants->shapes[shapeHit][15];
                            float3 dirIn = dir;
                            float3 refNorm = normals[pos];
                            float cosi = dot(dirIn, refNorm);

                            // If normal is same direction as ray, then flip
                            if (cosi > 0) {
                                refNorm.x*=-1.0f;refNorm.y*=-1.0f;refNorm.z*=-1.0f;
                                RI = 1.0f / RI;
                            }
                            else {
                                cosi*=-1.0f;
                            }

                            // Can refract check
                            float sinSq = RI*RI*(1.0f-cosi*cosi);
                            bool canRefract = 1.0f - sinSq > 0.0f;
                            
                            // Schlick approx
                            float r0 = (1.0f - RI) / (1.0f + RI);
                            r0 = r0 * r0;
                            float schlick = r0 + (1.0f - r0) * powf((1.0f - cosi), 5.0f);

                            if (!canRefract){//} || schlick > rands[2]) {
                                dir.x = dirIn.x - 2.0f*cosi*refNorm.x + blur*randDir.x;
                                dir.y = dirIn.y - 2.0f*cosi*refNorm.y + blur*randDir.y;
                                dir.z = dirIn.z - 2.0f*cosi*refNorm.z + blur*randDir.z;
                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                float cosine2 = dot(dir, refNorm);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;
                            }
                            else {

                                float refCalc = RI*cosi - sqrtf(1-sinSq);
                                dir.x = RI*dirIn.x + refCalc*refNorm.x + blur*randDir.x;
                                dir.y = RI*dirIn.y + refCalc*refNorm.y + blur*randDir.y;
                                dir.z = RI*dirIn.z + refCalc*refNorm.z + blur*randDir.z;
                                dir = normalize(dir);

                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                // Danger here, scattering pdf was going to 0 for refracting and making colour explode
                                float cosine2 = dot(dir, refNorm);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;					
                            }
                        }
                        else if (constants->shapes[shapeHit][6] == 2) {
                            // Metal material
                            shadowRays[pos] = 1;

                            float3 dirIn = dir;
                            float3 refNorm = normals[pos];
                            float blur = constants->shapes[shapeHit][14];

                            float prevDirNormalDot = dot(dirIn, refNorm);

                            dir.x = dirIn.x - 2.0f*prevDirNormalDot*refNorm.x + blur*randDir.x;
                            dir.y = dirIn.y - 2.0f*prevDirNormalDot*refNorm.y + blur*randDir.y;
                            dir.z = dirIn.z - 2.0f*prevDirNormalDot*refNorm.z + blur*randDir.z;

                            float cosine2 = dot(dir, refNorm);

                            // Same as scattering pdf, to make pdf 1, as it is specular
                            pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;
                        }
                        else {
                            // Lambertian/Light Material

                            dir = randDir;

                            // Check Light Mat
                            bool mixPdf;
                            int impInd, impShape;
                            if (constants->shapes[shapeHit][6] == 1) {
                                shadowRays[pos] = 1;
                                mixPdf = false;
                            } else {
                                // Get importance shape
                                mixPdf = constants->numImportantShapes > 0;

                                if (mixPdf) { 
                                    impInd = rands[3] * constants->numImportantShapes * 0.99999f;
                                    impShape = constants->importantShapes[impInd];
                                    if (impShape==shapeHit) {
                                        mixPdf = false;
                                        // mixPdf = constants->numImportantShapes > 1;
                                        // impInd = (impInd+1) % constants->numImportantShapes;
                                        // impShape = constants->importantShapes[impInd];
                                    } 
                                }
                            }

                            // Calculate PDF val
                            if (mixPdf) {
                                // 0
                                float p0 = 1.0f / M_PI;
                                bool choosePdf = rands[4] > 0.65f;
                                // dot(randomly generated dir, ray dir) / PI
                                if (choosePdf) {
                                    // Generate dir towards importance shape
                                    float3 randPos {0.0f, 0.0f, 0.0f};
                                    if (constants->shapes[impShape][7] == 0) {
                                        // Gen three new random variables : [0, 1]
                                        float aabbRands[3];
                                        for (int n = 0; n < 3; n++) 
                                            aabbRands[n] = RandBetween(s, 0, 1);
                                        randPos.x = (1.0f - aabbRands[0])*constants->shapes[impShape][8]  + aabbRands[0]*constants->shapes[impShape][11];
                                        randPos.y = (1.0f - aabbRands[1])*constants->shapes[impShape][9]  + aabbRands[1]*constants->shapes[impShape][12];
                                        randPos.z = (1.0f - aabbRands[2])*constants->shapes[impShape][10] + aabbRands[2]*constants->shapes[impShape][13];	
                                    } 
                                    else if (constants->shapes[impShape][7] == 1) {

                                        // Gen three new random variables : [-1, 1]
                                        float3 sphereRands;
                                        sphereRands.x = RandBetween(s, -1, 1);
                                        sphereRands.y = RandBetween(s, -1, 1);
                                        sphereRands.z = RandBetween(s, -1, 1);
                                        sphereRands = normalize(sphereRands);

                                        randPos.x = constants->shapes[impShape][0] + sphereRands.x*constants->shapes[impShape][8];
                                        randPos.y = constants->shapes[impShape][1] + sphereRands.y*constants->shapes[impShape][8];
                                        randPos.z = constants->shapes[impShape][2] + sphereRands.z*constants->shapes[impShape][8];
                                    }

                                    float3 directDir;
                                    directDir.x = randPos.x - posHit.x;
                                    directDir.y = randPos.y - posHit.y;
                                    directDir.z = randPos.z - posHit.z;
                                    float dirLen = sqrtf(dot(directDir, directDir));
                                    directDir.x /= dirLen;
                                    directDir.y /= dirLen;
                                    directDir.z /= dirLen;  

                                    //
                                    // Shadow Ray
                                    // Need to send shadow ray to see if point is in path of direct light
                                    bool shadowRayHit = false;

                                    for (int ind = 0; ind < constants->numShapes; ind++) {
                                        if (ind == impShape)
                                            continue;
                                        numRays++;
                                        int shapeType = (int)constants->shapes[ind][7];
                                        float tempT = INFINITY;
                                        // ----- intersect shapes -----
                                        // aabb
                                        if ( shapeType == 0) {
                                            int sign[3] = {dir.x < 0, dir.y < 0, dir.z < 0};
                                            float3 bounds[2];
                                            bounds[0].x = constants->shapes[ind][8];
                                            bounds[0].y = constants->shapes[ind][9];
                                            bounds[0].z = constants->shapes[ind][10];
                                            bounds[1].x = constants->shapes[ind][11];
                                            bounds[1].y = constants->shapes[ind][12];
                                            bounds[1].z = constants->shapes[ind][13];
                                            float tmin = (bounds[sign[0]].x - posHit.x) / dir.x;
                                            float tmax = (bounds[1 - sign[0]].x - posHit.x) / dir.x;
                                            float tymin = (bounds[sign[1]].y - posHit.y) / dir.y;
                                            float tymax = (bounds[1 - sign[1]].y - posHit.y) / dir.y;
                                            if ((tmin > tymax) || (tymin > tmax))
                                                continue;
                                            if (tymin > tmin)
                                                tmin = tymin;
                                            if (tymax < tmax)
                                                tmax = tymax;
                                            float tzmin = (bounds[sign[2]].z - posHit.z) / dir.z;
                                            float tzmax = (bounds[1 - sign[2]].z - posHit.z) / dir.z;
                                            if ((tmin > tzmax) || (tzmin > tmax))
                                                continue;
                                            if (tzmin > tmin)
                                                tmin = tzmin;
                                            if (tzmax < tmax)
                                                tmax = tzmax;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = tmin > E ? tmin : (tmax > E ? tmax : INFINITY); 
                                        }
                                        // sphere
                                        else if (shapeType == 1) {
                                            float3 L;
                                            L.x = constants->shapes[ind][0] - posHit.x;
                                            L.y = constants->shapes[ind][1] - posHit.y;
                                            L.z = constants->shapes[ind][2] - posHit.z;
                                            float tca = dot(L, dir);
                                            if (tca < E)
                                                continue;
                                            float dsq = dot(L,L) - tca * tca;
                                            float radiusSq = constants->shapes[ind][8] * constants->shapes[ind][8];
                                            if (radiusSq - dsq < E)
                                                continue;
                                            float thc = sqrtf(radiusSq - dsq);
                                            float t0 = tca - thc;
                                            float t1 = tca + thc;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = t0 > E ? t0 : (t1 > E ? t1 : INFINITY); 
                                        }
                                        if (tempT < dirLen) {
                                            shadowRayHit = true;
                                            break;
                                        }
                                    }

                                    if (!shadowRayHit) {
                                        float cosine = fabs(dot(directDir, randDir));
                                        if (cosine < 0.01) {
                                            p0 = 1 / M_PI;
                                        }
                                        else {
                                            shadowRays[pos]=1; 
                                            dir = directDir;
                                            p0 = fabs(cosine) / M_PI;
                                        }

                                    }
                                    // Shadow Ray
                                    //
                                    //


                                }
                                // 1
                                float p1 = 0;
                                if (constants->shapes[impShape][7] == 0) {
                                    // AABB pdf val
                                    float areaSum = 0;
                                    float xDiff = constants->shapes[impShape][11] - constants->shapes[impShape][8];
                                    float yDiff = constants->shapes[impShape][12] - constants->shapes[impShape][9];
                                    float zDiff = constants->shapes[impShape][13] - constants->shapes[impShape][10];
                                    areaSum += xDiff * yDiff * 2.0f;
                                    areaSum += zDiff * yDiff * 2.0f;
                                    areaSum += xDiff * zDiff * 2.0f;
                                    float cosine = dot(dir, normals[pos]);
                                    cosine = cosine < 0.0001f ? 0.0001f : cosine;

                                    float3 diff;
                                    diff.x = constants->shapes[impShape][0] - posHit.x;
                                    diff.y = constants->shapes[impShape][1] - posHit.y;
                                    diff.z = constants->shapes[impShape][2] - posHit.z;
                                    float dirLen = sqrtf(dot(diff, diff));


                                    // AABB needs magic number for pdf calc, TODO: LOOK INTO, was too bright before
                                    //p1 = 1 / (cosine * areaSum);
                                    p1 = dirLen / (cosine * areaSum);

                                } else if (constants->shapes[impShape][7] == 1) {
                                    // Sphere pdf val
                                    float3 diff =   {constants->shapes[impShape][0]-posHit.x, 
                                                     constants->shapes[impShape][1]-posHit.y, 
                                                     constants->shapes[impShape][2]-posHit.z};
                                    float distance_squared = dot(diff, diff);
                                    float cos_theta_max = sqrtf(1 - constants->shapes[impShape][8] * constants->shapes[impShape][8] / distance_squared);
                                    // NaN check
                                    cos_theta_max = (cos_theta_max != cos_theta_max) ? 0.9999f : cos_theta_max;
                                    float solid_angle = M_PI * (1.0f - cos_theta_max) *2.0f;

                                    // Sphere needs magic number for pdf calc, TODO: LOOK INTO, was too dark before
                                    //p1 = 1 / (solid_angle );
                                    p1 = constants->shapes[impShape][8] / (solid_angle * sqrtf(distance_squared)*4.0f);
                                }

                                pdfVals[pos] = 0.5f*p0 + 0.5f*p1;
                            }
                        }
                    }

                    numShapeHit++;
                    rayPositions[pos].x = posHit.x;
                    rayPositions[pos].y = posHit.y;
                    rayPositions[pos].z = posHit.z;
                    rayPositions[pos].w = shapeHit;

                } else {
                    back_col.x = 0.1f;
                    back_col.y = 0.1f;
                    back_col.z = (dir.y + 1.0f)/2.2f + 0.1f;
                    break;
                }
            }
        }

        float3 finalCol = {back_col.x, back_col.y, back_col.z};

        // Reverse through hit points and add up colour
        for (int pos = numShapeHit-1; pos >=0; pos--) {

            int shapeHit = (int)rayPositions[pos].w;

            float3 albedo = {constants->shapes[shapeHit][3], 
                                constants->shapes[shapeHit][4], 
                                constants->shapes[shapeHit][5]};
            int matType = (int)constants->shapes[shapeHit][6];	
            int shapeType = (int)constants->shapes[shapeHit][7];
        
            float3 normal = {normals[pos].x, normals[pos].y, normals[pos].z};

            float3 newDir;
            newDir.x = pos == numShapeHit-1 ? dir.x : rayPositions[pos + 1].x - rayPositions[pos].x;
            newDir.y = pos == numShapeHit-1 ? dir.y : rayPositions[pos + 1].y - rayPositions[pos].y;
            newDir.z = pos == numShapeHit-1 ? dir.z : rayPositions[pos + 1].z - rayPositions[pos].z;
            if (pos < numShapeHit-1) {
                newDir = normalize(newDir);
            }

            float3 emittance;
            emittance.x = matType == 1 ? albedo.x : 0;
            emittance.y = matType == 1 ? albedo.y : 0;
            emittance.z = matType == 1 ? albedo.z : 0;

            float pdf_val = pdfVals[pos]; 

            float cosine2 = dot(normal, newDir);

            float scattering_pdf = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;// just cosine/pi for lambertian
            float3 multVecs = {albedo.x*finalCol.x,   // albedo*incoming 
                               albedo.y*finalCol.y,  
                               albedo.z*finalCol.z}; 

            float directLightMult = shadowRays[pos]==1 && constants->numImportantShapes>1 ? constants->numImportantShapes : 1;

            float pdfs = scattering_pdf / pdf_val;
            finalCol.x = emittance.x + multVecs.x * pdfs * directLightMult;
            finalCol.y = emittance.y + multVecs.y * pdfs * directLightMult;
            finalCol.z = emittance.z + multVecs.z * pdfs * directLightMult;
        }
        ReturnStruct* ret = &CUDAReturn[pixel];
        ret->xyz[0] = finalCol.x;
        ret->xyz[1] = finalCol.y;
        ret->xyz[2] = finalCol.z;
        if (constants->getDenoiserInf == 1) {
            ret->normal[0] = normals[0].x;
            ret->normal[1] = normals[0].y;
            ret->normal[2] = normals[0].z;
            ret->albedo1[0] = constants->shapes[(int)rayPositions[0].w][3];
            ret->albedo1[1] = constants->shapes[(int)rayPositions[0].w][4];
            ret->albedo1[2] = constants->shapes[(int)rayPositions[0].w][5];
            ret->albedo2[0] = constants->shapes[(int)rayPositions[1].w][3];
            ret->albedo2[1] = constants->shapes[(int)rayPositions[1].w][4];
            ret->albedo2[2] = constants->shapes[(int)rayPositions[1].w][5];
            ret->directLight = 0.0f;
            for (int c = 0; c<constants->maxDepth; c++)
                ret->directLight += (float)shadowRays[c] / (float)constants->maxDepth;
            ret->worldPos[0] = rayPositions[0].x;
            ret->worldPos[1] = rayPositions[0].y;
            ret->worldPos[2] = rayPositions[0].z;
        }
        ret->raysSent = numRays;
    }

    void CUDARender::render() {
        int numPixels = xRes*yRes;

        RandomSeeds*  CUDASeeds;
        Constants*    CUDAConstants;
        ReturnStruct* CUDAReturn;

        cudaMallocManaged(&CUDASeeds,     numPixels*sizeof(RandomSeeds));
        cudaMallocManaged(&CUDAConstants, sizeof(Constants));
        cudaMallocManaged(&CUDAReturn,    numPixels*sizeof(ReturnStruct));

        memcpy(CUDAConstants, &renderer.constants, sizeof(Constants));

        // Generate random seeds for each pixel
        for (int ind = 0; ind < numPixels; ind++) {
            uint64_t s0 = renderer.constants.GloRandS[0];
            uint64_t s1 = renderer.constants.GloRandS[1];
            s1 ^= s0;
            renderer.constants.GloRandS[0] = (s0 << 49) | ((s0 >> (64 - 49)) ^ s1 ^ (s1 << 21));
            renderer.constants.GloRandS[1] = (s1 << 28) | (s1 >> (64 - 28));
            RandomSeeds s;
            s.s1 = renderer.constants.GloRandS[0];
            s.s2 = renderer.constants.GloRandS[1];
            CUDASeeds[ind] = s;
        }

        dim3 numBlocks(xRes/rootThreadsPerBlock + 1, 
                       yRes/rootThreadsPerBlock + 1); 

        CUDARenderFunc<<<numBlocks,dim3(rootThreadsPerBlock, rootThreadsPerBlock)>>>(CUDASeeds, CUDAConstants, CUDAReturn );       

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        ReturnStruct ret;
        sampleCount++;
        rayCount = 0;
        DenoisingInf* info;
		for (int index = 0; index < numPixels; index++) {
			ret = CUDAReturn[index];

            preScreen[index] += vec3(ret.xyz[0], ret.xyz[1], ret.xyz[2]);
			rayCount += ret.raysSent;

			// Denoiser info
			if (denoising) {
                info = &denoisingInf[index];
				normal[index]      += vec3(ret.normal[0], ret.normal[1], ret.normal[2]);
				albedo1[index]     += vec3(ret.albedo1[0], ret.albedo1[1], ret.albedo1[2]);
				albedo2[index]     += vec3(ret.albedo2[0], ret.albedo2[1], ret.albedo2[2]);
				directLight[index] += vec3(ret.directLight, ret.directLight, ret.directLight);
				worldPos[index]    += vec3(ret.worldPos[0], ret.worldPos[1], ret.worldPos[2]);
				// Standard Deviations
                info->stdDevVecs[0] += vec3(
                     pow(preScreen[index].x/sampleCount - ret.xyz[0],2),
                     pow(preScreen[index].y/sampleCount - ret.xyz[1],2),     
                     pow(preScreen[index].z/sampleCount - ret.xyz[2],2));
				info->stdDevVecs[1] += vec3(
                     pow(normal[index].x/sampleCount - ret.normal[0],2),
                     pow(normal[index].y/sampleCount - ret.normal[1],2),     
                     pow(normal[index].z/sampleCount - ret.normal[2],2));
				info->stdDevVecs[2] += vec3(
                    pow(albedo1[index].x/sampleCount - ret.albedo1[0],2),
				    pow(albedo1[index].y/sampleCount - ret.albedo1[1],2),    
				    pow(albedo1[index].z/sampleCount - ret.albedo1[2],2));
				info->stdDevVecs[3] += vec3(
                    pow(albedo2[index].x/sampleCount - ret.albedo2[0],2),
				    pow(albedo2[index].y/sampleCount - ret.albedo2[1],2),    
				    pow(albedo2[index].z/sampleCount - ret.albedo2[2],2));
				info->stdDevVecs[4] += vec3(
                    pow(worldPos[index].x/sampleCount - ret.worldPos[0],2),
				    pow(worldPos[index].y/sampleCount - ret.worldPos[1],2),   
				    pow(worldPos[index].z/sampleCount - ret.worldPos[2],2));
				info->stdDevVecs[5] += vec3(pow(directLight[index].x/sampleCount - ret.directLight,2),0,0); 
                info->stdDev[0] = (info->stdDevVecs[0][0] + info->stdDevVecs[0][1] + info->stdDevVecs[0][2])/sampleCount;
                info->stdDev[1] = (info->stdDevVecs[1][0] + info->stdDevVecs[1][1] + info->stdDevVecs[1][2])/sampleCount;
                info->stdDev[2] = (info->stdDevVecs[2][0] + info->stdDevVecs[2][1] + info->stdDevVecs[2][2])/sampleCount;
                info->stdDev[3] = (info->stdDevVecs[3][0] + info->stdDevVecs[3][1] + info->stdDevVecs[3][2])/sampleCount;
                info->stdDev[4] = (info->stdDevVecs[4][0] + info->stdDevVecs[4][1] + info->stdDevVecs[4][2])/sampleCount;
                info->stdDev[5] =  info->stdDevVecs[5][0]/sampleCount;
			}
		}  

        // Free memory
        cudaFree(CUDASeeds);
        cudaFree(CUDAConstants);
        cudaFree(CUDAReturn);
    }