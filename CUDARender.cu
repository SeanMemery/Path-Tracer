#include "CUDAHeader.h"
#include "Renderers.h"

    __device__ 
    float4 QMult(float4 q1, float4 q2 ) {
        auto A1 = (q1.w + q1.y) * (q2.y + q2.z);
        auto A3 = (q1.x - q1.z) * (q2.x + q2.w);
        auto A4 = (q1.x + q1.z) * (q2.x - q2.w);
        auto A2 = A1 + A3 + A4;
        auto A5 = (q1.w - q1.y) * (q2.y - q2.z);
        A5 = (A5 + A2) / 2.0f;

        auto Q1 = A5 - A1 + (q1.w - q1.z) * (q2.z - q2.w);
        auto Q2 = A5 - A2 + (q1.y + q1.x) * (q2.y + q2.x);
        auto Q3 = A5 - A3 + (q1.x - q1.y) * (q2.z + q2.w);
        auto Q4 = A5 - A4 + (q1.w + q1.z) * (q2.x - q2.y);

        return {Q1, Q2, Q3, Q4};
    }
    __device__
    void rotate(float3& to_rotate, float4 q) {
        float4 p  {0, to_rotate.x, to_rotate.y, to_rotate.z};
        float4 qR {q.x,-q.y,-q.z,-q.w};

        float4 ret = QMult(qR, QMult(p, q));
        to_rotate.x=ret.y;to_rotate.y=ret.z;to_rotate.z=ret.w;
    };

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
    float3 Intersect(float3 rayDir, float3 rayOrigin, Constants* constants, int& shapeHit, bool& hitAnything, float3& OBBSpacePosHit) {
        float E = 0.001f;
        float t = INFINITY;
        float3 posHit;
        for (int ind = 0; ind < constants->numShapes; ind++) {
            int shapeType = (int)constants->shapes[ind][0];
            int attrInd = (int)constants->shapes[ind][2];
            float tempT = INFINITY;
            // ----- intersect shapes -----
            // aabb
            if ( shapeType == 1) {
                // Transform Ray
                float3 rDir = {rayDir.x, rayDir.y, rayDir.z};
                float3 boxPos = {constants->objAttributes[attrInd + 0], constants->objAttributes[attrInd + 1], constants->objAttributes[attrInd + 2]};
                float3 rPos = {rayOrigin.x-boxPos.x, rayOrigin.y-boxPos.y, rayOrigin.z-boxPos.z};
                float4 rot = {constants->objAttributes[attrInd + 9], constants->objAttributes[attrInd + 10], constants->objAttributes[attrInd + 11], constants->objAttributes[attrInd + 12]};
                if (rot.y + rot.z + rot.w > E) {
                    rotate(rDir,rot);
                    rDir = normalize(rDir);
                    rotate(rPos,rot);
                }
                rPos.x+=boxPos.x;rPos.y+=boxPos.y;rPos.z+=boxPos.z;
                
                int sign[3] = {rDir.x < 0, rDir.y < 0, rDir.z < 0};
                float3 bounds[2];
                bounds[0].x = constants->objAttributes[attrInd + 3];
                bounds[0].y = constants->objAttributes[attrInd + 4];
                bounds[0].z = constants->objAttributes[attrInd + 5];
                bounds[1].x = constants->objAttributes[attrInd + 6];
                bounds[1].y = constants->objAttributes[attrInd + 7];
                bounds[1].z = constants->objAttributes[attrInd + 8];
                float tmin = (bounds[sign[0]].x - rPos.x) / rDir.x;
                float tmax = (bounds[1 - sign[0]].x - rPos.x) / rDir.x;
                float tymin = (bounds[sign[1]].y - rPos.y) / rDir.y;
                float tymax = (bounds[1 - sign[1]].y - rPos.y) / rDir.y;
                if ((tmin > tymax) || (tymin > tmax))
                    continue;
                if (tymin > tmin)
                    tmin = tymin;
                if (tymax < tmax)
                    tmax = tymax;
                float tzmin = (bounds[sign[2]].z - rPos.z) / rDir.z;
                float tzmax = (bounds[1 - sign[2]].z - rPos.z) / rDir.z;
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

                if (tempT < t) {
                    OBBSpacePosHit.x = rPos.x + rDir.x*tempT;
                    OBBSpacePosHit.y = rPos.y + rDir.y*tempT;
                    OBBSpacePosHit.z = rPos.z + rDir.z*tempT;          
                }  
            }
            // sphere
            else if (shapeType == 0) {
                float3 L;
                L.x = constants->objAttributes[attrInd+0] - rayOrigin.x;
                L.y = constants->objAttributes[attrInd+1] - rayOrigin.y;
                L.z = constants->objAttributes[attrInd+2] - rayOrigin.z;
                float tca = dot(L, rayDir);
                if (tca < E)
                    continue;
                float dsq = dot(L,L) - tca * tca;
                float radiusSq = constants->objAttributes[attrInd+3] * constants->objAttributes[attrInd+3];
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
            }
        }
        return posHit;
    }
    
    __device__
    bool ShadowIntersect(float3 rayDir, float3 rayOrigin, float maxT, int impShape, Constants* constants, float3 randDir) {
        float E = 0.001f;
        for (int ind = 0; ind < constants->numShapes; ind++) {
            if (ind == impShape)
                continue;
            int shapeType = constants->shapes[ind][0];
            int matInd = constants->shapes[ind][1];
            int attrInd   = constants->shapes[ind][2];
            float tempT = INFINITY;
            float3 posHit, obbPosHit;
            // ----- intersect shapes -----
            // aabb
            if ( shapeType == 1) {
                // Transform Ray
                float3 rDir = {rayDir.x, rayDir.y, rayDir.z};
                float3 boxPos = {constants->objAttributes[attrInd + 0], constants->objAttributes[attrInd + 1], constants->objAttributes[attrInd + 2]};
                float3 rPos = {rayOrigin.x-boxPos.x, rayOrigin.y-boxPos.y, rayOrigin.z-boxPos.z};
                float4 rot = {constants->objAttributes[attrInd + 9], constants->objAttributes[attrInd + 10], constants->objAttributes[attrInd + 11], constants->objAttributes[attrInd + 12]};
                if (rot.y + rot.z + rot.w > E) {
                    rotate(rDir,rot);
                    rDir = normalize(rDir);
                    rotate(rPos,rot);
                }
                rPos.x+=boxPos.x;rPos.y+=boxPos.y;rPos.z+=boxPos.z;

                int sign[3] = {rDir.x < 0, rDir.y < 0, rDir.z < 0};
                float3 bounds[2];
                bounds[0].x = constants->objAttributes[attrInd + 3];
                bounds[0].y = constants->objAttributes[attrInd + 4];
                bounds[0].z = constants->objAttributes[attrInd + 5];
                bounds[1].x = constants->objAttributes[attrInd + 6];
                bounds[1].y = constants->objAttributes[attrInd + 7];
                bounds[1].z = constants->objAttributes[attrInd + 8];
                float tmin = (bounds[sign[0]].x - rPos.x) / rDir.x;
                float tmax = (bounds[1 - sign[0]].x - rPos.x) / rDir.x;
                float tymin = (bounds[sign[1]].y - rPos.y) / rDir.y;
                float tymax = (bounds[1 - sign[1]].y - rPos.y) / rDir.y;
                if ((tmin > tymax) || (tymin > tmax))
                    continue;
                if (tymin > tmin)
                    tmin = tymin;
                if (tymax < tmax)
                    tmax = tymax;
                float tzmin = (bounds[sign[2]].z - rPos.z) / rDir.z;
                float tzmax = (bounds[1 - sign[2]].z - rPos.z) / rDir.z;
                if ((tmin > tzmax) || (tzmin > tmax))
                    continue;
                if (tzmin > tmin)
                    tmin = tzmin;
                if (tzmax < tmax)
                    tmax = tzmax;
                // Check times are positive, but use E for floating point accuracy
                tempT = tmin > E ? tmin : (tmax > E ? tmax : INFINITY); 

                if ((int)constants->matList[matInd][5]==3) {
                    obbPosHit.x = rPos.x + rDir.x*tempT;
                    obbPosHit.y = rPos.y + rDir.y*tempT;
                    obbPosHit.z = rPos.z + rDir.z*tempT;
                }             
            }
            // sphere
            else if (shapeType == 0) {
                float3 L;
                L.x = constants->objAttributes[attrInd+0] - rayOrigin.x;
                L.y = constants->objAttributes[attrInd+1] - rayOrigin.y;
                L.z = constants->objAttributes[attrInd+2] - rayOrigin.z;
                float tca = dot(L, rayDir);
                if (tca < E)
                    continue;
                float dsq = dot(L,L) - tca * tca;
                float radiusSq = constants->objAttributes[attrInd+3] * constants->objAttributes[attrInd+3];
                if (radiusSq - dsq < E)
                    continue;
                float thc = sqrtf(radiusSq - dsq);
                float t0 = tca - thc;
                float t1 = tca + thc;
                // Check times are positive, but use E for floating point accuracy
                tempT = t0 > E ? t0 : (t1 > E ? t1 : INFINITY); 
            }
            if (tempT < maxT) {
                // Dialectric Check
                if ((int)constants->matList[matInd][5]==3) {
                    float blur = constants->matList[matInd][3];
                    float RI = 1.0f / constants->matList[matInd][4];
                    // Get Normal
                    float3 refNorm {0.0f, 0.0f, 0.0f};
                    if (shapeType == 1) {
                        float3 bounds[2];
                        bounds[0].x = constants->objAttributes[attrInd + 3];
                        bounds[0].y = constants->objAttributes[attrInd + 4];
                        bounds[0].z = constants->objAttributes[attrInd + 5];
                        bounds[1].x = constants->objAttributes[attrInd + 6];
                        bounds[1].y = constants->objAttributes[attrInd + 7];
                        bounds[1].z = constants->objAttributes[attrInd + 8];

                        // Flat 
                        if (fabs(bounds[0].x - bounds[1].x) < E) {
                            refNorm.x = rayDir.x > 0 ? -1 : 1;
                        }
                        else if (fabs(bounds[0].y - bounds[1].y) < E) {
                            refNorm.y = rayDir.y > 0 ? -1 : 1;
                        }
                        else if (fabs(bounds[0].z - bounds[1].z) < E) {
                            refNorm.z = rayDir.z > 0 ? -1 : 1;
                        }
                        // Not Flat
                        else if (fabs(obbPosHit.x - bounds[0].x) < E)
                            refNorm.x = -1;
                        else if (fabs(obbPosHit.x - bounds[1].x) < E)
                            refNorm.x = 1;
                        else if (fabs(obbPosHit.y - bounds[0].y) < E)
                            refNorm.y = -1;
                        else if (fabs(obbPosHit.y - bounds[1].y) < E)
                            refNorm.y = 1;
                        else if (fabs(obbPosHit.z - bounds[0].z) < E)
                            refNorm.z = -1;
                        else if (fabs(obbPosHit.z - bounds[1].z) < E)
                            refNorm.z = 1;

                        // Transform Normal
                        float4 rot = {constants->objAttributes[attrInd + 9], -constants->objAttributes[attrInd + 10], -constants->objAttributes[attrInd + 11], -constants->objAttributes[attrInd + 12]};
                        rotate(refNorm, rot);
                        refNorm = normalize(refNorm);
                    }
                    else if (shapeType == 0) {
                            posHit.x = rayOrigin.x + rayDir.x*tempT;
                            posHit.y = rayOrigin.y + rayDir.y*tempT;
                            posHit.z = rayOrigin.z + rayDir.z*tempT;
                            refNorm.x = posHit.x - constants->objAttributes[attrInd+0];
                            refNorm.y = posHit.y - constants->objAttributes[attrInd+1];
                            refNorm.z = posHit.z - constants->objAttributes[attrInd+2];
                            refNorm = normalize(refNorm);
                        }
                    float cosi = dot(rayDir, refNorm);

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
                    bool canRefract = 1.0f - sinSq > E;
                            

                    if (!canRefract) {
                        rayDir.x = (rayDir.x - 2.0f*cosi*refNorm.x)*(1.0f - blur) + blur*randDir.x;
                        rayDir.y = (rayDir.y - 2.0f*cosi*refNorm.y)*(1.0f - blur) + blur*randDir.y;
                        rayDir.z = (rayDir.z - 2.0f*cosi*refNorm.z)*(1.0f - blur) + blur*randDir.z;
                        rayDir = normalize(rayDir);
                    }
                    else {
                        float refCalc = RI*cosi - sqrtf(1-sinSq);
                        rayDir.x = (RI*rayDir.x + refCalc*refNorm.x)*(1.0f - blur) + blur*randDir.x;
                        rayDir.y = (RI*rayDir.y + refCalc*refNorm.y)*(1.0f - blur) + blur*randDir.y;
                        rayDir.z = (RI*rayDir.z + refCalc*refNorm.z)*(1.0f - blur) + blur*randDir.z;
                        rayDir = normalize(rayDir);					
                    }
                    continue;
                }
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
        float3 back_col = {0,0,0};

        // Store ray collisions and reverse through them (last num is shape index)
        float4 rayPositions[12];
        float3 normals[12];
        float pdfVals[12];
        // Shadow Rays: counts succesful shadow rays i.e. direct lighting, done for each bounce to provide more info
        int shadowRays[12];
        for (int v=0; v<12; v++){
            pdfVals[v] = 1.0f / M_PI;
            shadowRays[v] = 0;
        }

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
            float3 OBBSpacePosHit {0.0f,0.0f,0.0f};
            bool hitAnything = false;
            int shapeTypeHit, matInd, attrInd;
            // Collide with shapes, generating new dirs as needed (i.e. random or specular)
            {

                float E = 0.00001f;

                // Find shape
                posHit = Intersect(dir, prevPos, constants, shapeHit, hitAnything, OBBSpacePosHit);
                shapeTypeHit = constants->shapes[shapeHit][0];
                matInd       = constants->shapes[shapeHit][1];
                attrInd      = constants->shapes[shapeHit][2];

                if (hitAnything) {

                    // Get Normal
                    {
                        if (shapeTypeHit == 1) {
                            float3 bounds[2];
                            bounds[0].x = constants->objAttributes[attrInd + 3];
                            bounds[0].y = constants->objAttributes[attrInd + 4];
                            bounds[0].z = constants->objAttributes[attrInd + 5];
                            bounds[1].x = constants->objAttributes[attrInd + 6];
                            bounds[1].y = constants->objAttributes[attrInd + 7];
                            bounds[1].z = constants->objAttributes[attrInd + 8];
                            normals[pos].x = 0.0f;
                            normals[pos].y = 0.0f;
                            normals[pos].z = 0.0f;

                            // Flat 
                            if (fabs(bounds[0].x - bounds[1].x) < E) {
                                normals[pos].x = dir.x > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0].y - bounds[1].y) < E) {
                                normals[pos].y = dir.y > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0].z - bounds[1].z) < E) {
                                normals[pos].z = dir.z > E ? -1 : 1;
                            }
                            // Not Flat
                            else if (fabs(OBBSpacePosHit.x - bounds[0].x) < E)
                                normals[pos].x = -1;
                            else if (fabs(OBBSpacePosHit.x - bounds[1].x) < E)
                                normals[pos].x = 1;
                            else if (fabs(OBBSpacePosHit.y - bounds[0].y) < E)
                                normals[pos].y = -1;
                            else if (fabs(OBBSpacePosHit.y - bounds[1].y) < E)
                                normals[pos].y = 1;
                            else if (fabs(OBBSpacePosHit.z - bounds[0].z) < E)
                                normals[pos].z = -1;
                            else if (fabs(OBBSpacePosHit.z - bounds[1].z) < E)
                                normals[pos].z = 1;

                            // Transform Normal
                            float4 rot = {constants->objAttributes[attrInd + 9], -constants->objAttributes[attrInd + 10], -constants->objAttributes[attrInd + 11], -constants->objAttributes[attrInd + 12]};
                            rotate(normals[pos], rot);
                            normals[pos] = normalize(normals[pos]);
                        }
                        else if (shapeTypeHit == 0) {
                            normals[pos].x = posHit.x - constants->objAttributes[attrInd+0];
                            normals[pos].y = posHit.y - constants->objAttributes[attrInd+1];
                            normals[pos].z = posHit.z - constants->objAttributes[attrInd+2];
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
                        if (constants->matList[matInd][5] == 3) {
                            // Dielectric Material
                            shadowRays[pos] = 1;
                            float blur = constants->matList[matInd][3];
                            float RI = 1.0 / constants->matList[matInd][4];
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
                            bool canRefract = 1.0f - sinSq > E;
                            
                            // Schlick approx
                            float r0 = (1.0f - RI) / (1.0f + RI);
                            r0 = r0 * r0;
                            float schlick = r0 + (1.0f - r0) * powf((1.0f - cosi), 5.0f);
                            
                            float schlickRand = RandBetween(s, 0, 1);

                            if (!canRefract || schlick > schlickRand) {
                                dir.x = (dirIn.x - 2.0f*cosi*refNorm.x)*(1.0f - blur) + blur*randDir.x;
                                dir.y = (dirIn.y - 2.0f*cosi*refNorm.y)*(1.0f - blur) + blur*randDir.y;
                                dir.z = (dirIn.z - 2.0f*cosi*refNorm.z)*(1.0f - blur) + blur*randDir.z;
                                dir = normalize(dir);
                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                float cosine2 = dot(dir, normals[pos]);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;
                            }
                            else {

                                float refCalc = RI*cosi - sqrtf(1-sinSq);
                                dir.x = (RI*dirIn.x + refCalc*refNorm.x)*(1.0f - blur) + blur*randDir.x;
                                dir.y = (RI*dirIn.y + refCalc*refNorm.y)*(1.0f - blur) + blur*randDir.y;
                                dir.z = (RI*dirIn.z + refCalc*refNorm.z)*(1.0f - blur) + blur*randDir.z;
                                dir = normalize(dir);

                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                // Danger here, scattering pdf was going to 0 for refracting and making colour explode
                                float cosine2 = dot(dir, normals[pos]);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;					
                            }
                        }
                        else if (constants->matList[matInd][5] == 2) {
                            // Metal material
                            shadowRays[pos] = 1;

                            float3 dirIn = dir;
                            float3 refNorm = normals[pos];
                            float blur = constants->matList[matInd][3];

                            float prevDirNormalDot = dot(dirIn, refNorm);

                            dir.x = (dirIn.x - 2.0f*prevDirNormalDot*refNorm.x)*(1.0f - blur) + blur*randDir.x;
                            dir.y = (dirIn.y - 2.0f*prevDirNormalDot*refNorm.y)*(1.0f - blur) + blur*randDir.y;
                            dir.z = (dirIn.z - 2.0f*prevDirNormalDot*refNorm.z)*(1.0f - blur) + blur*randDir.z;

                            float cosine2 = dot(dir, normals[pos]);

                            // Same as scattering pdf, to make pdf 1, as it is specular
                            pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;
                        }
                        else {
                            // Lambertian/Light Material

                            dir = randDir;

                            // Check Light Mat
                            bool mixPdf;
                            int impInd, impShape, impShapeAttrInd;
                            if (constants->matList[matInd][5] == 1) {
                                shadowRays[pos] = 1;
                                mixPdf = false;
                            } else {
                                // Get importance shape
                                mixPdf = constants->numImportantShapes > 0;

                                if (mixPdf) { 
                                    impInd = rands[3] * constants->numImportantShapes * 0.99999f;
                                    impShape = constants->importantShapes[impInd];
                                    impShapeAttrInd = constants->shapes[impShape][2];
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
                                    if (constants->shapes[impShape][0] == 1) {
                                        // Gen three new random variables : [0, 1]
                                        float aabbRands[3];
                                        for (int n = 0; n < 3; n++) 
                                            aabbRands[n] = RandBetween(s, 0, 1);
                                        randPos.x = (1.0f - aabbRands[0])*constants->objAttributes[impShapeAttrInd+3] + aabbRands[0]*constants->objAttributes[impShapeAttrInd+6];
                                        randPos.y = (1.0f - aabbRands[1])*constants->objAttributes[impShapeAttrInd+4] + aabbRands[1]*constants->objAttributes[impShapeAttrInd+7];
                                        randPos.z = (1.0f - aabbRands[2])*constants->objAttributes[impShapeAttrInd+5] + aabbRands[2]*constants->objAttributes[impShapeAttrInd+8];	
                                    } 
                                    else if (constants->shapes[impShape][7] == 0) {

                                        // Gen three new random variables : [-1, 1]
                                        float3 sphereRands;
                                        sphereRands.x = RandBetween(s, -1, 1);
                                        sphereRands.y = RandBetween(s, -1, 1);
                                        sphereRands.z = RandBetween(s, -1, 1);
                                        sphereRands = normalize(sphereRands);

                                        randPos.x = constants->objAttributes[impShapeAttrInd+0] + sphereRands.x*constants->objAttributes[impShapeAttrInd+3];
                                        randPos.y = constants->objAttributes[impShapeAttrInd+1] + sphereRands.y*constants->objAttributes[impShapeAttrInd+3];
                                        randPos.z = constants->objAttributes[impShapeAttrInd+2] + sphereRands.z*constants->objAttributes[impShapeAttrInd+3];
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


                                    if (!ShadowIntersect(directDir, posHit, dirLen, impShape, constants, randDir)) {
                                        float cosine = fabs(dot(directDir, randDir));
                                        if (cosine > 0.01f) {
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
                                if (constants->shapes[impShape][0] == 1) {
                                    // AABB pdf val
                                    float areaSum = 0;
                                    float xDiff = constants->objAttributes[impShapeAttrInd+3] - constants->objAttributes[impShapeAttrInd+6];
                                    float yDiff = constants->objAttributes[impShapeAttrInd+4] - constants->objAttributes[impShapeAttrInd+7];
                                    float zDiff = constants->objAttributes[impShapeAttrInd+5] - constants->objAttributes[impShapeAttrInd+8];
                                    areaSum += xDiff * yDiff * 2.0f;
                                    areaSum += zDiff * yDiff * 2.0f;
                                    areaSum += xDiff * zDiff * 2.0f;
                                    float cosine = dot(dir, normals[pos]);
                                    cosine = cosine < 0.0001f ? 0.0001f : cosine;

                                    float3 diff;
                                    diff.x = constants->objAttributes[impShapeAttrInd+0] - posHit.x;
                                    diff.y = constants->objAttributes[impShapeAttrInd+1] - posHit.y;
                                    diff.z = constants->objAttributes[impShapeAttrInd+2] - posHit.z;
                                    float dirLen = sqrtf(dot(diff, diff));


                                    // AABB needs magic number for pdf calc, TODO: LOOK INTO, was too bright before
                                    //p1 = 1 / (cosine * areaSum);
                                    p1 = dirLen / (cosine * areaSum);

                                } else if (constants->shapes[impShape][0] == 0) {
                                    // Sphere pdf val
                                    float3 diff =   {constants->objAttributes[impShapeAttrInd+0]-posHit.x, 
                                                     constants->objAttributes[impShapeAttrInd+1]-posHit.y, 
                                                     constants->objAttributes[impShapeAttrInd+2]-posHit.z};
                                    float distance_squared = dot(diff, diff);
                                    float cos_theta_max = sqrtf(1 - constants->objAttributes[impShapeAttrInd+3] * constants->objAttributes[impShapeAttrInd+3] / distance_squared);
                                    // NaN check
                                    cos_theta_max = (cos_theta_max != cos_theta_max) ? 0.9999f : cos_theta_max;
                                    float solid_angle = M_PI * (1.0f - cos_theta_max) *2.0f;

                                    // Sphere needs magic number for pdf calc, TODO: LOOK INTO, was too dark before
                                    //p1 = 1 / (solid_angle );
                                    p1 = constants->objAttributes[impShapeAttrInd+3] / (solid_angle * sqrtf(distance_squared)*4.0f);
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
            int matInd = constants->shapes[shapeHit][1];

            float3 albedo = {constants->matList[matInd][0], 
                             constants->matList[matInd][1], 
                             constants->matList[matInd][2]};
            int matType = (int)constants->matList[matInd][5];	
        
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
            int matIndAlb1 = constants->shapes[(int)rayPositions[0].w][1];
            ret->albedo1[0] =  constants->matList[matIndAlb1][0];
            ret->albedo1[1] =  constants->matList[matIndAlb1][1];
            ret->albedo1[2] =  constants->matList[matIndAlb1][2];
            int matIndAlb2 = constants->shapes[(int)rayPositions[1].w][1];
            ret->albedo2[0] =  constants->matList[matIndAlb2][0];
            ret->albedo2[1] =  constants->matList[matIndAlb2][1];
            ret->albedo2[2] =  constants->matList[matIndAlb2][2];
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

            memcpy(CUDAConstants, &constants, sizeof(Constants));

            // Generate random seeds for each pixel
            for (int ind = 0; ind < numPixels; ind++) {
                uint64_t s0 = renderer.GloRandS[0];
                uint64_t s1 = renderer.GloRandS[1];
                s1 ^= s0;
                renderer.GloRandS[0] = (s0 << 49) | ((s0 >> (64 - 49)) ^ s1 ^ (s1 << 21));
                renderer.GloRandS[1] = (s1 << 28) | (s1 >> (64 - 28));
                RandomSeeds s;
                s.s1 = renderer.GloRandS[0];
                s.s2 = renderer.GloRandS[1];
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