#define SKEPU_PRECOMPILED 1
#define SKEPU_OPENMP 1
#define SKEPU_CUDA 1
#include "Renderers.h"

    static ReturnStruct RenderFunc(skepu::Index2D ind, RandomSeeds seeds,  Constants  sConstants) {
        // Ray
        float camPos[3] = { sConstants.camPos[0],  sConstants.camPos[1],  sConstants.camPos[2]};
        float rayPos[3] = {camPos[0], camPos[1], camPos[2]};

        // Lambda Functions
            auto dot  = [=](float vec1[3], float vec2[3]) {return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];};
            auto norm = [&](float vec[3]) {auto d = sqrt(dot(vec,vec)); vec[0]/=d; vec[1]/=d; vec[2]/=d; };
            auto randBetween = [](RandomSeeds& seeds, float min, float max) {
                uint64_t s0 = seeds.s1;
                uint64_t s1 = seeds.s2;
                uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                double one_two = ((uint64_t)1 << 63) * (double)2.0;
                float rand = xorshiro / one_two;
                s1 ^= s0;
                seeds.s1 = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                seeds.s2 = (s1 << 28) | (s1 >> 36);

                rand *= max - min;
                rand += min;
                return rand;
            };
            auto QMult = [&](float q1[4], float q2[4] ) {
                auto A1 = (q1[3] + q1[1]) * (q2[1] + q2[2]);
                auto A3 = (q1[0] - q1[2]) * (q2[0] + q2[3]);
                auto A4 = (q1[0] + q1[2]) * (q2[0] - q2[3]);
                auto A2 = A1 + A3 + A4;
                auto A5 = (q1[3] - q1[1]) * (q2[1] - q2[2]);
                A5 = (A5 + A2) / 2.0f;

                auto Q1 = A5 - A1 + (q1[3] - q1[2]) * (q2[2] - q2[3]);
                auto Q2 = A5 - A2 + (q1[1] + q1[0]) * (q2[1] + q2[0]);
                auto Q3 = A5 - A3 + (q1[0] - q1[1]) * (q2[2] + q2[3]);
                auto Q4 = A5 - A4 + (q1[3] + q1[2]) * (q2[0] - q2[1]);

                q1[0] = Q1; q1[1] = Q2; q1[2] = Q3; q1[3] = Q4;
            };
            auto rotate = [&](float to_rotate[3], float q[4]) {

                float p[4]  {0, to_rotate[0], to_rotate[1], to_rotate[2]};
                float qR[4] {q[0],-q[1],-q[2],-q[3]};

                QMult(p, q);
                QMult(qR, p);
                to_rotate[0]=qR[1];to_rotate[1]=qR[2];to_rotate[2]=qR[3];
            };
        // Lambda Functions

        // Rand Samp
        float rSamps[2] = {0.0f, 0.0f};
        if (sConstants.randSamp>0.001f) {
            rSamps[0] = randBetween( seeds, -1, 1) * sConstants.randSamp;
            rSamps[1] = randBetween( seeds, -1, 1) * sConstants.randSamp;
        }

        float back_col[3] = { 0,0,0};

        // Pixel Coord
        float camForward[3] = { sConstants.camForward[0],  sConstants.camForward[1],  sConstants.camForward[2]};
        float camRight[3] = { sConstants.camRight[0],  sConstants.camRight[1],  sConstants.camRight[2]};

        float pY = - sConstants.maxAngleV + 2.0f* sConstants.maxAngleV*((float)ind.row/(float) sConstants.RESV);
        float pX = - sConstants.maxAngleH + 2.0f* sConstants.maxAngleH*((float)ind.col/(float) sConstants.RESH);

        float pix[3] = {0,0,0};
        pix[0] = camPos[0] +  sConstants.camForward[0]* sConstants.focalLength +  sConstants.camRight[0]*(pX+rSamps[0]) +  sConstants.camUp[0]*(pY+rSamps[1]);
        pix[1] = camPos[1] +  sConstants.camForward[1]* sConstants.focalLength +  sConstants.camRight[1]*(pX+rSamps[0]) +  sConstants.camUp[1]*(pY+rSamps[1]);
        pix[2] = camPos[2] +  sConstants.camForward[2]* sConstants.focalLength +  sConstants.camRight[2]*(pX+rSamps[0]) +  sConstants.camUp[2]*(pY+rSamps[1]);

        float rayDir[3] = {pix[0]-camPos[0], pix[1]-camPos[1], pix[2]-camPos[2]};
        norm(rayDir);
 
        // Store ray collisions and reverse through them (last num is shape index)
        float rayPositions[12][4];
        float normals[12][3];
        float pdfVals[12];
        // Shadow Rays: counts succesful shadow rays i.e. direct lighting, done for each bounce to provide more info
        int shadowRays[12];
        for (int v=0; v<12; v++){
            normals[v][0]=0.0f;normals[v][1]=0.0f;normals[v][2]=0.0f;
            pdfVals[v] = 1.0f / M_PI;
            shadowRays[v] = 0;
        }

        int numShapeHit = 0;
        int numRays = 0;
        float dir[3] = {rayDir[0], rayDir[1], rayDir[2]};
        for (int pos = 0; pos <  sConstants.maxDepth; pos++) {
            numRays++;
            int shapeHit;
            float prevPos[3];
            if (pos > 0) {
                prevPos[0] = rayPositions[pos-1][0];
                prevPos[1] = rayPositions[pos-1][1];
                prevPos[2] = rayPositions[pos-1][2];
            } else {
                prevPos[0] = camPos[0];
                prevPos[1] = camPos[1];
                prevPos[2] = camPos[2];
            }
            float posHit[3] {0.0f, 0.0f, 0.0f};
            float OBBSpacePosHit[3] {0.0f, 0.0f, 0.0f};
            bool hitAnything = false;
            int shapeTypeHit, attrInd, matInd;
            // Collide with shapes, generating new dirs as needed (i.e. random or specular)
            {

                float E = 0.00001f;

                // Find shape
                {
                    float t = INFINITY;
                    for (int ind = 0; ind <  sConstants.numShapes; ind++) {
                        int shapeType = sConstants.shapes[ind][0];
                        int aInd = sConstants.shapes[ind][2];
                        float tempT = INFINITY;
                        // ----- intersect shapes -----
                        // aabb
                        if ( shapeType == 1) {

                            // Transform Ray
                            float rDir[3] = {dir[0], dir[1], dir[2]};
                            float boxPos[3] = {sConstants.objAttributes[aInd + 0], sConstants.objAttributes[aInd + 1], sConstants.objAttributes[aInd + 2]};
                            float rPos[3] = {prevPos[0]-boxPos[0], prevPos[1]-boxPos[1], prevPos[2]-boxPos[2]};
                            float rot[4] = {sConstants.objAttributes[aInd + 9], sConstants.objAttributes[aInd + 10], sConstants.objAttributes[aInd + 11], sConstants.objAttributes[aInd + 12]};
                            if (rot[1] + rot[2] + rot[3] > E) {
                                rotate(rDir,rot);
                                norm(rDir); 
                                rotate(rPos,rot);
                            }
                            rPos[0]+=boxPos[0];rPos[1]+=boxPos[1];rPos[2]+=boxPos[2];

                            int sign[3] = {rDir[0] < 0, rDir[1] < 0, rDir[2] < 0};
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] =  sConstants.objAttributes[aInd + 3];
                            bounds[0][1] =  sConstants.objAttributes[aInd + 4];
                            bounds[0][2] =  sConstants.objAttributes[aInd + 5];
                            bounds[1][0] =  sConstants.objAttributes[aInd + 6];
                            bounds[1][1] =  sConstants.objAttributes[aInd + 7];
                            bounds[1][2] =  sConstants.objAttributes[aInd + 8];
                            float tmin = (bounds[sign[0]][0] - rPos[0]) / rDir[0];
                            float tmax = (bounds[1 - sign[0]][0] - rPos[0]) / rDir[0];
                            float tymin = (bounds[sign[1]][1] - rPos[1]) / rDir[1];
                            float tymax = (bounds[1 - sign[1]][1] - rPos[1]) / rDir[1];
                            if ((tmin > tymax) || (tymin > tmax))
                                continue;
                            if (tymin > tmin)
                                tmin = tymin;
                            if (tymax < tmax)
                                tmax = tymax;
                            float tzmin = (bounds[sign[2]][2] - rPos[2]) / rDir[2];
                            float tzmax = (bounds[1 - sign[2]][2] - rPos[2]) / rDir[2];
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
                                OBBSpacePosHit[0] = rPos[0] + rDir[0]*tempT;
                                OBBSpacePosHit[1] = rPos[1] + rDir[1]*tempT;
                                OBBSpacePosHit[2] = rPos[2] + rDir[2]*tempT;
                            }
                        }
                        // sphere
                        else if (shapeType == 0) {
                            float L[3] = {0,0,0};
                            L[0] =  sConstants.objAttributes[aInd + 0] - prevPos[0];
                            L[1] =  sConstants.objAttributes[aInd + 1] - prevPos[1];
                            L[2] =  sConstants.objAttributes[aInd + 2] - prevPos[2];
                            float tca = dot(L, dir);
                            if (tca < E)
                                continue;
                            float dsq = dot(L,L) - tca * tca;
                            float radiusSq =  sConstants.objAttributes[aInd + 3] *  sConstants.objAttributes[aInd + 3];
                            if (radiusSq - dsq < E)
                                continue;
                            float thc = sqrt(radiusSq - dsq);
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
                            posHit[0] = prevPos[0] + dir[0]*t;
                            posHit[1] = prevPos[1] + dir[1]*t;
                            posHit[2] = prevPos[2] + dir[2]*t;
                            shapeHit = ind;
                            attrInd = aInd;
                            matInd = sConstants.shapes[shapeHit][1];
                            shapeTypeHit = shapeType;
                        }
                    }
                }

                if (hitAnything) {

                    // Get Normal
                    {
                        if (shapeTypeHit == 1) {
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] =  sConstants.objAttributes[attrInd + 3];
                            bounds[0][1] =  sConstants.objAttributes[attrInd + 4];
                            bounds[0][2] =  sConstants.objAttributes[attrInd + 5];
                            bounds[1][0] =  sConstants.objAttributes[attrInd + 6];
                            bounds[1][1] =  sConstants.objAttributes[attrInd + 7];
                            bounds[1][2] =  sConstants.objAttributes[attrInd + 8];
                            normals[pos][0] = 0;
                            normals[pos][1] = 0;
                            normals[pos][2] = 0;

                            // Flat 
                            if (fabs(bounds[0][0] - bounds[1][0]) < E) {
                                normals[pos][0] = dir[0] > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0][1] - bounds[1][1]) < E) {
                                normals[pos][1] = dir[1] > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0][2] - bounds[1][2]) < E) {
                                normals[pos][2] = dir[2] > E ? -1 : 1;
                            }
                            // Not Flat
                            else if (fabs(OBBSpacePosHit[0] - bounds[0][0]) < E)
                                normals[pos][0] = -1;
                            else if (fabs(OBBSpacePosHit[0] - bounds[1][0]) < E)
                                normals[pos][0] = 1;
                            else if (fabs(OBBSpacePosHit[1] - bounds[0][1]) < E)
                                normals[pos][1] = -1;
                            else if (fabs(OBBSpacePosHit[1] - bounds[1][1]) < E)
                                normals[pos][1] = 1;
                            else if (fabs(OBBSpacePosHit[2] - bounds[0][2]) < E)
                                normals[pos][2] = -1;
                            else if (fabs(OBBSpacePosHit[2] - bounds[1][2]) < E)
                                normals[pos][2] = 1;

                            // Transform Normal
                            float rot[4] = {sConstants.objAttributes[attrInd + 9], -sConstants.objAttributes[attrInd + 10], -sConstants.objAttributes[attrInd + 11], -sConstants.objAttributes[attrInd + 12]};
                            rotate(normals[pos], rot);
                            norm(normals[pos]);
                        }
                        else if (shapeTypeHit == 0) {
                            normals[pos][0] = posHit[0] -  sConstants.objAttributes[attrInd + 0];
                            normals[pos][1] = posHit[1] -  sConstants.objAttributes[attrInd + 1];
                            normals[pos][2] = posHit[2] -  sConstants.objAttributes[attrInd + 2];
                            norm(normals[pos]);
                        }
                    }
                
                    // Gen new dirs and pdfs
                    {

                        // Random Dir Generation
                            float randDir[3];
                            float rands[5];
                            {
                                // Rand vals
                                for (int n = 0; n < 5; n++) 
                                    rands[n] = randBetween( seeds, 0,1);

                                float axis[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
                                // 2
                                // axis[2] = normal
                                axis[2][0] = normals[pos][0];
                                axis[2][1] = normals[pos][1];
                                axis[2][2] = normals[pos][2];
                                // 1
                                if (fabs(axis[2][0]) > 0.9) {
                                    // axis[1] = cross(axis[2], [0,1,0])
                                    axis[1][0] = -axis[2][2];
                                    axis[1][2] =  axis[2][0]; 
                                }
                                else {
                                    // axis[1] = cross(axis[2], [1,0,0])
                                    axis[1][1] =  axis[2][2];
                                    axis[1][2] = -axis[2][1];
                                }
                                norm(axis[1]);
                                // 0
                                // axis[0] = cross(axis[2], axis[1])
                                axis[0][0] = axis[2][1]*axis[1][2] - axis[2][2]*axis[1][1];
                                axis[0][1] = axis[2][2]*axis[1][0] - axis[2][0]*axis[1][2];
                                axis[0][2] = axis[2][0]*axis[1][1] - axis[2][1]*axis[1][0];
                                // rand dir
                                float phi = 2.0f * M_PI * rands[0];
                                float x = cos(phi) * sqrt(rands[1]);
                                float y = sin(phi) * sqrt(rands[1]);
                                float z = sqrt(1.0f - rands[1]);

                                randDir[0] = x * axis[0][0] + y * axis[1][0] + z * axis[2][0];
                                randDir[1] = x * axis[0][1] + y * axis[1][1] + z * axis[2][1];
                                randDir[2] = x * axis[0][2] + y * axis[1][2] + z * axis[2][2];
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
                        if ( sConstants.matList[matInd][5] == 3) {
                            // Dielectric Material
                            shadowRays[pos] = 1;
                            float blur =  sConstants.matList[matInd][3];
                            float RI = 1.0f /  sConstants.matList[matInd][4];
                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float refNorm[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};
                            float cosi = dot(dirIn, refNorm);

                            // If normal is same direction as ray, then flip
                            if (cosi > 0) {
                                refNorm[0]*=-1.0f;refNorm[1]*=-1.0f;refNorm[2]*=-1.0f;
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
                            float schlick = r0 + (1.0f - r0) * (float)pow((1.0f - cosi), 5.0f);

                            float schlickRand = randBetween(seeds, 0, 1);

                            if (!canRefract || schlick > schlickRand) {
                                dir[0] = (dirIn[0] - 2.0f*cosi*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                dir[1] = (dirIn[1] - 2.0f*cosi*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                dir[2] = (dirIn[2] - 2.0f*cosi*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                norm(dir);
                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                float cosine2 = dot(normals[pos], dir);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;
                            }
                            else {

                                float refCalc = RI*cosi - (float)sqrt(1.0f-sinSq);
                                dir[0] = (RI*dirIn[0] + refCalc*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                dir[1] = (RI*dirIn[1] + refCalc*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                dir[2] = (RI*dirIn[2] + refCalc*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                norm(dir);

                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                // Danger here, scattering pdf was going to 0 for refracting and making colour explode
                                float cosine2 = dot(normals[pos], dir);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;					
                            }
                        }
                        else if ( sConstants.matList[matInd][5] == 2) {
                            // Metal material
                            shadowRays[pos] = 1;

                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float blur =  sConstants.matList[matInd][3];

                            float prevDirNormalDot = dot(dirIn, normals[pos]);

                            dir[0] = (dirIn[0] - 2.0f*prevDirNormalDot*normals[pos][0])*(1.0f - blur) + blur*randDir[0];
                            dir[1] = (dirIn[1] - 2.0f*prevDirNormalDot*normals[pos][1])*(1.0f - blur) + blur*randDir[1];
                            dir[2] = (dirIn[2] - 2.0f*prevDirNormalDot*normals[pos][2])*(1.0f - blur) + blur*randDir[2];

                            float cosine2 = dot(dir, normals[pos]);

                            // Same as scattering pdf, to make pdf 1, as it is specular
                            pdfVals[pos] = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;
                        }
                        else {
                            // Lambertian/Light Material

                            dir[0] = randDir[0];
                            dir[1] = randDir[1];
                            dir[2] = randDir[2];

                            // Check Light Mat
                            bool mixPdf;
                            int impInd, impShape;
                            if ( sConstants.matList[matInd][5] == 1) {
                                shadowRays[pos] = 1;
                                mixPdf = false;
                            } else {
                                // Get importance shape
                                mixPdf =  sConstants.numImportantShapes > 0;

                                if (mixPdf) { 
                                    impInd = rands[3] *  sConstants.numImportantShapes * 0.99999f;
                                    impShape =  sConstants.importantShapes[impInd];
                                    if (impShape==shapeHit) {
                                        mixPdf = false;
                                        // mixPdf =  sConstants.numImportantShapes > 1;
                                        // impInd = (impInd+1) %  sConstants.numImportantShapes;
                                        // impShape =  sConstants.importantShapes[impInd];
                                    } 
                                }
                            }

                            // Calculate PDF val
                            if (mixPdf) {
                                // 0
                                float p0 = 1 / M_PI;
                                int choosePdf = rands[4] > 0.65f ? 1 : 0;
                                int impAttrInd = sConstants.shapes[impShape][2];
                                // dot(randomly generated dir, ray dir) / PI
                                if (choosePdf == 1) {
                                    // Generate dir towards importance shape
                                    float randPos[3] = {0,0,0};
                                    if ( sConstants.shapes[impShape][0] == 1) {
                                        // Gen three new random variables : [0, 1]
                                        float aabbRands[3];
                                        for (int n = 0; n < 3; n++) 
                                            aabbRands[n] = randBetween( seeds, 0,1);
                                        randPos[0] = (1.0f - aabbRands[0])* sConstants.objAttributes[impAttrInd+3] + aabbRands[0]* sConstants.objAttributes[impAttrInd+6];
                                        randPos[1] = (1.0f - aabbRands[1])* sConstants.objAttributes[impAttrInd+4] + aabbRands[1]* sConstants.objAttributes[impAttrInd+7];
                                        randPos[2] = (1.0f - aabbRands[2])* sConstants.objAttributes[impAttrInd+5] + aabbRands[2]* sConstants.objAttributes[impAttrInd+8];	
                                    } 
                                    else if ( sConstants.shapes[impShape][0] == 0) {
                                        // Gen three new random variables : [-1, 1]
                                        float sphereRands[3];
                                        sphereRands[0] = randBetween( seeds, -1,1);
                                        sphereRands[1] = randBetween( seeds, -1,1);
                                        sphereRands[2] = randBetween( seeds, -1,1);
                                        norm(sphereRands);
                                        
                                        randPos[0] =  sConstants.objAttributes[impAttrInd+0] + sphereRands[0]* sConstants.objAttributes[impAttrInd+3];
                                        randPos[1] =  sConstants.objAttributes[impAttrInd+1] + sphereRands[1]* sConstants.objAttributes[impAttrInd+3];
                                        randPos[2] =  sConstants.objAttributes[impAttrInd+2] + sphereRands[2]* sConstants.objAttributes[impAttrInd+3];
                                    }

                                    float directDir[3];
                                    directDir[0] = randPos[0] - posHit[0];
                                    directDir[1] = randPos[1] - posHit[1];
                                    directDir[2] = randPos[2] - posHit[2];
                                    float dirLen = sqrt(dot(directDir, directDir));
                                    directDir[0] /= dirLen; directDir[1] /= dirLen; directDir[2] /= dirLen;  

                                    //
                                    // Shadow Ray
                                    // Need to send shadow ray to see if point is in path of direct light
                                    bool shadowRayHit = false;
                                    float shadowDir[3] {directDir[0], directDir[1], directDir[2]};
                                    for (int ind = 0; ind <  sConstants.numShapes; ind++) {
                                        if (ind == impShape)
                                            continue;
                                        int shapeType = sConstants.shapes[ind][0];
                                        int sMatInd = sConstants.shapes[ind][1];
                                        int aInd = sConstants.shapes[ind][2];
                                        float tempT = INFINITY;
                                        float obbPosHit[3];
                                        // ----- intersect shapes -----
                                        // aabb
                                        if ( shapeType == 1) {
                                            // Transform Ray
                                            float rDir[3] = {shadowDir[0], shadowDir[1], shadowDir[2]};
                                            float boxPos[3] = {sConstants.objAttributes[aInd + 0], sConstants.objAttributes[aInd + 1], sConstants.objAttributes[aInd + 2]};
                                            float rPos[3] = {posHit[0]-boxPos[0], posHit[1]-boxPos[1], posHit[2]-boxPos[2]};
                                            float rot[4] = {sConstants.objAttributes[aInd + 9], sConstants.objAttributes[aInd + 10], sConstants.objAttributes[aInd + 11], sConstants.objAttributes[aInd + 12]};
                                            if (rot[1] + rot[2] + rot[3] > E) {
                                                rotate(rDir,rot);
                                                norm(rDir); 
                                                rotate(rPos,rot);
                                            }
                                            rPos[0]+=boxPos[0];rPos[1]+=boxPos[1];rPos[2]+=boxPos[2];

                                            int sign[3] = {rDir[0] < 0, rDir[1] < 0, rDir[2] < 0};
                                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                                            bounds[0][0] =  sConstants.objAttributes[aInd + 3];
                                            bounds[0][1] =  sConstants.objAttributes[aInd + 4];
                                            bounds[0][2] =  sConstants.objAttributes[aInd + 5];
                                            bounds[1][0] =  sConstants.objAttributes[aInd + 6];
                                            bounds[1][1] =  sConstants.objAttributes[aInd + 7];
                                            bounds[1][2] =  sConstants.objAttributes[aInd + 8];
                                            float tmin = (bounds[sign[0]][0] - rPos[0]) / rDir[0];
                                            float tmax = (bounds[1 - sign[0]][0] - rPos[0]) / rDir[0];
                                            float tymin = (bounds[sign[1]][1] - rPos[1]) / rDir[1];
                                            float tymax = (bounds[1 - sign[1]][1] - rPos[1]) / rDir[1];
                                            if ((tmin > tymax) || (tymin > tmax))
                                                continue;
                                            if (tymin > tmin)
                                                tmin = tymin;
                                            if (tymax < tmax)
                                                tmax = tymax;
                                            float tzmin = (bounds[sign[2]][2] - rPos[2]) / rDir[2];
                                            float tzmax = (bounds[1 - sign[2]][2] - rPos[2]) / rDir[2];
                                            if ((tmin > tzmax) || (tzmin > tmax))
                                                continue;
                                            if (tzmin > tmin)
                                                tmin = tzmin;
                                            if (tzmax < tmax)
                                                tmax = tzmax;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = tmin > E ? tmin : (tmax > E ? tmax : INFINITY); 

                                            if ((int)sConstants.matList[sMatInd][5]==3) {
                                                obbPosHit[0] = rPos[0] + rDir[0]*tempT;
                                                obbPosHit[1] = rPos[1] + rDir[1]*tempT;
                                                obbPosHit[2] = rPos[2] + rDir[2]*tempT;
                                            }  
                                        }
                                        // sphere
                                        else if (shapeType == 0) {
                                            float L[3];
                                            L[0] =  sConstants.objAttributes[aInd+0] - posHit[0];
                                            L[1] =  sConstants.objAttributes[aInd+1] - posHit[1];
                                            L[2] =  sConstants.objAttributes[aInd+2] - posHit[2];
                                            float tca = L[0]*shadowDir[0] + L[1]*shadowDir[1] + L[2]*shadowDir[2];
                                            if (tca < E)
                                                continue;
                                            float dsq = dot(L,L) - tca * tca;
                                            float radiusSq =  sConstants.objAttributes[aInd+3] *  sConstants.objAttributes[aInd+3];
                                            if (radiusSq - dsq < E)
                                                continue;
                                            float thc = sqrt(radiusSq - dsq);
                                            float t0 = tca - thc;
                                            float t1 = tca + thc;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = t0 > E ? t0 : (t1 > E ? t1 : INFINITY); 
                                            
                                        }
                                        if (tempT < dirLen) {
                                            // Dialectric Check
                                            if ((int)sConstants.matList[sMatInd][5]==3) {
                                                float blur = sConstants.matList[sMatInd][3];
                                                float RI = 1.0f / sConstants.matList[sMatInd][4];
                                                // Get Normal
                                                float refNorm[3] {0.0f, 0.0f, 0.0f};
                                                if (shapeTypeHit == 1) {
                                                    float bounds[2][3] = {{0,0,0}, {0,0,0}};
                                                    bounds[0][0] =  sConstants.objAttributes[attrInd + 3];
                                                    bounds[0][1] =  sConstants.objAttributes[attrInd + 4];
                                                    bounds[0][2] =  sConstants.objAttributes[attrInd + 5];
                                                    bounds[1][0] =  sConstants.objAttributes[attrInd + 6];
                                                    bounds[1][1] =  sConstants.objAttributes[attrInd + 7];
                                                    bounds[1][2] =  sConstants.objAttributes[attrInd + 8];

                                                    // Flat 
                                                    if (fabs(bounds[0][0] - bounds[1][0]) < E) {
                                                        refNorm[0] = shadowDir[0] > 0 ? -1 : 1;
                                                    }
                                                    else if (fabs(bounds[0][1] - bounds[1][1]) < E) {
                                                        refNorm[1] = shadowDir[1] > 0 ? -1 : 1;
                                                    }
                                                    else if (fabs(bounds[0][2] - bounds[1][2]) < E) {
                                                        refNorm[2] = shadowDir[2] > 0 ? -1 : 1;
                                                    }
                                                    // Not Flat
                                                    else if (fabs(obbPosHit[0] - bounds[0][0]) < E)
                                                        refNorm[0] = -1;
                                                    else if (fabs(obbPosHit[0] - bounds[1][0]) < E)
                                                        refNorm[0] = 1;
                                                    else if (fabs(obbPosHit[1] - bounds[0][1]) < E)
                                                        refNorm[1] = -1;
                                                    else if (fabs(obbPosHit[1] - bounds[1][1]) < E)
                                                        refNorm[1] = 1;
                                                    else if (fabs(obbPosHit[2] - bounds[0][0]) < E)
                                                        refNorm[2] = -1;
                                                    else if (fabs(obbPosHit[2] - bounds[1][0]) < E)
                                                        refNorm[2] = 1;

                                                    // Transform Normal
                                                    float rot[4] = {sConstants.objAttributes[attrInd + 9], -sConstants.objAttributes[attrInd + 10], -sConstants.objAttributes[attrInd + 11], -sConstants.objAttributes[attrInd + 12]};
                                                    rotate(refNorm, rot);
                                                    norm(refNorm);
                                                }
                                                else if (shapeTypeHit == 0) {
                                                    float sPosHit[3];
                                                    sPosHit[0] = posHit[0] + shadowDir[0]*tempT;
                                                    sPosHit[1] = posHit[1] + shadowDir[1]*tempT;
                                                    sPosHit[2] = posHit[2] + shadowDir[2]*tempT;
                                                    refNorm[0] = sPosHit[0] -  sConstants.objAttributes[attrInd + 0];
                                                    refNorm[1] = sPosHit[1] -  sConstants.objAttributes[attrInd + 1];
                                                    refNorm[2] = sPosHit[2] -  sConstants.objAttributes[attrInd + 2];
                                                    norm(refNorm);
                                                } 
                                                float cosi = dot(shadowDir, refNorm);
                                                // If normal is same direction as ray, then flip
                                                if (cosi > 0) {
                                                    refNorm[0]*=-1.0f;refNorm[1]*=-1.0f;refNorm[2]*=-1.0f;
                                                    RI = 1.0f / RI;
                                                }
                                                else {
                                                    cosi*=-1.0f;
                                                }

                                                // Can refract check
                                                float sinSq = RI*RI*(1.0f-cosi*cosi);
                                                bool canRefract = 1.0f - sinSq > E;

                                                if (!canRefract) {
                                                    shadowDir[0] = (shadowDir[0] - 2.0f*cosi*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                                    shadowDir[1] = (shadowDir[1] - 2.0f*cosi*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                                    shadowDir[2] = (shadowDir[2] - 2.0f*cosi*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                                    norm(shadowDir);
                                                }
                                                else {

                                                    float refCalc = RI*cosi - (float)sqrt(1.0f-sinSq);
                                                    shadowDir[0] = (RI*shadowDir[0] + refCalc*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                                    shadowDir[1] = (RI*shadowDir[1] + refCalc*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                                    shadowDir[2] = (RI*shadowDir[2] + refCalc*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                                    norm(shadowDir);					
                                                }
                                                continue;
                                            }                                           
                                            shadowRayHit = true;
                                            break;
                                        }
                                    }

                                    if (!shadowRayHit) {
                                        float cosine = fabs(dot(directDir, randDir));
                                        if (cosine > 0.01f) {
                                            shadowRays[pos]=1; 
                                            dir[0] = directDir[0];
                                            dir[1] = directDir[1];
                                            dir[2] = directDir[2];
                                            p0 = fabs(cosine) / M_PI;
                                        }

                                    }
                                    // Shadow Ray
                                    //
                                    //


                                }
                                // 1
                                float p1 = 0;
                                if ( sConstants.shapes[impShape][0] == 1) {
                                    // AABB pdf val
                                    float areaSum = 0;
                                    float xDiff =  sConstants.objAttributes[impAttrInd+3] -  sConstants.objAttributes[impAttrInd+6];
                                    float yDiff =  sConstants.objAttributes[impAttrInd+4] -  sConstants.objAttributes[impAttrInd+7];
                                    float zDiff =  sConstants.objAttributes[impAttrInd+5] -  sConstants.objAttributes[impAttrInd+8];
                                    areaSum += xDiff * yDiff * 2.0f;
                                    areaSum += zDiff * yDiff * 2.0f;
                                    areaSum += xDiff * zDiff * 2.0f;
                                    float cosine = dot(dir, normals[pos]);
                                    cosine = cosine < 0.0001f ? 0.0001f : cosine;

                                    float diff[3];
                                    diff[0] =  sConstants.objAttributes[impAttrInd+0] - posHit[0];
                                    diff[1] =  sConstants.objAttributes[impAttrInd+1] - posHit[1];
                                    diff[2] =  sConstants.objAttributes[impAttrInd+2] - posHit[2];
                                    float dirLen = sqrt(dot(diff, diff));


                                    // AABB needs magic number for pdf calc, TODO: LOOK INTO, was too bright before
                                    //p1 = 1 / (cosine * areaSum);
                                    p1 = dirLen / (cosine * areaSum);

                                } else if ( sConstants.shapes[impShape][0] == 0) {
                                    // Sphere pdf val
                                    float diff[3] = { sConstants.objAttributes[impAttrInd+0]-posHit[0], 
                                                      sConstants.objAttributes[impAttrInd+1]-posHit[1], 
                                                      sConstants.objAttributes[impAttrInd+2]-posHit[2]};
                                    auto distance_squared = dot(diff, diff);
                                    auto cos_theta_max = sqrt(1.0f -  sConstants.objAttributes[impAttrInd+3] *  sConstants.objAttributes[impAttrInd+3] / distance_squared);
                                    // NaN check
                                    cos_theta_max = (cos_theta_max != cos_theta_max) ? 0.9999f : cos_theta_max;
                                    auto solid_angle = M_PI * (1.0f - cos_theta_max) *2.0f;

                                    // Sphere needs magic number for pdf calc, TODO: LOOK INTO, was too dark before
                                    //p1 = 1 / (solid_angle );
                                    p1 =  sConstants.objAttributes[impAttrInd+3] / (solid_angle * sqrt(distance_squared)*4.0f);
                                }

                                pdfVals[pos] = 0.5f*p0 + 0.5f*p1;
                            }
                        }
                    }

                    numShapeHit++;
                    rayPositions[pos][0] = posHit[0];
                    rayPositions[pos][1] = posHit[1];
                    rayPositions[pos][2] = posHit[2];
                    rayPositions[pos][3] = shapeHit;

                } else {
                    back_col[0] = 0.1f;
                    back_col[1] = 0.1f;
                    back_col[2] = (dir[1] + 1.0f)/2.2f + 0.1f;
                    break;
                }
            }
        }

        float finalCol[3] = {back_col[0], 
                            back_col[1], 
                            back_col[2]};

        // Reverse through hit points and add up colour
        for (int pos = numShapeHit-1; pos >=0; pos--) {

            int shapeHit = (int)rayPositions[pos][3];
            int matInd = sConstants.shapes[shapeHit][1];

            float albedo[3] = {  sConstants.matList[matInd][0], 
                                 sConstants.matList[matInd][1], 
                                 sConstants.matList[matInd][2]};
            int matType = (int) sConstants.matList[matInd][5];	
        
            float normal[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};

            float newDir[3];
            newDir[0] = pos == numShapeHit-1 ? dir[0] : rayPositions[pos + 1][0] - rayPositions[pos][0];
            newDir[1] = pos == numShapeHit-1 ? dir[1] : rayPositions[pos + 1][1] - rayPositions[pos][1];
            newDir[2] = pos == numShapeHit-1 ? dir[2] : rayPositions[pos + 1][2] - rayPositions[pos][2];
            if (pos < numShapeHit-1) {
                norm(newDir);
            }

            float emittance[3];
            emittance[0] = matType == 1 ? albedo[0] : 0;
            emittance[1] = matType == 1 ? albedo[1] : 0;
            emittance[2] = matType == 1 ? albedo[2] : 0;

            float pdf_val = pdfVals[pos]; 

            float cosine2 = dot(normal, newDir);

            float scattering_pdf = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;// just cosine/pi for lambertian
            float multVecs[3] = {albedo[0]*finalCol[0],   // albedo*incoming 
                                  albedo[1]*finalCol[1],  
                                  albedo[2]*finalCol[2]}; 

            float directLightMult = shadowRays[pos]==1 &&  sConstants.numImportantShapes>1 ?  sConstants.numImportantShapes : 1;

            float pdfs = scattering_pdf / pdf_val;
            finalCol[0] = emittance[0] + multVecs[0] * pdfs * directLightMult;
            finalCol[1] = emittance[1] + multVecs[1] * pdfs * directLightMult;
            finalCol[2] = emittance[2] + multVecs[2] * pdfs * directLightMult;
        }
        ReturnStruct ret;
        ret.xyz[0] = finalCol[0];
        ret.xyz[1] = finalCol[1];
        ret.xyz[2] = finalCol[2];
        if ( sConstants.getDenoiserInf == 1) {
            ret.normal[0] = normals[0][0];
            ret.normal[1] = normals[0][1];
            ret.normal[2] = normals[0][2];
            int matIndAlb1 = sConstants.shapes[(int)rayPositions[0][3]][1];
            ret.albedo1[0] =  sConstants.matList[matIndAlb1][0];
            ret.albedo1[1] =  sConstants.matList[matIndAlb1][1];
            ret.albedo1[2] =  sConstants.matList[matIndAlb1][2];
            int matIndAlb2 = sConstants.shapes[(int)rayPositions[1][3]][1];
            ret.albedo2[0] =  sConstants.matList[matIndAlb2][0];
            ret.albedo2[1] =  sConstants.matList[matIndAlb2][1];
            ret.albedo2[2] =  sConstants.matList[matIndAlb2][2];
            ret.directLight = 0.0f;
            for (int c = 0; c< sConstants.maxDepth; c++)
                ret.directLight += (float)shadowRays[c] / (float) sConstants.maxDepth;
            ret.worldPos[0] = rayPositions[0][0];
            ret.worldPos[1] = rayPositions[0][1];
            ret.worldPos[2] = rayPositions[0][2];
        }
        ret.raysSent = numRays;
        return ret;
    }

    void Renderers::CPURender() {

        ReturnStruct ret;
        int index;
        sampleCount++;
        rayCount = 0;
        RandomSeeds s;
		for (int j = 0; j < yRes; j++) {
			for (int i = 0; i < xRes; i++) {

                // Generate random seeds for each pixel
				uint64_t s0 = GloRandS[0];
				uint64_t s1 = GloRandS[1];
				s1 ^= s0;
				GloRandS[0] = (s0 << 49) | ((s0 >> (64 - 49)) ^ s1 ^ (s1 << 21));
				GloRandS[1] = (s1 << 28) | (s1 >> (64 - 28));
				s.s1 = GloRandS[0];
				s.s2 = GloRandS[1];

                auto skepuInd = skepu::Index2D();
                skepuInd.row = j;
                skepuInd.col = i;
                
				ret = RenderFunc(skepuInd, s, constants);
				index = j*xRes + i;

                preScreen[index] += vec3(ret.xyz[0], ret.xyz[1], ret.xyz[2]);
				rayCount += ret.raysSent;

				// Denoiser info
				if (denoising) {
                    auto info = &denoisingInf[index];
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
                        pow(albedo1[index].x/sampleCount - ret.albedo1[0] ,2),
					    pow(albedo1[index].y/sampleCount - ret.albedo1[1],2),    
					    pow(albedo1[index].z/sampleCount - ret.albedo1[2],2));
					info->stdDevVecs[3] += vec3(
                        pow(albedo2[index].x/sampleCount - ret.albedo2[0] ,2),
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
                    info->stdDev[5] = info->stdDevVecs[5][0]/sampleCount;
				}
			}
		}  
    }
    void Renderers::OMPRender() {
        sampleCount++;
        rayCount = 0;
        #pragma omp parallel for
		for (int j = 0; j < yRes; j++) {
            #pragma omp parallel for
			for (int i = 0; i < xRes; i++) {

                // Generate random seeds for each pixel
                RandomSeeds s;
				uint64_t s0 = GloRandS[0];
				uint64_t s1 = GloRandS[1];
				s1 ^= s0;
				GloRandS[0] = (s0 << 49) | ((s0 >> (64 - 49)) ^ s1 ^ (s1 << 21));
				GloRandS[1] = (s1 << 28) | (s1 >> (64 - 28));
				s.s1 = GloRandS[0];
				s.s2 = GloRandS[1];

                auto skepuInd = skepu::Index2D();
                skepuInd.row = j;
                skepuInd.col = i;
                
				ReturnStruct ret = RenderFunc(skepuInd, s, constants);
				int index = j*xRes + i;

                preScreen[index] += vec3(ret.xyz[0], ret.xyz[1], ret.xyz[2]);
				rayCount += ret.raysSent;

				// Denoiser info
				if (denoising) {
                    auto info = &denoisingInf[index];
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
                        pow(albedo1[index].x/sampleCount - ret.albedo1[0] ,2),
					    pow(albedo1[index].y/sampleCount - ret.albedo1[1],2),    
					    pow(albedo1[index].z/sampleCount - ret.albedo1[2],2));
					info->stdDevVecs[3] += vec3(
                        pow(albedo2[index].x/sampleCount - ret.albedo2[0] ,2),
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
                    info->stdDev[5] = info->stdDevVecs[5][0]/sampleCount;
				}
			}
		}  
    }
    void Renderers::CUDARender() {
        CUDARender::render();
    }
    void Renderers::OpenGLRender() {

    }
    
struct skepu_userfunction_skepu_skel_1renderFunc_RenderFunc
{
constexpr static size_t totalArity = 3;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 1;
constexpr static bool usesPRNG = 0;
constexpr static size_t randomCount = SKEPU_NO_RANDOM;
using IndexType = skepu::Index2D;
using ElwiseArgs = std::tuple<RandomSeeds>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<Constants>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = ReturnStruct;

constexpr static bool prefersMatrix = 1;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ ReturnStruct CU(skepu::Index2D ind, RandomSeeds seeds, Constants sConstants)
{
        // Ray
        float camPos[3] = { sConstants.camPos[0],  sConstants.camPos[1],  sConstants.camPos[2]};
        float rayPos[3] = {camPos[0], camPos[1], camPos[2]};

        // Lambda Functions
            auto dot  = [=](float vec1[3], float vec2[3]) {return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];};
            auto norm = [&](float vec[3]) {auto d = sqrt(dot(vec,vec)); vec[0]/=d; vec[1]/=d; vec[2]/=d; };
            auto randBetween = [](RandomSeeds& seeds, float min, float max) {
                uint64_t s0 = seeds.s1;
                uint64_t s1 = seeds.s2;
                uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                double one_two = ((uint64_t)1 << 63) * (double)2.0;
                float rand = xorshiro / one_two;
                s1 ^= s0;
                seeds.s1 = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                seeds.s2 = (s1 << 28) | (s1 >> 36);

                rand *= max - min;
                rand += min;
                return rand;
            };
            auto QMult = [&](float q1[4], float q2[4] ) {
                auto A1 = (q1[3] + q1[1]) * (q2[1] + q2[2]);
                auto A3 = (q1[0] - q1[2]) * (q2[0] + q2[3]);
                auto A4 = (q1[0] + q1[2]) * (q2[0] - q2[3]);
                auto A2 = A1 + A3 + A4;
                auto A5 = (q1[3] - q1[1]) * (q2[1] - q2[2]);
                A5 = (A5 + A2) / 2.0f;

                auto Q1 = A5 - A1 + (q1[3] - q1[2]) * (q2[2] - q2[3]);
                auto Q2 = A5 - A2 + (q1[1] + q1[0]) * (q2[1] + q2[0]);
                auto Q3 = A5 - A3 + (q1[0] - q1[1]) * (q2[2] + q2[3]);
                auto Q4 = A5 - A4 + (q1[3] + q1[2]) * (q2[0] - q2[1]);

                q1[0] = Q1; q1[1] = Q2; q1[2] = Q3; q1[3] = Q4;
            };
            auto rotate = [&](float to_rotate[3], float q[4]) {

                float p[4]  {0, to_rotate[0], to_rotate[1], to_rotate[2]};
                float qR[4] {q[0],-q[1],-q[2],-q[3]};

                QMult(p, q);
                QMult(qR, p);
                to_rotate[0]=qR[1];to_rotate[1]=qR[2];to_rotate[2]=qR[3];
            };
        // Lambda Functions

        // Rand Samp
        float rSamps[2] = {0.0f, 0.0f};
        if (sConstants.randSamp>0.001f) {
            rSamps[0] = randBetween( seeds, -1, 1) * sConstants.randSamp;
            rSamps[1] = randBetween( seeds, -1, 1) * sConstants.randSamp;
        }

        float back_col[3] = { 0,0,0};

        // Pixel Coord
        float camForward[3] = { sConstants.camForward[0],  sConstants.camForward[1],  sConstants.camForward[2]};
        float camRight[3] = { sConstants.camRight[0],  sConstants.camRight[1],  sConstants.camRight[2]};

        float pY = - sConstants.maxAngleV + 2.0f* sConstants.maxAngleV*((float)ind.row/(float) sConstants.RESV);
        float pX = - sConstants.maxAngleH + 2.0f* sConstants.maxAngleH*((float)ind.col/(float) sConstants.RESH);

        float pix[3] = {0,0,0};
        pix[0] = camPos[0] +  sConstants.camForward[0]* sConstants.focalLength +  sConstants.camRight[0]*(pX+rSamps[0]) +  sConstants.camUp[0]*(pY+rSamps[1]);
        pix[1] = camPos[1] +  sConstants.camForward[1]* sConstants.focalLength +  sConstants.camRight[1]*(pX+rSamps[0]) +  sConstants.camUp[1]*(pY+rSamps[1]);
        pix[2] = camPos[2] +  sConstants.camForward[2]* sConstants.focalLength +  sConstants.camRight[2]*(pX+rSamps[0]) +  sConstants.camUp[2]*(pY+rSamps[1]);

        float rayDir[3] = {pix[0]-camPos[0], pix[1]-camPos[1], pix[2]-camPos[2]};
        norm(rayDir);
 
        // Store ray collisions and reverse through them (last num is shape index)
        float rayPositions[12][4];
        float normals[12][3];
        float pdfVals[12];
        // Shadow Rays: counts succesful shadow rays i.e. direct lighting, done for each bounce to provide more info
        int shadowRays[12];
        for (int v=0; v<12; v++){
            normals[v][0]=0.0f;normals[v][1]=0.0f;normals[v][2]=0.0f;
            pdfVals[v] = 1.0f / M_PI;
            shadowRays[v] = 0;
        }

        int numShapeHit = 0;
        int numRays = 0;
        float dir[3] = {rayDir[0], rayDir[1], rayDir[2]};
        for (int pos = 0; pos <  sConstants.maxDepth; pos++) {
            numRays++;
            int shapeHit;
            float prevPos[3];
            if (pos > 0) {
                prevPos[0] = rayPositions[pos-1][0];
                prevPos[1] = rayPositions[pos-1][1];
                prevPos[2] = rayPositions[pos-1][2];
            } else {
                prevPos[0] = camPos[0];
                prevPos[1] = camPos[1];
                prevPos[2] = camPos[2];
            }
            float posHit[3] {0.0f, 0.0f, 0.0f};
            float OBBSpacePosHit[3] {0.0f, 0.0f, 0.0f};
            bool hitAnything = false;
            int shapeTypeHit, attrInd, matInd;
            // Collide with shapes, generating new dirs as needed (i.e. random or specular)
            {

                float E = 0.00001f;

                // Find shape
                {
                    float t = INFINITY;
                    for (int ind = 0; ind <  sConstants.numShapes; ind++) {
                        int shapeType = sConstants.shapes[ind][0];
                        int aInd = sConstants.shapes[ind][2];
                        float tempT = INFINITY;
                        // ----- intersect shapes -----
                        // aabb
                        if ( shapeType == 1) {

                            // Transform Ray
                            float rDir[3] = {dir[0], dir[1], dir[2]};
                            float boxPos[3] = {sConstants.objAttributes[aInd + 0], sConstants.objAttributes[aInd + 1], sConstants.objAttributes[aInd + 2]};
                            float rPos[3] = {prevPos[0]-boxPos[0], prevPos[1]-boxPos[1], prevPos[2]-boxPos[2]};
                            float rot[4] = {sConstants.objAttributes[aInd + 9], sConstants.objAttributes[aInd + 10], sConstants.objAttributes[aInd + 11], sConstants.objAttributes[aInd + 12]};
                            if (rot[1] + rot[2] + rot[3] > E) {
                                rotate(rDir,rot);
                                norm(rDir); 
                                rotate(rPos,rot);
                            }
                            rPos[0]+=boxPos[0];rPos[1]+=boxPos[1];rPos[2]+=boxPos[2];

                            int sign[3] = {rDir[0] < 0, rDir[1] < 0, rDir[2] < 0};
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] =  sConstants.objAttributes[aInd + 3];
                            bounds[0][1] =  sConstants.objAttributes[aInd + 4];
                            bounds[0][2] =  sConstants.objAttributes[aInd + 5];
                            bounds[1][0] =  sConstants.objAttributes[aInd + 6];
                            bounds[1][1] =  sConstants.objAttributes[aInd + 7];
                            bounds[1][2] =  sConstants.objAttributes[aInd + 8];
                            float tmin = (bounds[sign[0]][0] - rPos[0]) / rDir[0];
                            float tmax = (bounds[1 - sign[0]][0] - rPos[0]) / rDir[0];
                            float tymin = (bounds[sign[1]][1] - rPos[1]) / rDir[1];
                            float tymax = (bounds[1 - sign[1]][1] - rPos[1]) / rDir[1];
                            if ((tmin > tymax) || (tymin > tmax))
                                continue;
                            if (tymin > tmin)
                                tmin = tymin;
                            if (tymax < tmax)
                                tmax = tymax;
                            float tzmin = (bounds[sign[2]][2] - rPos[2]) / rDir[2];
                            float tzmax = (bounds[1 - sign[2]][2] - rPos[2]) / rDir[2];
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
                                OBBSpacePosHit[0] = rPos[0] + rDir[0]*tempT;
                                OBBSpacePosHit[1] = rPos[1] + rDir[1]*tempT;
                                OBBSpacePosHit[2] = rPos[2] + rDir[2]*tempT;
                            }
                        }
                        // sphere
                        else if (shapeType == 0) {
                            float L[3] = {0,0,0};
                            L[0] =  sConstants.objAttributes[aInd + 0] - prevPos[0];
                            L[1] =  sConstants.objAttributes[aInd + 1] - prevPos[1];
                            L[2] =  sConstants.objAttributes[aInd + 2] - prevPos[2];
                            float tca = dot(L, dir);
                            if (tca < E)
                                continue;
                            float dsq = dot(L,L) - tca * tca;
                            float radiusSq =  sConstants.objAttributes[aInd + 3] *  sConstants.objAttributes[aInd + 3];
                            if (radiusSq - dsq < E)
                                continue;
                            float thc = sqrt(radiusSq - dsq);
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
                            posHit[0] = prevPos[0] + dir[0]*t;
                            posHit[1] = prevPos[1] + dir[1]*t;
                            posHit[2] = prevPos[2] + dir[2]*t;
                            shapeHit = ind;
                            attrInd = aInd;
                            matInd = sConstants.shapes[shapeHit][1];
                            shapeTypeHit = shapeType;
                        }
                    }
                }

                if (hitAnything) {

                    // Get Normal
                    {
                        if (shapeTypeHit == 1) {
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] =  sConstants.objAttributes[attrInd + 3];
                            bounds[0][1] =  sConstants.objAttributes[attrInd + 4];
                            bounds[0][2] =  sConstants.objAttributes[attrInd + 5];
                            bounds[1][0] =  sConstants.objAttributes[attrInd + 6];
                            bounds[1][1] =  sConstants.objAttributes[attrInd + 7];
                            bounds[1][2] =  sConstants.objAttributes[attrInd + 8];
                            normals[pos][0] = 0;
                            normals[pos][1] = 0;
                            normals[pos][2] = 0;

                            // Flat 
                            if (fabs(bounds[0][0] - bounds[1][0]) < E) {
                                normals[pos][0] = dir[0] > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0][1] - bounds[1][1]) < E) {
                                normals[pos][1] = dir[1] > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0][2] - bounds[1][2]) < E) {
                                normals[pos][2] = dir[2] > E ? -1 : 1;
                            }
                            // Not Flat
                            else if (fabs(OBBSpacePosHit[0] - bounds[0][0]) < E)
                                normals[pos][0] = -1;
                            else if (fabs(OBBSpacePosHit[0] - bounds[1][0]) < E)
                                normals[pos][0] = 1;
                            else if (fabs(OBBSpacePosHit[1] - bounds[0][1]) < E)
                                normals[pos][1] = -1;
                            else if (fabs(OBBSpacePosHit[1] - bounds[1][1]) < E)
                                normals[pos][1] = 1;
                            else if (fabs(OBBSpacePosHit[2] - bounds[0][2]) < E)
                                normals[pos][2] = -1;
                            else if (fabs(OBBSpacePosHit[2] - bounds[1][2]) < E)
                                normals[pos][2] = 1;

                            // Transform Normal
                            float rot[4] = {sConstants.objAttributes[attrInd + 9], -sConstants.objAttributes[attrInd + 10], -sConstants.objAttributes[attrInd + 11], -sConstants.objAttributes[attrInd + 12]};
                            rotate(normals[pos], rot);
                            norm(normals[pos]);
                        }
                        else if (shapeTypeHit == 0) {
                            normals[pos][0] = posHit[0] -  sConstants.objAttributes[attrInd + 0];
                            normals[pos][1] = posHit[1] -  sConstants.objAttributes[attrInd + 1];
                            normals[pos][2] = posHit[2] -  sConstants.objAttributes[attrInd + 2];
                            norm(normals[pos]);
                        }
                    }
                
                    // Gen new dirs and pdfs
                    {

                        // Random Dir Generation
                            float randDir[3];
                            float rands[5];
                            {
                                // Rand vals
                                for (int n = 0; n < 5; n++) 
                                    rands[n] = randBetween( seeds, 0,1);

                                float axis[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
                                // 2
                                // axis[2] = normal
                                axis[2][0] = normals[pos][0];
                                axis[2][1] = normals[pos][1];
                                axis[2][2] = normals[pos][2];
                                // 1
                                if (fabs(axis[2][0]) > 0.9) {
                                    // axis[1] = cross(axis[2], [0,1,0])
                                    axis[1][0] = -axis[2][2];
                                    axis[1][2] =  axis[2][0]; 
                                }
                                else {
                                    // axis[1] = cross(axis[2], [1,0,0])
                                    axis[1][1] =  axis[2][2];
                                    axis[1][2] = -axis[2][1];
                                }
                                norm(axis[1]);
                                // 0
                                // axis[0] = cross(axis[2], axis[1])
                                axis[0][0] = axis[2][1]*axis[1][2] - axis[2][2]*axis[1][1];
                                axis[0][1] = axis[2][2]*axis[1][0] - axis[2][0]*axis[1][2];
                                axis[0][2] = axis[2][0]*axis[1][1] - axis[2][1]*axis[1][0];
                                // rand dir
                                float phi = 2.0f * M_PI * rands[0];
                                float x = cos(phi) * sqrt(rands[1]);
                                float y = sin(phi) * sqrt(rands[1]);
                                float z = sqrt(1.0f - rands[1]);

                                randDir[0] = x * axis[0][0] + y * axis[1][0] + z * axis[2][0];
                                randDir[1] = x * axis[0][1] + y * axis[1][1] + z * axis[2][1];
                                randDir[2] = x * axis[0][2] + y * axis[1][2] + z * axis[2][2];
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
                        if ( sConstants.matList[matInd][5] == 3) {
                            // Dielectric Material
                            shadowRays[pos] = 1;
                            float blur =  sConstants.matList[matInd][3];
                            float RI = 1.0f /  sConstants.matList[matInd][4];
                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float refNorm[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};
                            float cosi = dot(dirIn, refNorm);

                            // If normal is same direction as ray, then flip
                            if (cosi > 0) {
                                refNorm[0]*=-1.0f;refNorm[1]*=-1.0f;refNorm[2]*=-1.0f;
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
                            float schlick = r0 + (1.0f - r0) * (float)pow((1.0f - cosi), 5.0f);

                            float schlickRand = randBetween(seeds, 0, 1);

                            if (!canRefract || schlick > schlickRand) {
                                dir[0] = (dirIn[0] - 2.0f*cosi*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                dir[1] = (dirIn[1] - 2.0f*cosi*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                dir[2] = (dirIn[2] - 2.0f*cosi*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                norm(dir);
                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                float cosine2 = dot(normals[pos], dir);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;
                            }
                            else {

                                float refCalc = RI*cosi - (float)sqrt(1.0f-sinSq);
                                dir[0] = (RI*dirIn[0] + refCalc*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                dir[1] = (RI*dirIn[1] + refCalc*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                dir[2] = (RI*dirIn[2] + refCalc*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                norm(dir);

                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                // Danger here, scattering pdf was going to 0 for refracting and making colour explode
                                float cosine2 = dot(normals[pos], dir);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;					
                            }
                        }
                        else if ( sConstants.matList[matInd][5] == 2) {
                            // Metal material
                            shadowRays[pos] = 1;

                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float blur =  sConstants.matList[matInd][3];

                            float prevDirNormalDot = dot(dirIn, normals[pos]);

                            dir[0] = (dirIn[0] - 2.0f*prevDirNormalDot*normals[pos][0])*(1.0f - blur) + blur*randDir[0];
                            dir[1] = (dirIn[1] - 2.0f*prevDirNormalDot*normals[pos][1])*(1.0f - blur) + blur*randDir[1];
                            dir[2] = (dirIn[2] - 2.0f*prevDirNormalDot*normals[pos][2])*(1.0f - blur) + blur*randDir[2];

                            float cosine2 = dot(dir, normals[pos]);

                            // Same as scattering pdf, to make pdf 1, as it is specular
                            pdfVals[pos] = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;
                        }
                        else {
                            // Lambertian/Light Material

                            dir[0] = randDir[0];
                            dir[1] = randDir[1];
                            dir[2] = randDir[2];

                            // Check Light Mat
                            bool mixPdf;
                            int impInd, impShape;
                            if ( sConstants.matList[matInd][5] == 1) {
                                shadowRays[pos] = 1;
                                mixPdf = false;
                            } else {
                                // Get importance shape
                                mixPdf =  sConstants.numImportantShapes > 0;

                                if (mixPdf) { 
                                    impInd = rands[3] *  sConstants.numImportantShapes * 0.99999f;
                                    impShape =  sConstants.importantShapes[impInd];
                                    if (impShape==shapeHit) {
                                        mixPdf = false;
                                        // mixPdf =  sConstants.numImportantShapes > 1;
                                        // impInd = (impInd+1) %  sConstants.numImportantShapes;
                                        // impShape =  sConstants.importantShapes[impInd];
                                    } 
                                }
                            }

                            // Calculate PDF val
                            if (mixPdf) {
                                // 0
                                float p0 = 1 / M_PI;
                                int choosePdf = rands[4] > 0.65f ? 1 : 0;
                                int impAttrInd = sConstants.shapes[impShape][2];
                                // dot(randomly generated dir, ray dir) / PI
                                if (choosePdf == 1) {
                                    // Generate dir towards importance shape
                                    float randPos[3] = {0,0,0};
                                    if ( sConstants.shapes[impShape][0] == 1) {
                                        // Gen three new random variables : [0, 1]
                                        float aabbRands[3];
                                        for (int n = 0; n < 3; n++) 
                                            aabbRands[n] = randBetween( seeds, 0,1);
                                        randPos[0] = (1.0f - aabbRands[0])* sConstants.objAttributes[impAttrInd+3] + aabbRands[0]* sConstants.objAttributes[impAttrInd+6];
                                        randPos[1] = (1.0f - aabbRands[1])* sConstants.objAttributes[impAttrInd+4] + aabbRands[1]* sConstants.objAttributes[impAttrInd+7];
                                        randPos[2] = (1.0f - aabbRands[2])* sConstants.objAttributes[impAttrInd+5] + aabbRands[2]* sConstants.objAttributes[impAttrInd+8];	
                                    } 
                                    else if ( sConstants.shapes[impShape][0] == 0) {
                                        // Gen three new random variables : [-1, 1]
                                        float sphereRands[3];
                                        sphereRands[0] = randBetween( seeds, -1,1);
                                        sphereRands[1] = randBetween( seeds, -1,1);
                                        sphereRands[2] = randBetween( seeds, -1,1);
                                        norm(sphereRands);
                                        
                                        randPos[0] =  sConstants.objAttributes[impAttrInd+0] + sphereRands[0]* sConstants.objAttributes[impAttrInd+3];
                                        randPos[1] =  sConstants.objAttributes[impAttrInd+1] + sphereRands[1]* sConstants.objAttributes[impAttrInd+3];
                                        randPos[2] =  sConstants.objAttributes[impAttrInd+2] + sphereRands[2]* sConstants.objAttributes[impAttrInd+3];
                                    }

                                    float directDir[3];
                                    directDir[0] = randPos[0] - posHit[0];
                                    directDir[1] = randPos[1] - posHit[1];
                                    directDir[2] = randPos[2] - posHit[2];
                                    float dirLen = sqrt(dot(directDir, directDir));
                                    directDir[0] /= dirLen; directDir[1] /= dirLen; directDir[2] /= dirLen;  

                                    //
                                    // Shadow Ray
                                    // Need to send shadow ray to see if point is in path of direct light
                                    bool shadowRayHit = false;
                                    float shadowDir[3] {directDir[0], directDir[1], directDir[2]};
                                    for (int ind = 0; ind <  sConstants.numShapes; ind++) {
                                        if (ind == impShape)
                                            continue;
                                        int shapeType = sConstants.shapes[ind][0];
                                        int sMatInd = sConstants.shapes[ind][1];
                                        int aInd = sConstants.shapes[ind][2];
                                        float tempT = INFINITY;
                                        float obbPosHit[3];
                                        // ----- intersect shapes -----
                                        // aabb
                                        if ( shapeType == 1) {
                                            // Transform Ray
                                            float rDir[3] = {shadowDir[0], shadowDir[1], shadowDir[2]};
                                            float boxPos[3] = {sConstants.objAttributes[aInd + 0], sConstants.objAttributes[aInd + 1], sConstants.objAttributes[aInd + 2]};
                                            float rPos[3] = {posHit[0]-boxPos[0], posHit[1]-boxPos[1], posHit[2]-boxPos[2]};
                                            float rot[4] = {sConstants.objAttributes[aInd + 9], sConstants.objAttributes[aInd + 10], sConstants.objAttributes[aInd + 11], sConstants.objAttributes[aInd + 12]};
                                            if (rot[1] + rot[2] + rot[3] > E) {
                                                rotate(rDir,rot);
                                                norm(rDir); 
                                                rotate(rPos,rot);
                                            }
                                            rPos[0]+=boxPos[0];rPos[1]+=boxPos[1];rPos[2]+=boxPos[2];

                                            int sign[3] = {rDir[0] < 0, rDir[1] < 0, rDir[2] < 0};
                                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                                            bounds[0][0] =  sConstants.objAttributes[aInd + 3];
                                            bounds[0][1] =  sConstants.objAttributes[aInd + 4];
                                            bounds[0][2] =  sConstants.objAttributes[aInd + 5];
                                            bounds[1][0] =  sConstants.objAttributes[aInd + 6];
                                            bounds[1][1] =  sConstants.objAttributes[aInd + 7];
                                            bounds[1][2] =  sConstants.objAttributes[aInd + 8];
                                            float tmin = (bounds[sign[0]][0] - rPos[0]) / rDir[0];
                                            float tmax = (bounds[1 - sign[0]][0] - rPos[0]) / rDir[0];
                                            float tymin = (bounds[sign[1]][1] - rPos[1]) / rDir[1];
                                            float tymax = (bounds[1 - sign[1]][1] - rPos[1]) / rDir[1];
                                            if ((tmin > tymax) || (tymin > tmax))
                                                continue;
                                            if (tymin > tmin)
                                                tmin = tymin;
                                            if (tymax < tmax)
                                                tmax = tymax;
                                            float tzmin = (bounds[sign[2]][2] - rPos[2]) / rDir[2];
                                            float tzmax = (bounds[1 - sign[2]][2] - rPos[2]) / rDir[2];
                                            if ((tmin > tzmax) || (tzmin > tmax))
                                                continue;
                                            if (tzmin > tmin)
                                                tmin = tzmin;
                                            if (tzmax < tmax)
                                                tmax = tzmax;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = tmin > E ? tmin : (tmax > E ? tmax : INFINITY); 

                                            if ((int)sConstants.matList[sMatInd][5]==3) {
                                                obbPosHit[0] = rPos[0] + rDir[0]*tempT;
                                                obbPosHit[1] = rPos[1] + rDir[1]*tempT;
                                                obbPosHit[2] = rPos[2] + rDir[2]*tempT;
                                            }  
                                        }
                                        // sphere
                                        else if (shapeType == 0) {
                                            float L[3];
                                            L[0] =  sConstants.objAttributes[aInd+0] - posHit[0];
                                            L[1] =  sConstants.objAttributes[aInd+1] - posHit[1];
                                            L[2] =  sConstants.objAttributes[aInd+2] - posHit[2];
                                            float tca = L[0]*shadowDir[0] + L[1]*shadowDir[1] + L[2]*shadowDir[2];
                                            if (tca < E)
                                                continue;
                                            float dsq = dot(L,L) - tca * tca;
                                            float radiusSq =  sConstants.objAttributes[aInd+3] *  sConstants.objAttributes[aInd+3];
                                            if (radiusSq - dsq < E)
                                                continue;
                                            float thc = sqrt(radiusSq - dsq);
                                            float t0 = tca - thc;
                                            float t1 = tca + thc;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = t0 > E ? t0 : (t1 > E ? t1 : INFINITY); 
                                            
                                        }
                                        if (tempT < dirLen) {
                                            // Dialectric Check
                                            if ((int)sConstants.matList[sMatInd][5]==3) {
                                                float blur = sConstants.matList[sMatInd][3];
                                                float RI = 1.0f / sConstants.matList[sMatInd][4];
                                                // Get Normal
                                                float refNorm[3] {0.0f, 0.0f, 0.0f};
                                                if (shapeTypeHit == 1) {
                                                    float bounds[2][3] = {{0,0,0}, {0,0,0}};
                                                    bounds[0][0] =  sConstants.objAttributes[attrInd + 3];
                                                    bounds[0][1] =  sConstants.objAttributes[attrInd + 4];
                                                    bounds[0][2] =  sConstants.objAttributes[attrInd + 5];
                                                    bounds[1][0] =  sConstants.objAttributes[attrInd + 6];
                                                    bounds[1][1] =  sConstants.objAttributes[attrInd + 7];
                                                    bounds[1][2] =  sConstants.objAttributes[attrInd + 8];

                                                    // Flat 
                                                    if (fabs(bounds[0][0] - bounds[1][0]) < E) {
                                                        refNorm[0] = shadowDir[0] > 0 ? -1 : 1;
                                                    }
                                                    else if (fabs(bounds[0][1] - bounds[1][1]) < E) {
                                                        refNorm[1] = shadowDir[1] > 0 ? -1 : 1;
                                                    }
                                                    else if (fabs(bounds[0][2] - bounds[1][2]) < E) {
                                                        refNorm[2] = shadowDir[2] > 0 ? -1 : 1;
                                                    }
                                                    // Not Flat
                                                    else if (fabs(obbPosHit[0] - bounds[0][0]) < E)
                                                        refNorm[0] = -1;
                                                    else if (fabs(obbPosHit[0] - bounds[1][0]) < E)
                                                        refNorm[0] = 1;
                                                    else if (fabs(obbPosHit[1] - bounds[0][1]) < E)
                                                        refNorm[1] = -1;
                                                    else if (fabs(obbPosHit[1] - bounds[1][1]) < E)
                                                        refNorm[1] = 1;
                                                    else if (fabs(obbPosHit[2] - bounds[0][0]) < E)
                                                        refNorm[2] = -1;
                                                    else if (fabs(obbPosHit[2] - bounds[1][0]) < E)
                                                        refNorm[2] = 1;

                                                    // Transform Normal
                                                    float rot[4] = {sConstants.objAttributes[attrInd + 9], -sConstants.objAttributes[attrInd + 10], -sConstants.objAttributes[attrInd + 11], -sConstants.objAttributes[attrInd + 12]};
                                                    rotate(refNorm, rot);
                                                    norm(refNorm);
                                                }
                                                else if (shapeTypeHit == 0) {
                                                    float sPosHit[3];
                                                    sPosHit[0] = posHit[0] + shadowDir[0]*tempT;
                                                    sPosHit[1] = posHit[1] + shadowDir[1]*tempT;
                                                    sPosHit[2] = posHit[2] + shadowDir[2]*tempT;
                                                    refNorm[0] = sPosHit[0] -  sConstants.objAttributes[attrInd + 0];
                                                    refNorm[1] = sPosHit[1] -  sConstants.objAttributes[attrInd + 1];
                                                    refNorm[2] = sPosHit[2] -  sConstants.objAttributes[attrInd + 2];
                                                    norm(refNorm);
                                                } 
                                                float cosi = dot(shadowDir, refNorm);
                                                // If normal is same direction as ray, then flip
                                                if (cosi > 0) {
                                                    refNorm[0]*=-1.0f;refNorm[1]*=-1.0f;refNorm[2]*=-1.0f;
                                                    RI = 1.0f / RI;
                                                }
                                                else {
                                                    cosi*=-1.0f;
                                                }

                                                // Can refract check
                                                float sinSq = RI*RI*(1.0f-cosi*cosi);
                                                bool canRefract = 1.0f - sinSq > E;

                                                if (!canRefract) {
                                                    shadowDir[0] = (shadowDir[0] - 2.0f*cosi*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                                    shadowDir[1] = (shadowDir[1] - 2.0f*cosi*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                                    shadowDir[2] = (shadowDir[2] - 2.0f*cosi*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                                    norm(shadowDir);
                                                }
                                                else {

                                                    float refCalc = RI*cosi - (float)sqrt(1.0f-sinSq);
                                                    shadowDir[0] = (RI*shadowDir[0] + refCalc*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                                    shadowDir[1] = (RI*shadowDir[1] + refCalc*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                                    shadowDir[2] = (RI*shadowDir[2] + refCalc*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                                    norm(shadowDir);					
                                                }
                                                continue;
                                            }                                           
                                            shadowRayHit = true;
                                            break;
                                        }
                                    }

                                    if (!shadowRayHit) {
                                        float cosine = fabs(dot(directDir, randDir));
                                        if (cosine > 0.01f) {
                                            shadowRays[pos]=1; 
                                            dir[0] = directDir[0];
                                            dir[1] = directDir[1];
                                            dir[2] = directDir[2];
                                            p0 = fabs(cosine) / M_PI;
                                        }

                                    }
                                    // Shadow Ray
                                    //
                                    //


                                }
                                // 1
                                float p1 = 0;
                                if ( sConstants.shapes[impShape][0] == 1) {
                                    // AABB pdf val
                                    float areaSum = 0;
                                    float xDiff =  sConstants.objAttributes[impAttrInd+3] -  sConstants.objAttributes[impAttrInd+6];
                                    float yDiff =  sConstants.objAttributes[impAttrInd+4] -  sConstants.objAttributes[impAttrInd+7];
                                    float zDiff =  sConstants.objAttributes[impAttrInd+5] -  sConstants.objAttributes[impAttrInd+8];
                                    areaSum += xDiff * yDiff * 2.0f;
                                    areaSum += zDiff * yDiff * 2.0f;
                                    areaSum += xDiff * zDiff * 2.0f;
                                    float cosine = dot(dir, normals[pos]);
                                    cosine = cosine < 0.0001f ? 0.0001f : cosine;

                                    float diff[3];
                                    diff[0] =  sConstants.objAttributes[impAttrInd+0] - posHit[0];
                                    diff[1] =  sConstants.objAttributes[impAttrInd+1] - posHit[1];
                                    diff[2] =  sConstants.objAttributes[impAttrInd+2] - posHit[2];
                                    float dirLen = sqrt(dot(diff, diff));


                                    // AABB needs magic number for pdf calc, TODO: LOOK INTO, was too bright before
                                    //p1 = 1 / (cosine * areaSum);
                                    p1 = dirLen / (cosine * areaSum);

                                } else if ( sConstants.shapes[impShape][0] == 0) {
                                    // Sphere pdf val
                                    float diff[3] = { sConstants.objAttributes[impAttrInd+0]-posHit[0], 
                                                      sConstants.objAttributes[impAttrInd+1]-posHit[1], 
                                                      sConstants.objAttributes[impAttrInd+2]-posHit[2]};
                                    auto distance_squared = dot(diff, diff);
                                    auto cos_theta_max = sqrt(1.0f -  sConstants.objAttributes[impAttrInd+3] *  sConstants.objAttributes[impAttrInd+3] / distance_squared);
                                    // NaN check
                                    cos_theta_max = (cos_theta_max != cos_theta_max) ? 0.9999f : cos_theta_max;
                                    auto solid_angle = M_PI * (1.0f - cos_theta_max) *2.0f;

                                    // Sphere needs magic number for pdf calc, TODO: LOOK INTO, was too dark before
                                    //p1 = 1 / (solid_angle );
                                    p1 =  sConstants.objAttributes[impAttrInd+3] / (solid_angle * sqrt(distance_squared)*4.0f);
                                }

                                pdfVals[pos] = 0.5f*p0 + 0.5f*p1;
                            }
                        }
                    }

                    numShapeHit++;
                    rayPositions[pos][0] = posHit[0];
                    rayPositions[pos][1] = posHit[1];
                    rayPositions[pos][2] = posHit[2];
                    rayPositions[pos][3] = shapeHit;

                } else {
                    back_col[0] = 0.1f;
                    back_col[1] = 0.1f;
                    back_col[2] = (dir[1] + 1.0f)/2.2f + 0.1f;
                    break;
                }
            }
        }

        float finalCol[3] = {back_col[0], 
                            back_col[1], 
                            back_col[2]};

        // Reverse through hit points and add up colour
        for (int pos = numShapeHit-1; pos >=0; pos--) {

            int shapeHit = (int)rayPositions[pos][3];
            int matInd = sConstants.shapes[shapeHit][1];

            float albedo[3] = {  sConstants.matList[matInd][0], 
                                 sConstants.matList[matInd][1], 
                                 sConstants.matList[matInd][2]};
            int matType = (int) sConstants.matList[matInd][5];	
        
            float normal[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};

            float newDir[3];
            newDir[0] = pos == numShapeHit-1 ? dir[0] : rayPositions[pos + 1][0] - rayPositions[pos][0];
            newDir[1] = pos == numShapeHit-1 ? dir[1] : rayPositions[pos + 1][1] - rayPositions[pos][1];
            newDir[2] = pos == numShapeHit-1 ? dir[2] : rayPositions[pos + 1][2] - rayPositions[pos][2];
            if (pos < numShapeHit-1) {
                norm(newDir);
            }

            float emittance[3];
            emittance[0] = matType == 1 ? albedo[0] : 0;
            emittance[1] = matType == 1 ? albedo[1] : 0;
            emittance[2] = matType == 1 ? albedo[2] : 0;

            float pdf_val = pdfVals[pos]; 

            float cosine2 = dot(normal, newDir);

            float scattering_pdf = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;// just cosine/pi for lambertian
            float multVecs[3] = {albedo[0]*finalCol[0],   // albedo*incoming 
                                  albedo[1]*finalCol[1],  
                                  albedo[2]*finalCol[2]}; 

            float directLightMult = shadowRays[pos]==1 &&  sConstants.numImportantShapes>1 ?  sConstants.numImportantShapes : 1;

            float pdfs = scattering_pdf / pdf_val;
            finalCol[0] = emittance[0] + multVecs[0] * pdfs * directLightMult;
            finalCol[1] = emittance[1] + multVecs[1] * pdfs * directLightMult;
            finalCol[2] = emittance[2] + multVecs[2] * pdfs * directLightMult;
        }
        ReturnStruct ret;
        ret.xyz[0] = finalCol[0];
        ret.xyz[1] = finalCol[1];
        ret.xyz[2] = finalCol[2];
        if ( sConstants.getDenoiserInf == 1) {
            ret.normal[0] = normals[0][0];
            ret.normal[1] = normals[0][1];
            ret.normal[2] = normals[0][2];
            int matIndAlb1 = sConstants.shapes[(int)rayPositions[0][3]][1];
            ret.albedo1[0] =  sConstants.matList[matIndAlb1][0];
            ret.albedo1[1] =  sConstants.matList[matIndAlb1][1];
            ret.albedo1[2] =  sConstants.matList[matIndAlb1][2];
            int matIndAlb2 = sConstants.shapes[(int)rayPositions[1][3]][1];
            ret.albedo2[0] =  sConstants.matList[matIndAlb2][0];
            ret.albedo2[1] =  sConstants.matList[matIndAlb2][1];
            ret.albedo2[2] =  sConstants.matList[matIndAlb2][2];
            ret.directLight = 0.0f;
            for (int c = 0; c< sConstants.maxDepth; c++)
                ret.directLight += (float)shadowRays[c] / (float) sConstants.maxDepth;
            ret.worldPos[0] = rayPositions[0][0];
            ret.worldPos[1] = rayPositions[0][1];
            ret.worldPos[2] = rayPositions[0][2];
        }
        ret.raysSent = numRays;
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE ReturnStruct OMP(skepu::Index2D ind, RandomSeeds seeds, Constants sConstants)
{
        // Ray
        float camPos[3] = { sConstants.camPos[0],  sConstants.camPos[1],  sConstants.camPos[2]};
        float rayPos[3] = {camPos[0], camPos[1], camPos[2]};

        // Lambda Functions
            auto dot  = [=](float vec1[3], float vec2[3]) {return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];};
            auto norm = [&](float vec[3]) {auto d = sqrt(dot(vec,vec)); vec[0]/=d; vec[1]/=d; vec[2]/=d; };
            auto randBetween = [](RandomSeeds& seeds, float min, float max) {
                uint64_t s0 = seeds.s1;
                uint64_t s1 = seeds.s2;
                uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                double one_two = ((uint64_t)1 << 63) * (double)2.0;
                float rand = xorshiro / one_two;
                s1 ^= s0;
                seeds.s1 = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                seeds.s2 = (s1 << 28) | (s1 >> 36);

                rand *= max - min;
                rand += min;
                return rand;
            };
            auto QMult = [&](float q1[4], float q2[4] ) {
                auto A1 = (q1[3] + q1[1]) * (q2[1] + q2[2]);
                auto A3 = (q1[0] - q1[2]) * (q2[0] + q2[3]);
                auto A4 = (q1[0] + q1[2]) * (q2[0] - q2[3]);
                auto A2 = A1 + A3 + A4;
                auto A5 = (q1[3] - q1[1]) * (q2[1] - q2[2]);
                A5 = (A5 + A2) / 2.0f;

                auto Q1 = A5 - A1 + (q1[3] - q1[2]) * (q2[2] - q2[3]);
                auto Q2 = A5 - A2 + (q1[1] + q1[0]) * (q2[1] + q2[0]);
                auto Q3 = A5 - A3 + (q1[0] - q1[1]) * (q2[2] + q2[3]);
                auto Q4 = A5 - A4 + (q1[3] + q1[2]) * (q2[0] - q2[1]);

                q1[0] = Q1; q1[1] = Q2; q1[2] = Q3; q1[3] = Q4;
            };
            auto rotate = [&](float to_rotate[3], float q[4]) {

                float p[4]  {0, to_rotate[0], to_rotate[1], to_rotate[2]};
                float qR[4] {q[0],-q[1],-q[2],-q[3]};

                QMult(p, q);
                QMult(qR, p);
                to_rotate[0]=qR[1];to_rotate[1]=qR[2];to_rotate[2]=qR[3];
            };
        // Lambda Functions

        // Rand Samp
        float rSamps[2] = {0.0f, 0.0f};
        if (sConstants.randSamp>0.001f) {
            rSamps[0] = randBetween( seeds, -1, 1) * sConstants.randSamp;
            rSamps[1] = randBetween( seeds, -1, 1) * sConstants.randSamp;
        }

        float back_col[3] = { 0,0,0};

        // Pixel Coord
        float camForward[3] = { sConstants.camForward[0],  sConstants.camForward[1],  sConstants.camForward[2]};
        float camRight[3] = { sConstants.camRight[0],  sConstants.camRight[1],  sConstants.camRight[2]};

        float pY = - sConstants.maxAngleV + 2.0f* sConstants.maxAngleV*((float)ind.row/(float) sConstants.RESV);
        float pX = - sConstants.maxAngleH + 2.0f* sConstants.maxAngleH*((float)ind.col/(float) sConstants.RESH);

        float pix[3] = {0,0,0};
        pix[0] = camPos[0] +  sConstants.camForward[0]* sConstants.focalLength +  sConstants.camRight[0]*(pX+rSamps[0]) +  sConstants.camUp[0]*(pY+rSamps[1]);
        pix[1] = camPos[1] +  sConstants.camForward[1]* sConstants.focalLength +  sConstants.camRight[1]*(pX+rSamps[0]) +  sConstants.camUp[1]*(pY+rSamps[1]);
        pix[2] = camPos[2] +  sConstants.camForward[2]* sConstants.focalLength +  sConstants.camRight[2]*(pX+rSamps[0]) +  sConstants.camUp[2]*(pY+rSamps[1]);

        float rayDir[3] = {pix[0]-camPos[0], pix[1]-camPos[1], pix[2]-camPos[2]};
        norm(rayDir);
 
        // Store ray collisions and reverse through them (last num is shape index)
        float rayPositions[12][4];
        float normals[12][3];
        float pdfVals[12];
        // Shadow Rays: counts succesful shadow rays i.e. direct lighting, done for each bounce to provide more info
        int shadowRays[12];
        for (int v=0; v<12; v++){
            normals[v][0]=0.0f;normals[v][1]=0.0f;normals[v][2]=0.0f;
            pdfVals[v] = 1.0f / M_PI;
            shadowRays[v] = 0;
        }

        int numShapeHit = 0;
        int numRays = 0;
        float dir[3] = {rayDir[0], rayDir[1], rayDir[2]};
        for (int pos = 0; pos <  sConstants.maxDepth; pos++) {
            numRays++;
            int shapeHit;
            float prevPos[3];
            if (pos > 0) {
                prevPos[0] = rayPositions[pos-1][0];
                prevPos[1] = rayPositions[pos-1][1];
                prevPos[2] = rayPositions[pos-1][2];
            } else {
                prevPos[0] = camPos[0];
                prevPos[1] = camPos[1];
                prevPos[2] = camPos[2];
            }
            float posHit[3] {0.0f, 0.0f, 0.0f};
            float OBBSpacePosHit[3] {0.0f, 0.0f, 0.0f};
            bool hitAnything = false;
            int shapeTypeHit, attrInd, matInd;
            // Collide with shapes, generating new dirs as needed (i.e. random or specular)
            {

                float E = 0.00001f;

                // Find shape
                {
                    float t = INFINITY;
                    for (int ind = 0; ind <  sConstants.numShapes; ind++) {
                        int shapeType = sConstants.shapes[ind][0];
                        int aInd = sConstants.shapes[ind][2];
                        float tempT = INFINITY;
                        // ----- intersect shapes -----
                        // aabb
                        if ( shapeType == 1) {

                            // Transform Ray
                            float rDir[3] = {dir[0], dir[1], dir[2]};
                            float boxPos[3] = {sConstants.objAttributes[aInd + 0], sConstants.objAttributes[aInd + 1], sConstants.objAttributes[aInd + 2]};
                            float rPos[3] = {prevPos[0]-boxPos[0], prevPos[1]-boxPos[1], prevPos[2]-boxPos[2]};
                            float rot[4] = {sConstants.objAttributes[aInd + 9], sConstants.objAttributes[aInd + 10], sConstants.objAttributes[aInd + 11], sConstants.objAttributes[aInd + 12]};
                            if (rot[1] + rot[2] + rot[3] > E) {
                                rotate(rDir,rot);
                                norm(rDir); 
                                rotate(rPos,rot);
                            }
                            rPos[0]+=boxPos[0];rPos[1]+=boxPos[1];rPos[2]+=boxPos[2];

                            int sign[3] = {rDir[0] < 0, rDir[1] < 0, rDir[2] < 0};
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] =  sConstants.objAttributes[aInd + 3];
                            bounds[0][1] =  sConstants.objAttributes[aInd + 4];
                            bounds[0][2] =  sConstants.objAttributes[aInd + 5];
                            bounds[1][0] =  sConstants.objAttributes[aInd + 6];
                            bounds[1][1] =  sConstants.objAttributes[aInd + 7];
                            bounds[1][2] =  sConstants.objAttributes[aInd + 8];
                            float tmin = (bounds[sign[0]][0] - rPos[0]) / rDir[0];
                            float tmax = (bounds[1 - sign[0]][0] - rPos[0]) / rDir[0];
                            float tymin = (bounds[sign[1]][1] - rPos[1]) / rDir[1];
                            float tymax = (bounds[1 - sign[1]][1] - rPos[1]) / rDir[1];
                            if ((tmin > tymax) || (tymin > tmax))
                                continue;
                            if (tymin > tmin)
                                tmin = tymin;
                            if (tymax < tmax)
                                tmax = tymax;
                            float tzmin = (bounds[sign[2]][2] - rPos[2]) / rDir[2];
                            float tzmax = (bounds[1 - sign[2]][2] - rPos[2]) / rDir[2];
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
                                OBBSpacePosHit[0] = rPos[0] + rDir[0]*tempT;
                                OBBSpacePosHit[1] = rPos[1] + rDir[1]*tempT;
                                OBBSpacePosHit[2] = rPos[2] + rDir[2]*tempT;
                            }
                        }
                        // sphere
                        else if (shapeType == 0) {
                            float L[3] = {0,0,0};
                            L[0] =  sConstants.objAttributes[aInd + 0] - prevPos[0];
                            L[1] =  sConstants.objAttributes[aInd + 1] - prevPos[1];
                            L[2] =  sConstants.objAttributes[aInd + 2] - prevPos[2];
                            float tca = dot(L, dir);
                            if (tca < E)
                                continue;
                            float dsq = dot(L,L) - tca * tca;
                            float radiusSq =  sConstants.objAttributes[aInd + 3] *  sConstants.objAttributes[aInd + 3];
                            if (radiusSq - dsq < E)
                                continue;
                            float thc = sqrt(radiusSq - dsq);
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
                            posHit[0] = prevPos[0] + dir[0]*t;
                            posHit[1] = prevPos[1] + dir[1]*t;
                            posHit[2] = prevPos[2] + dir[2]*t;
                            shapeHit = ind;
                            attrInd = aInd;
                            matInd = sConstants.shapes[shapeHit][1];
                            shapeTypeHit = shapeType;
                        }
                    }
                }

                if (hitAnything) {

                    // Get Normal
                    {
                        if (shapeTypeHit == 1) {
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] =  sConstants.objAttributes[attrInd + 3];
                            bounds[0][1] =  sConstants.objAttributes[attrInd + 4];
                            bounds[0][2] =  sConstants.objAttributes[attrInd + 5];
                            bounds[1][0] =  sConstants.objAttributes[attrInd + 6];
                            bounds[1][1] =  sConstants.objAttributes[attrInd + 7];
                            bounds[1][2] =  sConstants.objAttributes[attrInd + 8];
                            normals[pos][0] = 0;
                            normals[pos][1] = 0;
                            normals[pos][2] = 0;

                            // Flat 
                            if (fabs(bounds[0][0] - bounds[1][0]) < E) {
                                normals[pos][0] = dir[0] > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0][1] - bounds[1][1]) < E) {
                                normals[pos][1] = dir[1] > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0][2] - bounds[1][2]) < E) {
                                normals[pos][2] = dir[2] > E ? -1 : 1;
                            }
                            // Not Flat
                            else if (fabs(OBBSpacePosHit[0] - bounds[0][0]) < E)
                                normals[pos][0] = -1;
                            else if (fabs(OBBSpacePosHit[0] - bounds[1][0]) < E)
                                normals[pos][0] = 1;
                            else if (fabs(OBBSpacePosHit[1] - bounds[0][1]) < E)
                                normals[pos][1] = -1;
                            else if (fabs(OBBSpacePosHit[1] - bounds[1][1]) < E)
                                normals[pos][1] = 1;
                            else if (fabs(OBBSpacePosHit[2] - bounds[0][2]) < E)
                                normals[pos][2] = -1;
                            else if (fabs(OBBSpacePosHit[2] - bounds[1][2]) < E)
                                normals[pos][2] = 1;

                            // Transform Normal
                            float rot[4] = {sConstants.objAttributes[attrInd + 9], -sConstants.objAttributes[attrInd + 10], -sConstants.objAttributes[attrInd + 11], -sConstants.objAttributes[attrInd + 12]};
                            rotate(normals[pos], rot);
                            norm(normals[pos]);
                        }
                        else if (shapeTypeHit == 0) {
                            normals[pos][0] = posHit[0] -  sConstants.objAttributes[attrInd + 0];
                            normals[pos][1] = posHit[1] -  sConstants.objAttributes[attrInd + 1];
                            normals[pos][2] = posHit[2] -  sConstants.objAttributes[attrInd + 2];
                            norm(normals[pos]);
                        }
                    }
                
                    // Gen new dirs and pdfs
                    {

                        // Random Dir Generation
                            float randDir[3];
                            float rands[5];
                            {
                                // Rand vals
                                for (int n = 0; n < 5; n++) 
                                    rands[n] = randBetween( seeds, 0,1);

                                float axis[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
                                // 2
                                // axis[2] = normal
                                axis[2][0] = normals[pos][0];
                                axis[2][1] = normals[pos][1];
                                axis[2][2] = normals[pos][2];
                                // 1
                                if (fabs(axis[2][0]) > 0.9) {
                                    // axis[1] = cross(axis[2], [0,1,0])
                                    axis[1][0] = -axis[2][2];
                                    axis[1][2] =  axis[2][0]; 
                                }
                                else {
                                    // axis[1] = cross(axis[2], [1,0,0])
                                    axis[1][1] =  axis[2][2];
                                    axis[1][2] = -axis[2][1];
                                }
                                norm(axis[1]);
                                // 0
                                // axis[0] = cross(axis[2], axis[1])
                                axis[0][0] = axis[2][1]*axis[1][2] - axis[2][2]*axis[1][1];
                                axis[0][1] = axis[2][2]*axis[1][0] - axis[2][0]*axis[1][2];
                                axis[0][2] = axis[2][0]*axis[1][1] - axis[2][1]*axis[1][0];
                                // rand dir
                                float phi = 2.0f * M_PI * rands[0];
                                float x = cos(phi) * sqrt(rands[1]);
                                float y = sin(phi) * sqrt(rands[1]);
                                float z = sqrt(1.0f - rands[1]);

                                randDir[0] = x * axis[0][0] + y * axis[1][0] + z * axis[2][0];
                                randDir[1] = x * axis[0][1] + y * axis[1][1] + z * axis[2][1];
                                randDir[2] = x * axis[0][2] + y * axis[1][2] + z * axis[2][2];
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
                        if ( sConstants.matList[matInd][5] == 3) {
                            // Dielectric Material
                            shadowRays[pos] = 1;
                            float blur =  sConstants.matList[matInd][3];
                            float RI = 1.0f /  sConstants.matList[matInd][4];
                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float refNorm[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};
                            float cosi = dot(dirIn, refNorm);

                            // If normal is same direction as ray, then flip
                            if (cosi > 0) {
                                refNorm[0]*=-1.0f;refNorm[1]*=-1.0f;refNorm[2]*=-1.0f;
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
                            float schlick = r0 + (1.0f - r0) * (float)pow((1.0f - cosi), 5.0f);

                            float schlickRand = randBetween(seeds, 0, 1);

                            if (!canRefract || schlick > schlickRand) {
                                dir[0] = (dirIn[0] - 2.0f*cosi*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                dir[1] = (dirIn[1] - 2.0f*cosi*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                dir[2] = (dirIn[2] - 2.0f*cosi*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                norm(dir);
                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                float cosine2 = dot(normals[pos], dir);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;
                            }
                            else {

                                float refCalc = RI*cosi - (float)sqrt(1.0f-sinSq);
                                dir[0] = (RI*dirIn[0] + refCalc*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                dir[1] = (RI*dirIn[1] + refCalc*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                dir[2] = (RI*dirIn[2] + refCalc*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                norm(dir);

                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                // Danger here, scattering pdf was going to 0 for refracting and making colour explode
                                float cosine2 = dot(normals[pos], dir);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;					
                            }
                        }
                        else if ( sConstants.matList[matInd][5] == 2) {
                            // Metal material
                            shadowRays[pos] = 1;

                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float blur =  sConstants.matList[matInd][3];

                            float prevDirNormalDot = dot(dirIn, normals[pos]);

                            dir[0] = (dirIn[0] - 2.0f*prevDirNormalDot*normals[pos][0])*(1.0f - blur) + blur*randDir[0];
                            dir[1] = (dirIn[1] - 2.0f*prevDirNormalDot*normals[pos][1])*(1.0f - blur) + blur*randDir[1];
                            dir[2] = (dirIn[2] - 2.0f*prevDirNormalDot*normals[pos][2])*(1.0f - blur) + blur*randDir[2];

                            float cosine2 = dot(dir, normals[pos]);

                            // Same as scattering pdf, to make pdf 1, as it is specular
                            pdfVals[pos] = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;
                        }
                        else {
                            // Lambertian/Light Material

                            dir[0] = randDir[0];
                            dir[1] = randDir[1];
                            dir[2] = randDir[2];

                            // Check Light Mat
                            bool mixPdf;
                            int impInd, impShape;
                            if ( sConstants.matList[matInd][5] == 1) {
                                shadowRays[pos] = 1;
                                mixPdf = false;
                            } else {
                                // Get importance shape
                                mixPdf =  sConstants.numImportantShapes > 0;

                                if (mixPdf) { 
                                    impInd = rands[3] *  sConstants.numImportantShapes * 0.99999f;
                                    impShape =  sConstants.importantShapes[impInd];
                                    if (impShape==shapeHit) {
                                        mixPdf = false;
                                        // mixPdf =  sConstants.numImportantShapes > 1;
                                        // impInd = (impInd+1) %  sConstants.numImportantShapes;
                                        // impShape =  sConstants.importantShapes[impInd];
                                    } 
                                }
                            }

                            // Calculate PDF val
                            if (mixPdf) {
                                // 0
                                float p0 = 1 / M_PI;
                                int choosePdf = rands[4] > 0.65f ? 1 : 0;
                                int impAttrInd = sConstants.shapes[impShape][2];
                                // dot(randomly generated dir, ray dir) / PI
                                if (choosePdf == 1) {
                                    // Generate dir towards importance shape
                                    float randPos[3] = {0,0,0};
                                    if ( sConstants.shapes[impShape][0] == 1) {
                                        // Gen three new random variables : [0, 1]
                                        float aabbRands[3];
                                        for (int n = 0; n < 3; n++) 
                                            aabbRands[n] = randBetween( seeds, 0,1);
                                        randPos[0] = (1.0f - aabbRands[0])* sConstants.objAttributes[impAttrInd+3] + aabbRands[0]* sConstants.objAttributes[impAttrInd+6];
                                        randPos[1] = (1.0f - aabbRands[1])* sConstants.objAttributes[impAttrInd+4] + aabbRands[1]* sConstants.objAttributes[impAttrInd+7];
                                        randPos[2] = (1.0f - aabbRands[2])* sConstants.objAttributes[impAttrInd+5] + aabbRands[2]* sConstants.objAttributes[impAttrInd+8];	
                                    } 
                                    else if ( sConstants.shapes[impShape][0] == 0) {
                                        // Gen three new random variables : [-1, 1]
                                        float sphereRands[3];
                                        sphereRands[0] = randBetween( seeds, -1,1);
                                        sphereRands[1] = randBetween( seeds, -1,1);
                                        sphereRands[2] = randBetween( seeds, -1,1);
                                        norm(sphereRands);
                                        
                                        randPos[0] =  sConstants.objAttributes[impAttrInd+0] + sphereRands[0]* sConstants.objAttributes[impAttrInd+3];
                                        randPos[1] =  sConstants.objAttributes[impAttrInd+1] + sphereRands[1]* sConstants.objAttributes[impAttrInd+3];
                                        randPos[2] =  sConstants.objAttributes[impAttrInd+2] + sphereRands[2]* sConstants.objAttributes[impAttrInd+3];
                                    }

                                    float directDir[3];
                                    directDir[0] = randPos[0] - posHit[0];
                                    directDir[1] = randPos[1] - posHit[1];
                                    directDir[2] = randPos[2] - posHit[2];
                                    float dirLen = sqrt(dot(directDir, directDir));
                                    directDir[0] /= dirLen; directDir[1] /= dirLen; directDir[2] /= dirLen;  

                                    //
                                    // Shadow Ray
                                    // Need to send shadow ray to see if point is in path of direct light
                                    bool shadowRayHit = false;
                                    float shadowDir[3] {directDir[0], directDir[1], directDir[2]};
                                    for (int ind = 0; ind <  sConstants.numShapes; ind++) {
                                        if (ind == impShape)
                                            continue;
                                        int shapeType = sConstants.shapes[ind][0];
                                        int sMatInd = sConstants.shapes[ind][1];
                                        int aInd = sConstants.shapes[ind][2];
                                        float tempT = INFINITY;
                                        float obbPosHit[3];
                                        // ----- intersect shapes -----
                                        // aabb
                                        if ( shapeType == 1) {
                                            // Transform Ray
                                            float rDir[3] = {shadowDir[0], shadowDir[1], shadowDir[2]};
                                            float boxPos[3] = {sConstants.objAttributes[aInd + 0], sConstants.objAttributes[aInd + 1], sConstants.objAttributes[aInd + 2]};
                                            float rPos[3] = {posHit[0]-boxPos[0], posHit[1]-boxPos[1], posHit[2]-boxPos[2]};
                                            float rot[4] = {sConstants.objAttributes[aInd + 9], sConstants.objAttributes[aInd + 10], sConstants.objAttributes[aInd + 11], sConstants.objAttributes[aInd + 12]};
                                            if (rot[1] + rot[2] + rot[3] > E) {
                                                rotate(rDir,rot);
                                                norm(rDir); 
                                                rotate(rPos,rot);
                                            }
                                            rPos[0]+=boxPos[0];rPos[1]+=boxPos[1];rPos[2]+=boxPos[2];

                                            int sign[3] = {rDir[0] < 0, rDir[1] < 0, rDir[2] < 0};
                                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                                            bounds[0][0] =  sConstants.objAttributes[aInd + 3];
                                            bounds[0][1] =  sConstants.objAttributes[aInd + 4];
                                            bounds[0][2] =  sConstants.objAttributes[aInd + 5];
                                            bounds[1][0] =  sConstants.objAttributes[aInd + 6];
                                            bounds[1][1] =  sConstants.objAttributes[aInd + 7];
                                            bounds[1][2] =  sConstants.objAttributes[aInd + 8];
                                            float tmin = (bounds[sign[0]][0] - rPos[0]) / rDir[0];
                                            float tmax = (bounds[1 - sign[0]][0] - rPos[0]) / rDir[0];
                                            float tymin = (bounds[sign[1]][1] - rPos[1]) / rDir[1];
                                            float tymax = (bounds[1 - sign[1]][1] - rPos[1]) / rDir[1];
                                            if ((tmin > tymax) || (tymin > tmax))
                                                continue;
                                            if (tymin > tmin)
                                                tmin = tymin;
                                            if (tymax < tmax)
                                                tmax = tymax;
                                            float tzmin = (bounds[sign[2]][2] - rPos[2]) / rDir[2];
                                            float tzmax = (bounds[1 - sign[2]][2] - rPos[2]) / rDir[2];
                                            if ((tmin > tzmax) || (tzmin > tmax))
                                                continue;
                                            if (tzmin > tmin)
                                                tmin = tzmin;
                                            if (tzmax < tmax)
                                                tmax = tzmax;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = tmin > E ? tmin : (tmax > E ? tmax : INFINITY); 

                                            if ((int)sConstants.matList[sMatInd][5]==3) {
                                                obbPosHit[0] = rPos[0] + rDir[0]*tempT;
                                                obbPosHit[1] = rPos[1] + rDir[1]*tempT;
                                                obbPosHit[2] = rPos[2] + rDir[2]*tempT;
                                            }  
                                        }
                                        // sphere
                                        else if (shapeType == 0) {
                                            float L[3];
                                            L[0] =  sConstants.objAttributes[aInd+0] - posHit[0];
                                            L[1] =  sConstants.objAttributes[aInd+1] - posHit[1];
                                            L[2] =  sConstants.objAttributes[aInd+2] - posHit[2];
                                            float tca = L[0]*shadowDir[0] + L[1]*shadowDir[1] + L[2]*shadowDir[2];
                                            if (tca < E)
                                                continue;
                                            float dsq = dot(L,L) - tca * tca;
                                            float radiusSq =  sConstants.objAttributes[aInd+3] *  sConstants.objAttributes[aInd+3];
                                            if (radiusSq - dsq < E)
                                                continue;
                                            float thc = sqrt(radiusSq - dsq);
                                            float t0 = tca - thc;
                                            float t1 = tca + thc;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = t0 > E ? t0 : (t1 > E ? t1 : INFINITY); 
                                            
                                        }
                                        if (tempT < dirLen) {
                                            // Dialectric Check
                                            if ((int)sConstants.matList[sMatInd][5]==3) {
                                                float blur = sConstants.matList[sMatInd][3];
                                                float RI = 1.0f / sConstants.matList[sMatInd][4];
                                                // Get Normal
                                                float refNorm[3] {0.0f, 0.0f, 0.0f};
                                                if (shapeTypeHit == 1) {
                                                    float bounds[2][3] = {{0,0,0}, {0,0,0}};
                                                    bounds[0][0] =  sConstants.objAttributes[attrInd + 3];
                                                    bounds[0][1] =  sConstants.objAttributes[attrInd + 4];
                                                    bounds[0][2] =  sConstants.objAttributes[attrInd + 5];
                                                    bounds[1][0] =  sConstants.objAttributes[attrInd + 6];
                                                    bounds[1][1] =  sConstants.objAttributes[attrInd + 7];
                                                    bounds[1][2] =  sConstants.objAttributes[attrInd + 8];

                                                    // Flat 
                                                    if (fabs(bounds[0][0] - bounds[1][0]) < E) {
                                                        refNorm[0] = shadowDir[0] > 0 ? -1 : 1;
                                                    }
                                                    else if (fabs(bounds[0][1] - bounds[1][1]) < E) {
                                                        refNorm[1] = shadowDir[1] > 0 ? -1 : 1;
                                                    }
                                                    else if (fabs(bounds[0][2] - bounds[1][2]) < E) {
                                                        refNorm[2] = shadowDir[2] > 0 ? -1 : 1;
                                                    }
                                                    // Not Flat
                                                    else if (fabs(obbPosHit[0] - bounds[0][0]) < E)
                                                        refNorm[0] = -1;
                                                    else if (fabs(obbPosHit[0] - bounds[1][0]) < E)
                                                        refNorm[0] = 1;
                                                    else if (fabs(obbPosHit[1] - bounds[0][1]) < E)
                                                        refNorm[1] = -1;
                                                    else if (fabs(obbPosHit[1] - bounds[1][1]) < E)
                                                        refNorm[1] = 1;
                                                    else if (fabs(obbPosHit[2] - bounds[0][0]) < E)
                                                        refNorm[2] = -1;
                                                    else if (fabs(obbPosHit[2] - bounds[1][0]) < E)
                                                        refNorm[2] = 1;

                                                    // Transform Normal
                                                    float rot[4] = {sConstants.objAttributes[attrInd + 9], -sConstants.objAttributes[attrInd + 10], -sConstants.objAttributes[attrInd + 11], -sConstants.objAttributes[attrInd + 12]};
                                                    rotate(refNorm, rot);
                                                    norm(refNorm);
                                                }
                                                else if (shapeTypeHit == 0) {
                                                    float sPosHit[3];
                                                    sPosHit[0] = posHit[0] + shadowDir[0]*tempT;
                                                    sPosHit[1] = posHit[1] + shadowDir[1]*tempT;
                                                    sPosHit[2] = posHit[2] + shadowDir[2]*tempT;
                                                    refNorm[0] = sPosHit[0] -  sConstants.objAttributes[attrInd + 0];
                                                    refNorm[1] = sPosHit[1] -  sConstants.objAttributes[attrInd + 1];
                                                    refNorm[2] = sPosHit[2] -  sConstants.objAttributes[attrInd + 2];
                                                    norm(refNorm);
                                                } 
                                                float cosi = dot(shadowDir, refNorm);
                                                // If normal is same direction as ray, then flip
                                                if (cosi > 0) {
                                                    refNorm[0]*=-1.0f;refNorm[1]*=-1.0f;refNorm[2]*=-1.0f;
                                                    RI = 1.0f / RI;
                                                }
                                                else {
                                                    cosi*=-1.0f;
                                                }

                                                // Can refract check
                                                float sinSq = RI*RI*(1.0f-cosi*cosi);
                                                bool canRefract = 1.0f - sinSq > E;

                                                if (!canRefract) {
                                                    shadowDir[0] = (shadowDir[0] - 2.0f*cosi*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                                    shadowDir[1] = (shadowDir[1] - 2.0f*cosi*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                                    shadowDir[2] = (shadowDir[2] - 2.0f*cosi*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                                    norm(shadowDir);
                                                }
                                                else {

                                                    float refCalc = RI*cosi - (float)sqrt(1.0f-sinSq);
                                                    shadowDir[0] = (RI*shadowDir[0] + refCalc*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                                    shadowDir[1] = (RI*shadowDir[1] + refCalc*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                                    shadowDir[2] = (RI*shadowDir[2] + refCalc*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                                    norm(shadowDir);					
                                                }
                                                continue;
                                            }                                           
                                            shadowRayHit = true;
                                            break;
                                        }
                                    }

                                    if (!shadowRayHit) {
                                        float cosine = fabs(dot(directDir, randDir));
                                        if (cosine > 0.01f) {
                                            shadowRays[pos]=1; 
                                            dir[0] = directDir[0];
                                            dir[1] = directDir[1];
                                            dir[2] = directDir[2];
                                            p0 = fabs(cosine) / M_PI;
                                        }

                                    }
                                    // Shadow Ray
                                    //
                                    //


                                }
                                // 1
                                float p1 = 0;
                                if ( sConstants.shapes[impShape][0] == 1) {
                                    // AABB pdf val
                                    float areaSum = 0;
                                    float xDiff =  sConstants.objAttributes[impAttrInd+3] -  sConstants.objAttributes[impAttrInd+6];
                                    float yDiff =  sConstants.objAttributes[impAttrInd+4] -  sConstants.objAttributes[impAttrInd+7];
                                    float zDiff =  sConstants.objAttributes[impAttrInd+5] -  sConstants.objAttributes[impAttrInd+8];
                                    areaSum += xDiff * yDiff * 2.0f;
                                    areaSum += zDiff * yDiff * 2.0f;
                                    areaSum += xDiff * zDiff * 2.0f;
                                    float cosine = dot(dir, normals[pos]);
                                    cosine = cosine < 0.0001f ? 0.0001f : cosine;

                                    float diff[3];
                                    diff[0] =  sConstants.objAttributes[impAttrInd+0] - posHit[0];
                                    diff[1] =  sConstants.objAttributes[impAttrInd+1] - posHit[1];
                                    diff[2] =  sConstants.objAttributes[impAttrInd+2] - posHit[2];
                                    float dirLen = sqrt(dot(diff, diff));


                                    // AABB needs magic number for pdf calc, TODO: LOOK INTO, was too bright before
                                    //p1 = 1 / (cosine * areaSum);
                                    p1 = dirLen / (cosine * areaSum);

                                } else if ( sConstants.shapes[impShape][0] == 0) {
                                    // Sphere pdf val
                                    float diff[3] = { sConstants.objAttributes[impAttrInd+0]-posHit[0], 
                                                      sConstants.objAttributes[impAttrInd+1]-posHit[1], 
                                                      sConstants.objAttributes[impAttrInd+2]-posHit[2]};
                                    auto distance_squared = dot(diff, diff);
                                    auto cos_theta_max = sqrt(1.0f -  sConstants.objAttributes[impAttrInd+3] *  sConstants.objAttributes[impAttrInd+3] / distance_squared);
                                    // NaN check
                                    cos_theta_max = (cos_theta_max != cos_theta_max) ? 0.9999f : cos_theta_max;
                                    auto solid_angle = M_PI * (1.0f - cos_theta_max) *2.0f;

                                    // Sphere needs magic number for pdf calc, TODO: LOOK INTO, was too dark before
                                    //p1 = 1 / (solid_angle );
                                    p1 =  sConstants.objAttributes[impAttrInd+3] / (solid_angle * sqrt(distance_squared)*4.0f);
                                }

                                pdfVals[pos] = 0.5f*p0 + 0.5f*p1;
                            }
                        }
                    }

                    numShapeHit++;
                    rayPositions[pos][0] = posHit[0];
                    rayPositions[pos][1] = posHit[1];
                    rayPositions[pos][2] = posHit[2];
                    rayPositions[pos][3] = shapeHit;

                } else {
                    back_col[0] = 0.1f;
                    back_col[1] = 0.1f;
                    back_col[2] = (dir[1] + 1.0f)/2.2f + 0.1f;
                    break;
                }
            }
        }

        float finalCol[3] = {back_col[0], 
                            back_col[1], 
                            back_col[2]};

        // Reverse through hit points and add up colour
        for (int pos = numShapeHit-1; pos >=0; pos--) {

            int shapeHit = (int)rayPositions[pos][3];
            int matInd = sConstants.shapes[shapeHit][1];

            float albedo[3] = {  sConstants.matList[matInd][0], 
                                 sConstants.matList[matInd][1], 
                                 sConstants.matList[matInd][2]};
            int matType = (int) sConstants.matList[matInd][5];	
        
            float normal[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};

            float newDir[3];
            newDir[0] = pos == numShapeHit-1 ? dir[0] : rayPositions[pos + 1][0] - rayPositions[pos][0];
            newDir[1] = pos == numShapeHit-1 ? dir[1] : rayPositions[pos + 1][1] - rayPositions[pos][1];
            newDir[2] = pos == numShapeHit-1 ? dir[2] : rayPositions[pos + 1][2] - rayPositions[pos][2];
            if (pos < numShapeHit-1) {
                norm(newDir);
            }

            float emittance[3];
            emittance[0] = matType == 1 ? albedo[0] : 0;
            emittance[1] = matType == 1 ? albedo[1] : 0;
            emittance[2] = matType == 1 ? albedo[2] : 0;

            float pdf_val = pdfVals[pos]; 

            float cosine2 = dot(normal, newDir);

            float scattering_pdf = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;// just cosine/pi for lambertian
            float multVecs[3] = {albedo[0]*finalCol[0],   // albedo*incoming 
                                  albedo[1]*finalCol[1],  
                                  albedo[2]*finalCol[2]}; 

            float directLightMult = shadowRays[pos]==1 &&  sConstants.numImportantShapes>1 ?  sConstants.numImportantShapes : 1;

            float pdfs = scattering_pdf / pdf_val;
            finalCol[0] = emittance[0] + multVecs[0] * pdfs * directLightMult;
            finalCol[1] = emittance[1] + multVecs[1] * pdfs * directLightMult;
            finalCol[2] = emittance[2] + multVecs[2] * pdfs * directLightMult;
        }
        ReturnStruct ret;
        ret.xyz[0] = finalCol[0];
        ret.xyz[1] = finalCol[1];
        ret.xyz[2] = finalCol[2];
        if ( sConstants.getDenoiserInf == 1) {
            ret.normal[0] = normals[0][0];
            ret.normal[1] = normals[0][1];
            ret.normal[2] = normals[0][2];
            int matIndAlb1 = sConstants.shapes[(int)rayPositions[0][3]][1];
            ret.albedo1[0] =  sConstants.matList[matIndAlb1][0];
            ret.albedo1[1] =  sConstants.matList[matIndAlb1][1];
            ret.albedo1[2] =  sConstants.matList[matIndAlb1][2];
            int matIndAlb2 = sConstants.shapes[(int)rayPositions[1][3]][1];
            ret.albedo2[0] =  sConstants.matList[matIndAlb2][0];
            ret.albedo2[1] =  sConstants.matList[matIndAlb2][1];
            ret.albedo2[2] =  sConstants.matList[matIndAlb2][2];
            ret.directLight = 0.0f;
            for (int c = 0; c< sConstants.maxDepth; c++)
                ret.directLight += (float)shadowRays[c] / (float) sConstants.maxDepth;
            ret.worldPos[0] = rayPositions[0][0];
            ret.worldPos[1] = rayPositions[0][1];
            ret.worldPos[2] = rayPositions[0][2];
        }
        ret.raysSent = numRays;
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE ReturnStruct CPU(skepu::Index2D ind, RandomSeeds seeds, Constants sConstants)
{
        // Ray
        float camPos[3] = { sConstants.camPos[0],  sConstants.camPos[1],  sConstants.camPos[2]};
        float rayPos[3] = {camPos[0], camPos[1], camPos[2]};

        // Lambda Functions
            auto dot  = [=](float vec1[3], float vec2[3]) {return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];};
            auto norm = [&](float vec[3]) {auto d = sqrt(dot(vec,vec)); vec[0]/=d; vec[1]/=d; vec[2]/=d; };
            auto randBetween = [](RandomSeeds& seeds, float min, float max) {
                uint64_t s0 = seeds.s1;
                uint64_t s1 = seeds.s2;
                uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                double one_two = ((uint64_t)1 << 63) * (double)2.0;
                float rand = xorshiro / one_two;
                s1 ^= s0;
                seeds.s1 = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                seeds.s2 = (s1 << 28) | (s1 >> 36);

                rand *= max - min;
                rand += min;
                return rand;
            };
            auto QMult = [&](float q1[4], float q2[4] ) {
                auto A1 = (q1[3] + q1[1]) * (q2[1] + q2[2]);
                auto A3 = (q1[0] - q1[2]) * (q2[0] + q2[3]);
                auto A4 = (q1[0] + q1[2]) * (q2[0] - q2[3]);
                auto A2 = A1 + A3 + A4;
                auto A5 = (q1[3] - q1[1]) * (q2[1] - q2[2]);
                A5 = (A5 + A2) / 2.0f;

                auto Q1 = A5 - A1 + (q1[3] - q1[2]) * (q2[2] - q2[3]);
                auto Q2 = A5 - A2 + (q1[1] + q1[0]) * (q2[1] + q2[0]);
                auto Q3 = A5 - A3 + (q1[0] - q1[1]) * (q2[2] + q2[3]);
                auto Q4 = A5 - A4 + (q1[3] + q1[2]) * (q2[0] - q2[1]);

                q1[0] = Q1; q1[1] = Q2; q1[2] = Q3; q1[3] = Q4;
            };
            auto rotate = [&](float to_rotate[3], float q[4]) {

                float p[4]  {0, to_rotate[0], to_rotate[1], to_rotate[2]};
                float qR[4] {q[0],-q[1],-q[2],-q[3]};

                QMult(p, q);
                QMult(qR, p);
                to_rotate[0]=qR[1];to_rotate[1]=qR[2];to_rotate[2]=qR[3];
            };
        // Lambda Functions

        // Rand Samp
        float rSamps[2] = {0.0f, 0.0f};
        if (sConstants.randSamp>0.001f) {
            rSamps[0] = randBetween( seeds, -1, 1) * sConstants.randSamp;
            rSamps[1] = randBetween( seeds, -1, 1) * sConstants.randSamp;
        }

        float back_col[3] = { 0,0,0};

        // Pixel Coord
        float camForward[3] = { sConstants.camForward[0],  sConstants.camForward[1],  sConstants.camForward[2]};
        float camRight[3] = { sConstants.camRight[0],  sConstants.camRight[1],  sConstants.camRight[2]};

        float pY = - sConstants.maxAngleV + 2.0f* sConstants.maxAngleV*((float)ind.row/(float) sConstants.RESV);
        float pX = - sConstants.maxAngleH + 2.0f* sConstants.maxAngleH*((float)ind.col/(float) sConstants.RESH);

        float pix[3] = {0,0,0};
        pix[0] = camPos[0] +  sConstants.camForward[0]* sConstants.focalLength +  sConstants.camRight[0]*(pX+rSamps[0]) +  sConstants.camUp[0]*(pY+rSamps[1]);
        pix[1] = camPos[1] +  sConstants.camForward[1]* sConstants.focalLength +  sConstants.camRight[1]*(pX+rSamps[0]) +  sConstants.camUp[1]*(pY+rSamps[1]);
        pix[2] = camPos[2] +  sConstants.camForward[2]* sConstants.focalLength +  sConstants.camRight[2]*(pX+rSamps[0]) +  sConstants.camUp[2]*(pY+rSamps[1]);

        float rayDir[3] = {pix[0]-camPos[0], pix[1]-camPos[1], pix[2]-camPos[2]};
        norm(rayDir);
 
        // Store ray collisions and reverse through them (last num is shape index)
        float rayPositions[12][4];
        float normals[12][3];
        float pdfVals[12];
        // Shadow Rays: counts succesful shadow rays i.e. direct lighting, done for each bounce to provide more info
        int shadowRays[12];
        for (int v=0; v<12; v++){
            normals[v][0]=0.0f;normals[v][1]=0.0f;normals[v][2]=0.0f;
            pdfVals[v] = 1.0f / M_PI;
            shadowRays[v] = 0;
        }

        int numShapeHit = 0;
        int numRays = 0;
        float dir[3] = {rayDir[0], rayDir[1], rayDir[2]};
        for (int pos = 0; pos <  sConstants.maxDepth; pos++) {
            numRays++;
            int shapeHit;
            float prevPos[3];
            if (pos > 0) {
                prevPos[0] = rayPositions[pos-1][0];
                prevPos[1] = rayPositions[pos-1][1];
                prevPos[2] = rayPositions[pos-1][2];
            } else {
                prevPos[0] = camPos[0];
                prevPos[1] = camPos[1];
                prevPos[2] = camPos[2];
            }
            float posHit[3] {0.0f, 0.0f, 0.0f};
            float OBBSpacePosHit[3] {0.0f, 0.0f, 0.0f};
            bool hitAnything = false;
            int shapeTypeHit, attrInd, matInd;
            // Collide with shapes, generating new dirs as needed (i.e. random or specular)
            {

                float E = 0.00001f;

                // Find shape
                {
                    float t = INFINITY;
                    for (int ind = 0; ind <  sConstants.numShapes; ind++) {
                        int shapeType = sConstants.shapes[ind][0];
                        int aInd = sConstants.shapes[ind][2];
                        float tempT = INFINITY;
                        // ----- intersect shapes -----
                        // aabb
                        if ( shapeType == 1) {

                            // Transform Ray
                            float rDir[3] = {dir[0], dir[1], dir[2]};
                            float boxPos[3] = {sConstants.objAttributes[aInd + 0], sConstants.objAttributes[aInd + 1], sConstants.objAttributes[aInd + 2]};
                            float rPos[3] = {prevPos[0]-boxPos[0], prevPos[1]-boxPos[1], prevPos[2]-boxPos[2]};
                            float rot[4] = {sConstants.objAttributes[aInd + 9], sConstants.objAttributes[aInd + 10], sConstants.objAttributes[aInd + 11], sConstants.objAttributes[aInd + 12]};
                            if (rot[1] + rot[2] + rot[3] > E) {
                                rotate(rDir,rot);
                                norm(rDir); 
                                rotate(rPos,rot);
                            }
                            rPos[0]+=boxPos[0];rPos[1]+=boxPos[1];rPos[2]+=boxPos[2];

                            int sign[3] = {rDir[0] < 0, rDir[1] < 0, rDir[2] < 0};
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] =  sConstants.objAttributes[aInd + 3];
                            bounds[0][1] =  sConstants.objAttributes[aInd + 4];
                            bounds[0][2] =  sConstants.objAttributes[aInd + 5];
                            bounds[1][0] =  sConstants.objAttributes[aInd + 6];
                            bounds[1][1] =  sConstants.objAttributes[aInd + 7];
                            bounds[1][2] =  sConstants.objAttributes[aInd + 8];
                            float tmin = (bounds[sign[0]][0] - rPos[0]) / rDir[0];
                            float tmax = (bounds[1 - sign[0]][0] - rPos[0]) / rDir[0];
                            float tymin = (bounds[sign[1]][1] - rPos[1]) / rDir[1];
                            float tymax = (bounds[1 - sign[1]][1] - rPos[1]) / rDir[1];
                            if ((tmin > tymax) || (tymin > tmax))
                                continue;
                            if (tymin > tmin)
                                tmin = tymin;
                            if (tymax < tmax)
                                tmax = tymax;
                            float tzmin = (bounds[sign[2]][2] - rPos[2]) / rDir[2];
                            float tzmax = (bounds[1 - sign[2]][2] - rPos[2]) / rDir[2];
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
                                OBBSpacePosHit[0] = rPos[0] + rDir[0]*tempT;
                                OBBSpacePosHit[1] = rPos[1] + rDir[1]*tempT;
                                OBBSpacePosHit[2] = rPos[2] + rDir[2]*tempT;
                            }
                        }
                        // sphere
                        else if (shapeType == 0) {
                            float L[3] = {0,0,0};
                            L[0] =  sConstants.objAttributes[aInd + 0] - prevPos[0];
                            L[1] =  sConstants.objAttributes[aInd + 1] - prevPos[1];
                            L[2] =  sConstants.objAttributes[aInd + 2] - prevPos[2];
                            float tca = dot(L, dir);
                            if (tca < E)
                                continue;
                            float dsq = dot(L,L) - tca * tca;
                            float radiusSq =  sConstants.objAttributes[aInd + 3] *  sConstants.objAttributes[aInd + 3];
                            if (radiusSq - dsq < E)
                                continue;
                            float thc = sqrt(radiusSq - dsq);
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
                            posHit[0] = prevPos[0] + dir[0]*t;
                            posHit[1] = prevPos[1] + dir[1]*t;
                            posHit[2] = prevPos[2] + dir[2]*t;
                            shapeHit = ind;
                            attrInd = aInd;
                            matInd = sConstants.shapes[shapeHit][1];
                            shapeTypeHit = shapeType;
                        }
                    }
                }

                if (hitAnything) {

                    // Get Normal
                    {
                        if (shapeTypeHit == 1) {
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] =  sConstants.objAttributes[attrInd + 3];
                            bounds[0][1] =  sConstants.objAttributes[attrInd + 4];
                            bounds[0][2] =  sConstants.objAttributes[attrInd + 5];
                            bounds[1][0] =  sConstants.objAttributes[attrInd + 6];
                            bounds[1][1] =  sConstants.objAttributes[attrInd + 7];
                            bounds[1][2] =  sConstants.objAttributes[attrInd + 8];
                            normals[pos][0] = 0;
                            normals[pos][1] = 0;
                            normals[pos][2] = 0;

                            // Flat 
                            if (fabs(bounds[0][0] - bounds[1][0]) < E) {
                                normals[pos][0] = dir[0] > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0][1] - bounds[1][1]) < E) {
                                normals[pos][1] = dir[1] > E ? -1 : 1;
                            }
                            else if (fabs(bounds[0][2] - bounds[1][2]) < E) {
                                normals[pos][2] = dir[2] > E ? -1 : 1;
                            }
                            // Not Flat
                            else if (fabs(OBBSpacePosHit[0] - bounds[0][0]) < E)
                                normals[pos][0] = -1;
                            else if (fabs(OBBSpacePosHit[0] - bounds[1][0]) < E)
                                normals[pos][0] = 1;
                            else if (fabs(OBBSpacePosHit[1] - bounds[0][1]) < E)
                                normals[pos][1] = -1;
                            else if (fabs(OBBSpacePosHit[1] - bounds[1][1]) < E)
                                normals[pos][1] = 1;
                            else if (fabs(OBBSpacePosHit[2] - bounds[0][2]) < E)
                                normals[pos][2] = -1;
                            else if (fabs(OBBSpacePosHit[2] - bounds[1][2]) < E)
                                normals[pos][2] = 1;

                            // Transform Normal
                            float rot[4] = {sConstants.objAttributes[attrInd + 9], -sConstants.objAttributes[attrInd + 10], -sConstants.objAttributes[attrInd + 11], -sConstants.objAttributes[attrInd + 12]};
                            rotate(normals[pos], rot);
                            norm(normals[pos]);
                        }
                        else if (shapeTypeHit == 0) {
                            normals[pos][0] = posHit[0] -  sConstants.objAttributes[attrInd + 0];
                            normals[pos][1] = posHit[1] -  sConstants.objAttributes[attrInd + 1];
                            normals[pos][2] = posHit[2] -  sConstants.objAttributes[attrInd + 2];
                            norm(normals[pos]);
                        }
                    }
                
                    // Gen new dirs and pdfs
                    {

                        // Random Dir Generation
                            float randDir[3];
                            float rands[5];
                            {
                                // Rand vals
                                for (int n = 0; n < 5; n++) 
                                    rands[n] = randBetween( seeds, 0,1);

                                float axis[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
                                // 2
                                // axis[2] = normal
                                axis[2][0] = normals[pos][0];
                                axis[2][1] = normals[pos][1];
                                axis[2][2] = normals[pos][2];
                                // 1
                                if (fabs(axis[2][0]) > 0.9) {
                                    // axis[1] = cross(axis[2], [0,1,0])
                                    axis[1][0] = -axis[2][2];
                                    axis[1][2] =  axis[2][0]; 
                                }
                                else {
                                    // axis[1] = cross(axis[2], [1,0,0])
                                    axis[1][1] =  axis[2][2];
                                    axis[1][2] = -axis[2][1];
                                }
                                norm(axis[1]);
                                // 0
                                // axis[0] = cross(axis[2], axis[1])
                                axis[0][0] = axis[2][1]*axis[1][2] - axis[2][2]*axis[1][1];
                                axis[0][1] = axis[2][2]*axis[1][0] - axis[2][0]*axis[1][2];
                                axis[0][2] = axis[2][0]*axis[1][1] - axis[2][1]*axis[1][0];
                                // rand dir
                                float phi = 2.0f * M_PI * rands[0];
                                float x = cos(phi) * sqrt(rands[1]);
                                float y = sin(phi) * sqrt(rands[1]);
                                float z = sqrt(1.0f - rands[1]);

                                randDir[0] = x * axis[0][0] + y * axis[1][0] + z * axis[2][0];
                                randDir[1] = x * axis[0][1] + y * axis[1][1] + z * axis[2][1];
                                randDir[2] = x * axis[0][2] + y * axis[1][2] + z * axis[2][2];
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
                        if ( sConstants.matList[matInd][5] == 3) {
                            // Dielectric Material
                            shadowRays[pos] = 1;
                            float blur =  sConstants.matList[matInd][3];
                            float RI = 1.0f /  sConstants.matList[matInd][4];
                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float refNorm[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};
                            float cosi = dot(dirIn, refNorm);

                            // If normal is same direction as ray, then flip
                            if (cosi > 0) {
                                refNorm[0]*=-1.0f;refNorm[1]*=-1.0f;refNorm[2]*=-1.0f;
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
                            float schlick = r0 + (1.0f - r0) * (float)pow((1.0f - cosi), 5.0f);

                            float schlickRand = randBetween(seeds, 0, 1);

                            if (!canRefract || schlick > schlickRand) {
                                dir[0] = (dirIn[0] - 2.0f*cosi*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                dir[1] = (dirIn[1] - 2.0f*cosi*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                dir[2] = (dirIn[2] - 2.0f*cosi*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                norm(dir);
                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                float cosine2 = dot(normals[pos], dir);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;
                            }
                            else {

                                float refCalc = RI*cosi - (float)sqrt(1.0f-sinSq);
                                dir[0] = (RI*dirIn[0] + refCalc*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                dir[1] = (RI*dirIn[1] + refCalc*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                dir[2] = (RI*dirIn[2] + refCalc*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                norm(dir);

                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                // Danger here, scattering pdf was going to 0 for refracting and making colour explode
                                float cosine2 = dot(normals[pos], dir);
                                pdfVals[pos] = cosine2 < E ? E : cosine2 / M_PI;					
                            }
                        }
                        else if ( sConstants.matList[matInd][5] == 2) {
                            // Metal material
                            shadowRays[pos] = 1;

                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float blur =  sConstants.matList[matInd][3];

                            float prevDirNormalDot = dot(dirIn, normals[pos]);

                            dir[0] = (dirIn[0] - 2.0f*prevDirNormalDot*normals[pos][0])*(1.0f - blur) + blur*randDir[0];
                            dir[1] = (dirIn[1] - 2.0f*prevDirNormalDot*normals[pos][1])*(1.0f - blur) + blur*randDir[1];
                            dir[2] = (dirIn[2] - 2.0f*prevDirNormalDot*normals[pos][2])*(1.0f - blur) + blur*randDir[2];

                            float cosine2 = dot(dir, normals[pos]);

                            // Same as scattering pdf, to make pdf 1, as it is specular
                            pdfVals[pos] = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;
                        }
                        else {
                            // Lambertian/Light Material

                            dir[0] = randDir[0];
                            dir[1] = randDir[1];
                            dir[2] = randDir[2];

                            // Check Light Mat
                            bool mixPdf;
                            int impInd, impShape;
                            if ( sConstants.matList[matInd][5] == 1) {
                                shadowRays[pos] = 1;
                                mixPdf = false;
                            } else {
                                // Get importance shape
                                mixPdf =  sConstants.numImportantShapes > 0;

                                if (mixPdf) { 
                                    impInd = rands[3] *  sConstants.numImportantShapes * 0.99999f;
                                    impShape =  sConstants.importantShapes[impInd];
                                    if (impShape==shapeHit) {
                                        mixPdf = false;
                                        // mixPdf =  sConstants.numImportantShapes > 1;
                                        // impInd = (impInd+1) %  sConstants.numImportantShapes;
                                        // impShape =  sConstants.importantShapes[impInd];
                                    } 
                                }
                            }

                            // Calculate PDF val
                            if (mixPdf) {
                                // 0
                                float p0 = 1 / M_PI;
                                int choosePdf = rands[4] > 0.65f ? 1 : 0;
                                int impAttrInd = sConstants.shapes[impShape][2];
                                // dot(randomly generated dir, ray dir) / PI
                                if (choosePdf == 1) {
                                    // Generate dir towards importance shape
                                    float randPos[3] = {0,0,0};
                                    if ( sConstants.shapes[impShape][0] == 1) {
                                        // Gen three new random variables : [0, 1]
                                        float aabbRands[3];
                                        for (int n = 0; n < 3; n++) 
                                            aabbRands[n] = randBetween( seeds, 0,1);
                                        randPos[0] = (1.0f - aabbRands[0])* sConstants.objAttributes[impAttrInd+3] + aabbRands[0]* sConstants.objAttributes[impAttrInd+6];
                                        randPos[1] = (1.0f - aabbRands[1])* sConstants.objAttributes[impAttrInd+4] + aabbRands[1]* sConstants.objAttributes[impAttrInd+7];
                                        randPos[2] = (1.0f - aabbRands[2])* sConstants.objAttributes[impAttrInd+5] + aabbRands[2]* sConstants.objAttributes[impAttrInd+8];	
                                    } 
                                    else if ( sConstants.shapes[impShape][0] == 0) {
                                        // Gen three new random variables : [-1, 1]
                                        float sphereRands[3];
                                        sphereRands[0] = randBetween( seeds, -1,1);
                                        sphereRands[1] = randBetween( seeds, -1,1);
                                        sphereRands[2] = randBetween( seeds, -1,1);
                                        norm(sphereRands);
                                        
                                        randPos[0] =  sConstants.objAttributes[impAttrInd+0] + sphereRands[0]* sConstants.objAttributes[impAttrInd+3];
                                        randPos[1] =  sConstants.objAttributes[impAttrInd+1] + sphereRands[1]* sConstants.objAttributes[impAttrInd+3];
                                        randPos[2] =  sConstants.objAttributes[impAttrInd+2] + sphereRands[2]* sConstants.objAttributes[impAttrInd+3];
                                    }

                                    float directDir[3];
                                    directDir[0] = randPos[0] - posHit[0];
                                    directDir[1] = randPos[1] - posHit[1];
                                    directDir[2] = randPos[2] - posHit[2];
                                    float dirLen = sqrt(dot(directDir, directDir));
                                    directDir[0] /= dirLen; directDir[1] /= dirLen; directDir[2] /= dirLen;  

                                    //
                                    // Shadow Ray
                                    // Need to send shadow ray to see if point is in path of direct light
                                    bool shadowRayHit = false;
                                    float shadowDir[3] {directDir[0], directDir[1], directDir[2]};
                                    for (int ind = 0; ind <  sConstants.numShapes; ind++) {
                                        if (ind == impShape)
                                            continue;
                                        int shapeType = sConstants.shapes[ind][0];
                                        int sMatInd = sConstants.shapes[ind][1];
                                        int aInd = sConstants.shapes[ind][2];
                                        float tempT = INFINITY;
                                        float obbPosHit[3];
                                        // ----- intersect shapes -----
                                        // aabb
                                        if ( shapeType == 1) {
                                            // Transform Ray
                                            float rDir[3] = {shadowDir[0], shadowDir[1], shadowDir[2]};
                                            float boxPos[3] = {sConstants.objAttributes[aInd + 0], sConstants.objAttributes[aInd + 1], sConstants.objAttributes[aInd + 2]};
                                            float rPos[3] = {posHit[0]-boxPos[0], posHit[1]-boxPos[1], posHit[2]-boxPos[2]};
                                            float rot[4] = {sConstants.objAttributes[aInd + 9], sConstants.objAttributes[aInd + 10], sConstants.objAttributes[aInd + 11], sConstants.objAttributes[aInd + 12]};
                                            if (rot[1] + rot[2] + rot[3] > E) {
                                                rotate(rDir,rot);
                                                norm(rDir); 
                                                rotate(rPos,rot);
                                            }
                                            rPos[0]+=boxPos[0];rPos[1]+=boxPos[1];rPos[2]+=boxPos[2];

                                            int sign[3] = {rDir[0] < 0, rDir[1] < 0, rDir[2] < 0};
                                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                                            bounds[0][0] =  sConstants.objAttributes[aInd + 3];
                                            bounds[0][1] =  sConstants.objAttributes[aInd + 4];
                                            bounds[0][2] =  sConstants.objAttributes[aInd + 5];
                                            bounds[1][0] =  sConstants.objAttributes[aInd + 6];
                                            bounds[1][1] =  sConstants.objAttributes[aInd + 7];
                                            bounds[1][2] =  sConstants.objAttributes[aInd + 8];
                                            float tmin = (bounds[sign[0]][0] - rPos[0]) / rDir[0];
                                            float tmax = (bounds[1 - sign[0]][0] - rPos[0]) / rDir[0];
                                            float tymin = (bounds[sign[1]][1] - rPos[1]) / rDir[1];
                                            float tymax = (bounds[1 - sign[1]][1] - rPos[1]) / rDir[1];
                                            if ((tmin > tymax) || (tymin > tmax))
                                                continue;
                                            if (tymin > tmin)
                                                tmin = tymin;
                                            if (tymax < tmax)
                                                tmax = tymax;
                                            float tzmin = (bounds[sign[2]][2] - rPos[2]) / rDir[2];
                                            float tzmax = (bounds[1 - sign[2]][2] - rPos[2]) / rDir[2];
                                            if ((tmin > tzmax) || (tzmin > tmax))
                                                continue;
                                            if (tzmin > tmin)
                                                tmin = tzmin;
                                            if (tzmax < tmax)
                                                tmax = tzmax;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = tmin > E ? tmin : (tmax > E ? tmax : INFINITY); 

                                            if ((int)sConstants.matList[sMatInd][5]==3) {
                                                obbPosHit[0] = rPos[0] + rDir[0]*tempT;
                                                obbPosHit[1] = rPos[1] + rDir[1]*tempT;
                                                obbPosHit[2] = rPos[2] + rDir[2]*tempT;
                                            }  
                                        }
                                        // sphere
                                        else if (shapeType == 0) {
                                            float L[3];
                                            L[0] =  sConstants.objAttributes[aInd+0] - posHit[0];
                                            L[1] =  sConstants.objAttributes[aInd+1] - posHit[1];
                                            L[2] =  sConstants.objAttributes[aInd+2] - posHit[2];
                                            float tca = L[0]*shadowDir[0] + L[1]*shadowDir[1] + L[2]*shadowDir[2];
                                            if (tca < E)
                                                continue;
                                            float dsq = dot(L,L) - tca * tca;
                                            float radiusSq =  sConstants.objAttributes[aInd+3] *  sConstants.objAttributes[aInd+3];
                                            if (radiusSq - dsq < E)
                                                continue;
                                            float thc = sqrt(radiusSq - dsq);
                                            float t0 = tca - thc;
                                            float t1 = tca + thc;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = t0 > E ? t0 : (t1 > E ? t1 : INFINITY); 
                                            
                                        }
                                        if (tempT < dirLen) {
                                            // Dialectric Check
                                            if ((int)sConstants.matList[sMatInd][5]==3) {
                                                float blur = sConstants.matList[sMatInd][3];
                                                float RI = 1.0f / sConstants.matList[sMatInd][4];
                                                // Get Normal
                                                float refNorm[3] {0.0f, 0.0f, 0.0f};
                                                if (shapeTypeHit == 1) {
                                                    float bounds[2][3] = {{0,0,0}, {0,0,0}};
                                                    bounds[0][0] =  sConstants.objAttributes[attrInd + 3];
                                                    bounds[0][1] =  sConstants.objAttributes[attrInd + 4];
                                                    bounds[0][2] =  sConstants.objAttributes[attrInd + 5];
                                                    bounds[1][0] =  sConstants.objAttributes[attrInd + 6];
                                                    bounds[1][1] =  sConstants.objAttributes[attrInd + 7];
                                                    bounds[1][2] =  sConstants.objAttributes[attrInd + 8];

                                                    // Flat 
                                                    if (fabs(bounds[0][0] - bounds[1][0]) < E) {
                                                        refNorm[0] = shadowDir[0] > 0 ? -1 : 1;
                                                    }
                                                    else if (fabs(bounds[0][1] - bounds[1][1]) < E) {
                                                        refNorm[1] = shadowDir[1] > 0 ? -1 : 1;
                                                    }
                                                    else if (fabs(bounds[0][2] - bounds[1][2]) < E) {
                                                        refNorm[2] = shadowDir[2] > 0 ? -1 : 1;
                                                    }
                                                    // Not Flat
                                                    else if (fabs(obbPosHit[0] - bounds[0][0]) < E)
                                                        refNorm[0] = -1;
                                                    else if (fabs(obbPosHit[0] - bounds[1][0]) < E)
                                                        refNorm[0] = 1;
                                                    else if (fabs(obbPosHit[1] - bounds[0][1]) < E)
                                                        refNorm[1] = -1;
                                                    else if (fabs(obbPosHit[1] - bounds[1][1]) < E)
                                                        refNorm[1] = 1;
                                                    else if (fabs(obbPosHit[2] - bounds[0][0]) < E)
                                                        refNorm[2] = -1;
                                                    else if (fabs(obbPosHit[2] - bounds[1][0]) < E)
                                                        refNorm[2] = 1;

                                                    // Transform Normal
                                                    float rot[4] = {sConstants.objAttributes[attrInd + 9], -sConstants.objAttributes[attrInd + 10], -sConstants.objAttributes[attrInd + 11], -sConstants.objAttributes[attrInd + 12]};
                                                    rotate(refNorm, rot);
                                                    norm(refNorm);
                                                }
                                                else if (shapeTypeHit == 0) {
                                                    float sPosHit[3];
                                                    sPosHit[0] = posHit[0] + shadowDir[0]*tempT;
                                                    sPosHit[1] = posHit[1] + shadowDir[1]*tempT;
                                                    sPosHit[2] = posHit[2] + shadowDir[2]*tempT;
                                                    refNorm[0] = sPosHit[0] -  sConstants.objAttributes[attrInd + 0];
                                                    refNorm[1] = sPosHit[1] -  sConstants.objAttributes[attrInd + 1];
                                                    refNorm[2] = sPosHit[2] -  sConstants.objAttributes[attrInd + 2];
                                                    norm(refNorm);
                                                } 
                                                float cosi = dot(shadowDir, refNorm);
                                                // If normal is same direction as ray, then flip
                                                if (cosi > 0) {
                                                    refNorm[0]*=-1.0f;refNorm[1]*=-1.0f;refNorm[2]*=-1.0f;
                                                    RI = 1.0f / RI;
                                                }
                                                else {
                                                    cosi*=-1.0f;
                                                }

                                                // Can refract check
                                                float sinSq = RI*RI*(1.0f-cosi*cosi);
                                                bool canRefract = 1.0f - sinSq > E;

                                                if (!canRefract) {
                                                    shadowDir[0] = (shadowDir[0] - 2.0f*cosi*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                                    shadowDir[1] = (shadowDir[1] - 2.0f*cosi*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                                    shadowDir[2] = (shadowDir[2] - 2.0f*cosi*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                                    norm(shadowDir);
                                                }
                                                else {

                                                    float refCalc = RI*cosi - (float)sqrt(1.0f-sinSq);
                                                    shadowDir[0] = (RI*shadowDir[0] + refCalc*refNorm[0])*(1.0f - blur) + blur*randDir[0];
                                                    shadowDir[1] = (RI*shadowDir[1] + refCalc*refNorm[1])*(1.0f - blur) + blur*randDir[1];
                                                    shadowDir[2] = (RI*shadowDir[2] + refCalc*refNorm[2])*(1.0f - blur) + blur*randDir[2];
                                                    norm(shadowDir);					
                                                }
                                                continue;
                                            }                                           
                                            shadowRayHit = true;
                                            break;
                                        }
                                    }

                                    if (!shadowRayHit) {
                                        float cosine = fabs(dot(directDir, randDir));
                                        if (cosine > 0.01f) {
                                            shadowRays[pos]=1; 
                                            dir[0] = directDir[0];
                                            dir[1] = directDir[1];
                                            dir[2] = directDir[2];
                                            p0 = fabs(cosine) / M_PI;
                                        }

                                    }
                                    // Shadow Ray
                                    //
                                    //


                                }
                                // 1
                                float p1 = 0;
                                if ( sConstants.shapes[impShape][0] == 1) {
                                    // AABB pdf val
                                    float areaSum = 0;
                                    float xDiff =  sConstants.objAttributes[impAttrInd+3] -  sConstants.objAttributes[impAttrInd+6];
                                    float yDiff =  sConstants.objAttributes[impAttrInd+4] -  sConstants.objAttributes[impAttrInd+7];
                                    float zDiff =  sConstants.objAttributes[impAttrInd+5] -  sConstants.objAttributes[impAttrInd+8];
                                    areaSum += xDiff * yDiff * 2.0f;
                                    areaSum += zDiff * yDiff * 2.0f;
                                    areaSum += xDiff * zDiff * 2.0f;
                                    float cosine = dot(dir, normals[pos]);
                                    cosine = cosine < 0.0001f ? 0.0001f : cosine;

                                    float diff[3];
                                    diff[0] =  sConstants.objAttributes[impAttrInd+0] - posHit[0];
                                    diff[1] =  sConstants.objAttributes[impAttrInd+1] - posHit[1];
                                    diff[2] =  sConstants.objAttributes[impAttrInd+2] - posHit[2];
                                    float dirLen = sqrt(dot(diff, diff));


                                    // AABB needs magic number for pdf calc, TODO: LOOK INTO, was too bright before
                                    //p1 = 1 / (cosine * areaSum);
                                    p1 = dirLen / (cosine * areaSum);

                                } else if ( sConstants.shapes[impShape][0] == 0) {
                                    // Sphere pdf val
                                    float diff[3] = { sConstants.objAttributes[impAttrInd+0]-posHit[0], 
                                                      sConstants.objAttributes[impAttrInd+1]-posHit[1], 
                                                      sConstants.objAttributes[impAttrInd+2]-posHit[2]};
                                    auto distance_squared = dot(diff, diff);
                                    auto cos_theta_max = sqrt(1.0f -  sConstants.objAttributes[impAttrInd+3] *  sConstants.objAttributes[impAttrInd+3] / distance_squared);
                                    // NaN check
                                    cos_theta_max = (cos_theta_max != cos_theta_max) ? 0.9999f : cos_theta_max;
                                    auto solid_angle = M_PI * (1.0f - cos_theta_max) *2.0f;

                                    // Sphere needs magic number for pdf calc, TODO: LOOK INTO, was too dark before
                                    //p1 = 1 / (solid_angle );
                                    p1 =  sConstants.objAttributes[impAttrInd+3] / (solid_angle * sqrt(distance_squared)*4.0f);
                                }

                                pdfVals[pos] = 0.5f*p0 + 0.5f*p1;
                            }
                        }
                    }

                    numShapeHit++;
                    rayPositions[pos][0] = posHit[0];
                    rayPositions[pos][1] = posHit[1];
                    rayPositions[pos][2] = posHit[2];
                    rayPositions[pos][3] = shapeHit;

                } else {
                    back_col[0] = 0.1f;
                    back_col[1] = 0.1f;
                    back_col[2] = (dir[1] + 1.0f)/2.2f + 0.1f;
                    break;
                }
            }
        }

        float finalCol[3] = {back_col[0], 
                            back_col[1], 
                            back_col[2]};

        // Reverse through hit points and add up colour
        for (int pos = numShapeHit-1; pos >=0; pos--) {

            int shapeHit = (int)rayPositions[pos][3];
            int matInd = sConstants.shapes[shapeHit][1];

            float albedo[3] = {  sConstants.matList[matInd][0], 
                                 sConstants.matList[matInd][1], 
                                 sConstants.matList[matInd][2]};
            int matType = (int) sConstants.matList[matInd][5];	
        
            float normal[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};

            float newDir[3];
            newDir[0] = pos == numShapeHit-1 ? dir[0] : rayPositions[pos + 1][0] - rayPositions[pos][0];
            newDir[1] = pos == numShapeHit-1 ? dir[1] : rayPositions[pos + 1][1] - rayPositions[pos][1];
            newDir[2] = pos == numShapeHit-1 ? dir[2] : rayPositions[pos + 1][2] - rayPositions[pos][2];
            if (pos < numShapeHit-1) {
                norm(newDir);
            }

            float emittance[3];
            emittance[0] = matType == 1 ? albedo[0] : 0;
            emittance[1] = matType == 1 ? albedo[1] : 0;
            emittance[2] = matType == 1 ? albedo[2] : 0;

            float pdf_val = pdfVals[pos]; 

            float cosine2 = dot(normal, newDir);

            float scattering_pdf = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;// just cosine/pi for lambertian
            float multVecs[3] = {albedo[0]*finalCol[0],   // albedo*incoming 
                                  albedo[1]*finalCol[1],  
                                  albedo[2]*finalCol[2]}; 

            float directLightMult = shadowRays[pos]==1 &&  sConstants.numImportantShapes>1 ?  sConstants.numImportantShapes : 1;

            float pdfs = scattering_pdf / pdf_val;
            finalCol[0] = emittance[0] + multVecs[0] * pdfs * directLightMult;
            finalCol[1] = emittance[1] + multVecs[1] * pdfs * directLightMult;
            finalCol[2] = emittance[2] + multVecs[2] * pdfs * directLightMult;
        }
        ReturnStruct ret;
        ret.xyz[0] = finalCol[0];
        ret.xyz[1] = finalCol[1];
        ret.xyz[2] = finalCol[2];
        if ( sConstants.getDenoiserInf == 1) {
            ret.normal[0] = normals[0][0];
            ret.normal[1] = normals[0][1];
            ret.normal[2] = normals[0][2];
            int matIndAlb1 = sConstants.shapes[(int)rayPositions[0][3]][1];
            ret.albedo1[0] =  sConstants.matList[matIndAlb1][0];
            ret.albedo1[1] =  sConstants.matList[matIndAlb1][1];
            ret.albedo1[2] =  sConstants.matList[matIndAlb1][2];
            int matIndAlb2 = sConstants.shapes[(int)rayPositions[1][3]][1];
            ret.albedo2[0] =  sConstants.matList[matIndAlb2][0];
            ret.albedo2[1] =  sConstants.matList[matIndAlb2][1];
            ret.albedo2[2] =  sConstants.matList[matIndAlb2][2];
            ret.directLight = 0.0f;
            for (int c = 0; c< sConstants.maxDepth; c++)
                ret.directLight += (float)shadowRays[c] / (float) sConstants.maxDepth;
            ret.worldPos[0] = rayPositions[0][0];
            ret.worldPos[1] = rayPositions[0][1];
            ret.worldPos[2] = rayPositions[0][2];
        }
        ret.raysSent = numRays;
        return ret;
   
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "skepu_skel_1_SkePURenderers_MapKernel_RenderFunc.cu"
void Renderers::SkePURender() {

        // Configure SkePU
        auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(skepuBackend)};
        spec.activateBackend();
        skepu::backend::Map<1, skepu_userfunction_skepu_skel_1renderFunc_RenderFunc, decltype(&skepu_skel_1_SkePURenderers_MapKernel_RenderFunc), void> renderFunc(skepu_skel_1_SkePURenderers_MapKernel_RenderFunc);
        renderFunc.setBackend(spec);
        auto outputContainer = skepu::Matrix<ReturnStruct>(yRes, xRes);
        auto seeds = skepu::Matrix<RandomSeeds>(yRes, xRes);

		// Generate random seeds for each pixel
		for (int j = 0; j < yRes; j++) {
			for (int i = 0; i < xRes; i++) {
				uint64_t s0 = GloRandS[0];
				uint64_t s1 = GloRandS[1];
				s1 ^= s0;
				GloRandS[0] = (s0 << 49) | ((s0 >> (64 - 49)) ^ s1 ^ (s1 << 21));
				GloRandS[1] = (s1 << 28) | (s1 >> (64 - 28));
				RandomSeeds s;
				s.s1 = GloRandS[0];
				s.s2 = GloRandS[1];
				seeds(j, i) = s;
			}
		}

		renderFunc(outputContainer, seeds, constants); 
		outputContainer.updateHost(); // TODO: Check speed difference when using [] vs () + updateHost
        ReturnStruct ret;
        int index;
        sampleCount++;
        rayCount = 0;
        DenoisingInf* info;
		for (int j = 0; j < yRes; j++) {
			for (int i = 0; i < xRes; i++) {
				ret = outputContainer(j, i);
				index = j*xRes + i;

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
		}        
    }
    
    void Renderers::UpdateConstants() {
        // Update Constants

        constants = Constants();

        UpdateCam(false);

        constants.maxAngleH = tan(M_PI * cam.hfov/360.0f);
        constants.maxAngleV = tan(M_PI * cam.vfov/360.0f);
        constants.focalLength = cam.focalLen;

        GloRandS[0] = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        GloRandS[1] = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        constants.maxDepth = maxDepth;
        constants.randSamp = randSamp;

        uint numImportantShapes = 0;
        for (int i = 0; i < std::min(10, (int)scene.importantList.size()); i++) {
            numImportantShapes++;
            constants.importantShapes[i] = scene.importantList[i];
        }
        constants.numImportantShapes = numImportantShapes;

        constants.getDenoiserInf = denoising ? 1 : 0;

        auto mats = scene.matList;
        for (int matInd = 0; matInd < std::min(50, (int)mats.size()); matInd++) {
            auto m = mats[matInd];
            constants.matList[matInd][0] = m->alb.x;
            constants.matList[matInd][1] = m->alb.y;
            constants.matList[matInd][2] = m->alb.z;   
            constants.matList[matInd][3] = m->blur;   
            constants.matList[matInd][4] = m->RI;   
            constants.matList[matInd][5] = m->matType;            
        }


        auto shapes = scene.objList;
        int numShapesAdded = 0;
        int numAttr = 0;
        for (int shapeInd = 0; shapeInd < std::min(50, (int)shapes.size()); shapeInd++) {
            auto s = shapes[shapeInd];
            constants.shapes[numShapesAdded][0] = s->type;
            constants.shapes[numShapesAdded][1] = s->mat_ind;
            constants.shapes[numShapesAdded][2] = numAttr;

            // Add Object Attributes
            if (s->type==0) {
                // Sphere
                auto sphere = std::dynamic_pointer_cast<Sphere>(s);
                constants.objAttributes[numAttr++] = sphere->pos.x;
                constants.objAttributes[numAttr++] = sphere->pos.y;
                constants.objAttributes[numAttr++] = sphere->pos.z;
                constants.objAttributes[numAttr++] = sphere->r; 
            }
            else if (s->type==1) {
                // AABB
                auto aabb = std::dynamic_pointer_cast<AABB>(s);
                constants.objAttributes[numAttr++] = aabb->pos.x;
                constants.objAttributes[numAttr++] = aabb->pos.y;
                constants.objAttributes[numAttr++] = aabb->pos.z;
                constants.objAttributes[numAttr++] = aabb->pos.x + aabb->min.x;
                constants.objAttributes[numAttr++] = aabb->pos.y + aabb->min.y;
                constants.objAttributes[numAttr++] = aabb->pos.z + aabb->min.z;
                constants.objAttributes[numAttr++] = aabb->pos.x + aabb->max.x;
                constants.objAttributes[numAttr++] = aabb->pos.y + aabb->max.y;
                constants.objAttributes[numAttr++] = aabb->pos.z + aabb->max.z;

                constants.objAttributes[numAttr++] = aabb->q[0];
                constants.objAttributes[numAttr++] = aabb->q[1];
                constants.objAttributes[numAttr++] = aabb->q[2];
                constants.objAttributes[numAttr++] = aabb->q[3];
            }

            numShapesAdded++;
        }
        constants.numShapes = numShapesAdded;

        CUDARender::UpdateConstants();
    }
    void Renderers::UpdateCam(bool c) {
        vec3 camPos = cam.pos;
        vec3 camForward = cam.forward;
        vec3 camRight = cam.right;
        vec3 camUp = cam.up;

        constants.camPos[0] = camPos.x; constants.camPos[1] = camPos.y; constants.camPos[2] = camPos.z;
        constants.camForward[0] = camForward.x; constants.camForward[1] = camForward.y; constants.camForward[2] = camForward.z;
        constants.camRight[0] = camRight.x; constants.camRight[1] = camRight.y; constants.camRight[2] = camRight.z;
        constants.camUp[0] = camUp.x; constants.camUp[1] = camUp.y; constants.camUp[2] = camUp.z;

        constants.RESH = xRes;
        constants.RESV = yRes;

        if (c) {
            CUDARender::UpdateConstants();    
        }
    }

    void Renderers::CPUAutoExp() {
        exposure = 0.0f;
        int numPixels = xRes*yRes;
        vec3 l_vec = vec3(0.2125,0.7154,0.0721);
        for (int ind =0; ind < numPixels; ind++) {
            exposure += 9.6f*l_vec.dot(preScreen[ind]/sampleCount)/numPixels;
        }
    }

    #include <mutex>
    void Renderers::OMPAutoExp() {
        exposure = 0.0f;
        int numPixels = xRes*yRes;
        vec3 l_vec = vec3(0.2125,0.7154,0.0721);
        std::mutex lock;

        #pragma omp parallel for
        for (int ind =0; ind < numPixels; ind++) {
            std::lock_guard<std::mutex> guard(lock);
            exposure += 9.6f*l_vec.dot(preScreen[ind]/sampleCount)/numPixels;
        }
    }

    static float SkePUExposureFuncCalc(vec3 col, float div) {
        return (0.2125f * col.x + 0.7154 * col.y + 0.0721 * col.z)/div;
    }

    
struct skepu_userfunction_skepu_skel_0exposureFunc_SkePUExposureFuncCalc
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
constexpr static bool usesPRNG = 0;
constexpr static size_t randomCount = SKEPU_NO_RANDOM;
using IndexType = void;
using ElwiseArgs = std::tuple<class vec3>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<float>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(class vec3 col, float div)
{
        return (0.2125f * col.x + 0.7154 * col.y + 0.0721 * col.z)/div;
   
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(class vec3 col, float div)
{
        return (0.2125f * col.x + 0.7154 * col.y + 0.0721 * col.z)/div;
   
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(class vec3 col, float div)
{
        return (0.2125f * col.x + 0.7154 * col.y + 0.0721 * col.z)/div;
   
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_0exposureFunc_add_float
{
using T = float;
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
constexpr static bool usesPRNG = 0;
constexpr static size_t randomCount = SKEPU_NO_RANDOM;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float lhs, float rhs)
{
  return lhs + rhs;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float lhs, float rhs)
{
  return lhs + rhs;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float lhs, float rhs)
{
  return lhs + rhs;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "skepu_skel_0_SkePURenderers_MapReduceKernel_SkePUExposureFuncCalc_add_float.cu"
void Renderers::SkePUAutoExp() {

        int numPixels = xRes*yRes;
        auto screen = skepu::Vector<vec3>(numPixels);

        for (int ind =0; ind < numPixels; ind++) {
            screen(ind) = preScreen[ind];
        }

        // Configure SkePU
        auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(skepuBackend)};
        spec.activateBackend();
        skepu::backend::MapReduce<1, skepu_userfunction_skepu_skel_0exposureFunc_SkePUExposureFuncCalc, skepu_userfunction_skepu_skel_0exposureFunc_add_float, decltype(&skepu_skel_0_SkePURenderers_MapReduceKernel_SkePUExposureFuncCalc_add_float), decltype(&skepu_skel_0_SkePURenderers_MapReduceKernel_SkePUExposureFuncCalc_add_float_ReduceOnly), void> exposureFunc(skepu_skel_0_SkePURenderers_MapReduceKernel_SkePUExposureFuncCalc_add_float, skepu_skel_0_SkePURenderers_MapReduceKernel_SkePUExposureFuncCalc_add_float_ReduceOnly);
        exposureFunc.setBackend(spec);

        exposure = 9.6f * exposureFunc(screen, numPixels*sampleCount);
    }


