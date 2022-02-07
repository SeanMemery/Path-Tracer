#include "Renderers.h"

    static ReturnStruct RenderFunc(skepu::Index2D ind, RandomSeeds seeds, Constants constants) {
        // Ray
        float camPos[3] = {constants.camPos[0], constants.camPos[1], constants.camPos[2]};
        float rayPos[3] = {camPos[0], camPos[1], camPos[2]};

        // Random seeds
        uint64_t randSeeds[2];
        randSeeds[0] = seeds.s1; 
        randSeeds[1] = seeds.s2; 

        // Rand Samp
        float rSamps[2] = {0.0f, 0.0f};
        if (constants.randSamp>0.001f) {
            // Rand vals
            for (int n = 0; n < 2; n++) {
                uint64_t s0 = randSeeds[0];
                uint64_t s1 = randSeeds[1];
                uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                float one_two = ((uint64_t)1 << 63) * (float)2.0;
                rSamps[n] = xorshiro / one_two;
                s1 ^= s0;
                randSeeds[0] = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                randSeeds[1] = (s1 << 28) | (s1 >> 36);
            }
            rSamps[0] *= 2;rSamps[1] *= 2;
            rSamps[0] -= 1;rSamps[1] -= 1;
            rSamps[0] *= constants.randSamp;rSamps[1] *= constants.randSamp;
        }


        float back_col[3] = {constants.backgroundColour[0], constants.backgroundColour[1], constants.backgroundColour[2]};

        // Pixel Coord
        float camForward[3] = {constants.camForward[0], constants.camForward[1], constants.camForward[2]};
        float camRight[3] = {constants.camRight[0], constants.camRight[1], constants.camRight[2]};

        float pY = -constants.maxAngleV + 2*constants.maxAngleV*((float)ind.row/(float)constants.RESV);
        float pX = -constants.maxAngleH + 2*constants.maxAngleH*((float)ind.col/(float)constants.RESH);

        float pix[3] = {0,0,0};
        pix[0] = camPos[0] + constants.camForward[0]*constants.focalLength + constants.camRight[0]*(pX+rSamps[0]) + constants.camUp[0]*(pY+rSamps[1]);
        pix[1] = camPos[1] + constants.camForward[1]*constants.focalLength + constants.camRight[1]*(pX+rSamps[0]) + constants.camUp[1]*(pY+rSamps[1]);
        pix[2] = camPos[2] + constants.camForward[2]*constants.focalLength + constants.camRight[2]*(pX+rSamps[0]) + constants.camUp[2]*(pY+rSamps[1]);

        float rayDir[3] = {pix[0]-camPos[0], pix[1]-camPos[1], pix[2]-camPos[2]};
        float n1 = sqrt(rayDir[0]*rayDir[0] + rayDir[1]*rayDir[1] + rayDir[2]*rayDir[2]);
        rayDir[0]/=n1;
        rayDir[1]/=n1;
        rayDir[2]/=n1;   

        // Store ray collisions and reverse through them (last num is shape index)
        float rayPositions[12][4];
        float normals[12][3];
        float pdfVals[12];
        // Shadow Rays: counts succesful shadow rays i.e. direct lighting, done for each bounce to provide more info
        int shadowRays[12];
        for (int v=0; v<12; v++){
            pdfVals[v] = 1.0 / (4.0f*M_PI);
            shadowRays[v] = 0.0f;
        }

        /*
            - loop through shapes
            - append to positions if hit shape, break if not
            - add shape index as 4th component 

            constants.shapes
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
        float dir[3] = {rayDir[0], rayDir[1], rayDir[2]};
        for (int pos = 0; pos < constants.maxDepth; pos++) {
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
            float posHit[3];
            bool hitAnything = false;
            int shapeTypeHit;
            // Collide with shapes, generating new dirs as needed (i.e. random or specular)
            {

                float E = 0.001f;

                // Find shape
                {
                    float t = INFINITY;
                    for (int ind = 0; ind < constants.numShapes; ind++) {
                        int shapeType = (int)constants.shapes[ind][7];
                        float tempT = INFINITY;
                        // ----- intersect shapes -----
                        // aabb
                        if ( shapeType == 0) {
                            int sign[3] = {dir[0] < 0, dir[1] < 0, dir[2] < 0};
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] = constants.shapes[ind][8];
                            bounds[0][1] = constants.shapes[ind][9];
                            bounds[0][2] = constants.shapes[ind][10];
                            bounds[1][0] = constants.shapes[ind][11];
                            bounds[1][1] = constants.shapes[ind][12];
                            bounds[1][2] = constants.shapes[ind][13];
                            float tmin = (bounds[sign[0]][0] - prevPos[0]) / dir[0];
                            float tmax = (bounds[1 - sign[0]][0] - prevPos[0]) / dir[0];
                            float tymin = (bounds[sign[1]][1] - prevPos[1]) / dir[1];
                            float tymax = (bounds[1 - sign[1]][1] - prevPos[1]) / dir[1];
                            if ((tmin > tymax) || (tymin > tmax))
                                continue;
                            if (tymin > tmin)
                                tmin = tymin;
                            if (tymax < tmax)
                                tmax = tymax;
                            float tzmin = (bounds[sign[2]][2] - prevPos[2]) / dir[2];
                            float tzmax = (bounds[1 - sign[2]][2] - prevPos[2]) / dir[2];
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
                            float L[3] = {0,0,0};
                            L[0] = constants.shapes[ind][0] - prevPos[0];
                            L[1] = constants.shapes[ind][1] - prevPos[1];
                            L[2] = constants.shapes[ind][2] - prevPos[2];
                            float tca = L[0]*dir[0] + L[1]*dir[1] + L[2]*dir[2];
                            if (tca < E)
                                continue;
                            float dsq = L[0]*L[0] + L[1]*L[1] + L[2]*L[2] - tca * tca;
                            float radiusSq = constants.shapes[ind][8] * constants.shapes[ind][8];
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
                            shapeTypeHit = shapeType;
                        }
                    }
                }

                if (hitAnything) {

                    // Get Normal
                    {
                        if (shapeTypeHit == 0) {
                            float bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] = constants.shapes[shapeHit][8];
                            bounds[0][1] = constants.shapes[shapeHit][9];
                            bounds[0][2] = constants.shapes[shapeHit][10];
                            bounds[1][0] = constants.shapes[shapeHit][11];
                            bounds[1][1] = constants.shapes[shapeHit][12];
                            bounds[1][2] = constants.shapes[shapeHit][13];
                            normals[pos][0] = 0;
                            normals[pos][1] = 0;
                            normals[pos][2] = 0;

                            // Flat 
                            if (fabs(bounds[0][0] - bounds[1][0]) < E) {
                                normals[pos][0] = dir[0] > 0 ? -1 : 1;
                            }
                            else if (fabs(bounds[0][1] - bounds[1][1]) < E) {
                                normals[pos][1] = dir[1] > 0 ? -1 : 1;
                            }
                            else if (fabs(bounds[0][2] - bounds[1][2]) < E) {
                                normals[pos][2] = dir[2] > 0 ? -1 : 1;
                            }
                            // Non Flat
                            else if (fabs(posHit[0] - bounds[0][0]) < E)
                                normals[pos][0] = -1;
                            else if (fabs(posHit[0] - bounds[1][0]) < E)
                                normals[pos][0] = 1;
                            else if (fabs(posHit[1] - bounds[0][1]) < E)
                                normals[pos][1] = -1;
                            else if (fabs(posHit[1] - bounds[1][1]) < E)
                                normals[pos][1] = 1;
                            else if (fabs(posHit[2] - bounds[0][2]) < E)
                                normals[pos][2] = -1;
                            else if (fabs(posHit[2] - bounds[1][2]) < E)
                                normals[pos][2] = 1;
                        }
                        else if (shapeTypeHit == 1) {
                            normals[pos][0] = posHit[0] - constants.shapes[shapeHit][0];
                            normals[pos][1] = posHit[1] - constants.shapes[shapeHit][1];
                            normals[pos][2] = posHit[2] - constants.shapes[shapeHit][2];
                            float n = sqrt(normals[pos][0]*normals[pos][0] +
                                        normals[pos][1]*normals[pos][1] + 
                                        normals[pos][2]*normals[pos][2]);
                            normals[pos][0] /= n;
                            normals[pos][1] /= n;
                            normals[pos][2] /= n;
                        }
                    }
                
                    // Gen new dirs and pdfs
                    {

                        // Random Dir Generation
                            float randDir[3];
                            float rands[5];
                            {
                                // Rand vals
                                for (int n = 0; n < 5; n++) {
                                    uint64_t s0 = randSeeds[0];
                                    uint64_t s1 = randSeeds[1];
                                    uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                                    float one_two = ((uint64_t)1 << 63) * (float)2.0;
                                    rands[n] =  xorshiro / one_two;
                                    s1 ^= s0;
                                    randSeeds[0] = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                                    randSeeds[1] = (s1 << 28) | (s1 >> 36);
                                }

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
                                    axis[1][2] = axis[2][0]; 
                                }
                                else {
                                    // axis[1] = cross(axis[2], [1,0,0])
                                    axis[1][1] = axis[2][2];
                                    axis[1][2] = -axis[2][1];
                                }
                                float n = sqrt(axis[1][0]*axis[1][0] + axis[1][1]*axis[1][1] + axis[1][2]*axis[1][2]);
                                axis[1][0] /= n; axis[1][1] /= n; axis[1][2] /= n;
                                // 0
                                // axis[0] = cross(axis[2], axis[1])
                                axis[0][0] = axis[2][1]*axis[1][2] - axis[2][2]*axis[1][1];
                                axis[0][1] = axis[2][2]*axis[1][0] - axis[2][0]*axis[1][2];
                                axis[0][2] = axis[2][0]*axis[1][1] - axis[2][1]*axis[1][0];
                                // rand dir
                                float phi = 2.0 * M_PI * rands[0];
                                float x = cos(phi) * sqrt(rands[1]);
                                float y = sin(phi) * sqrt(rands[1]);
                                float z = sqrt(1 - rands[1]);

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
                        if (constants.shapes[shapeHit][6] == 3) {
                            // Dielectric Material
                            shadowRays[pos] = 1;
                            float blur = constants.shapes[shapeHit][14];
                            float RI = 1.0 / constants.shapes[shapeHit][15];
                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float refNorm[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};
                            float cosi = dirIn[0]*refNorm[0] + dirIn[1]*refNorm[1] + dirIn[2]*refNorm[2];

                            // If normal is same direction as ray, then flip
                            if (cosi > 0) {
                                refNorm[0]*=-1.0;refNorm[1]*=-1.0;refNorm[2]*=-1.0;
                                RI = 1.0 / RI;
                            }
                            else {
                                cosi*=-1.0;
                            }

                            // Can refract check
                            float sinSq = RI*RI*(1-cosi*cosi);
                            bool canRefract = 1 - sinSq > 0;
                            
                            // Schlick approx
                            float r0 = (1.0 - RI) / (1.0 + RI);
                            r0 = r0 * r0;
                            float schlick = r0 + (1.0 - r0) * pow((1.0 - cosi), 5.0);

                            if (!canRefract){//} || schlick > rands[2]) {
                                dir[0] = dirIn[0] - 2*cosi*refNorm[0] + blur*randDir[0];
                                dir[1] = dirIn[1] - 2*cosi*refNorm[1] + blur*randDir[1];
                                dir[2] = dirIn[2] - 2*cosi*refNorm[2] + blur*randDir[2];
                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                float cosine2 = normals[pos][0] * dir[0] + 
                                                normals[pos][1] * dir[1] + 
                                                normals[pos][2] * dir[2];
                                pdfVals[pos] = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;
                            }
                            else {

                                float refCalc = RI*cosi - sqrt(1-sinSq);
                                dir[0] = RI*dirIn[0] + refCalc*refNorm[0] + blur*randDir[0];
                                dir[1] = RI*dirIn[1] + refCalc*refNorm[1] + blur*randDir[1];
                                dir[2] = RI*dirIn[2] + refCalc*refNorm[2] + blur*randDir[2];

                                float length = sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
                                dir[0]/=length;dir[1]/=length;dir[2]/=length;

                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                // Danger here, scattering pdf was going to 0 for refracting and making colour explode
                                float cosine2 = normals[pos][0] * dir[0] + 
                                                normals[pos][1] * dir[1] + 
                                                normals[pos][2] * dir[2];
                                pdfVals[pos] = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;					
                            }
                        }
                        else if (constants.shapes[shapeHit][6] == 2) {
                            // Metal material
                            shadowRays[pos] = 1;

                            float dirIn[3] = {dir[0], dir[1], dir[2]};
                            float blur = constants.shapes[shapeHit][14];

                            float prevDirNormalDot = dirIn[0]*normals[pos][0] + 
                                                    dirIn[1]*normals[pos][1] + 
                                                    dirIn[2]*normals[pos][2];

                            dir[0] = dirIn[0] - 2*prevDirNormalDot*normals[pos][0] + blur*randDir[0];
                            dir[1] = dirIn[1] - 2*prevDirNormalDot*normals[pos][1] + blur*randDir[1];
                            dir[2] = dirIn[2] - 2*prevDirNormalDot*normals[pos][2] + blur*randDir[2];

                            float cosine2 = normals[pos][0] * dir[0] + 
                                            normals[pos][1] * dir[1] + 
                                            normals[pos][2] * dir[2];

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
                            if (constants.shapes[shapeHit][6] == 1) {
                                shadowRays[pos] = 1;
                                mixPdf = false;
                            } else {
                                // Get importance shape
                                mixPdf = constants.numImportantShapes > 0;

                                if (mixPdf) { 
                                    impInd = rands[3] * constants.numImportantShapes * 0.99999f;
                                    impShape = constants.importantShapes[impInd];
                                    if (impShape==shapeHit) {
                                        mixPdf = false;
                                        // mixPdf = constants.numImportantShapes > 1;
                                        // impInd = (impInd+1) % constants.numImportantShapes;
                                        // impShape = constants.importantShapes[impInd];
                                    } 
                                }
                            }

                            // Calculate PDF val
                            if (mixPdf) {
                                // 0
                                float p0 = 1 / M_PI;
                                int choosePdf = rands[4] > 0.65f ? 1 : 0;
                                // dot(randomly generated dir, ray dir) / PI
                                if (choosePdf == 1) {
                                    // Generate dir towards importance shape
                                    float randPos[3] = {0,0,0};
                                    if (constants.shapes[impShape][7] == 0) {
                                        // Gen three new random variables : [0, 1]
                                        float aabbRands[3];
                                        for (int n = 0; n < 3; n++) {
                                            uint64_t s0 = randSeeds[0];
                                            uint64_t s1 = randSeeds[1];
                                            uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                                            float one_two = ((uint64_t)1 << 63) * (float)2.0;
                                            aabbRands[n] =  xorshiro / one_two;
                                            s1 ^= s0;
                                            randSeeds[0] = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                                            randSeeds[1] = (s1 << 28) | (s1 >> 36);
                                        }
                                        randPos[0] = (1.0f - aabbRands[0])*constants.shapes[impShape][8]  + aabbRands[0]*constants.shapes[impShape][11];
                                        randPos[1] = (1.0f - aabbRands[1])*constants.shapes[impShape][9]  + aabbRands[1]*constants.shapes[impShape][12];
                                        randPos[2] = (1.0f - aabbRands[2])*constants.shapes[impShape][10] + aabbRands[2]*constants.shapes[impShape][13];	
                                    } 
                                    else if (constants.shapes[impShape][7] == 1) {
                                        // Gen three new random variables : [-1, 1]
                                        float sphereRands[3];
                                        for (int n = 0; n < 3; n++) {
                                            uint64_t s0 = randSeeds[0];
                                            uint64_t s1 = randSeeds[1];
                                            uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                                            float one_two = ((uint64_t)1 << 63) * (float)2.0;
                                            sphereRands[n] =  (xorshiro / one_two)*2.0f - 1.0f;
                                            s1 ^= s0;
                                            randSeeds[0] = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                                            randSeeds[1] = (s1 << 28) | (s1 >> 36);
                                        }
                                        float sphereN = sqrt(sphereRands[0]*sphereRands[0] + 
                                                            sphereRands[1]*sphereRands[1] + 
                                                            sphereRands[2]*sphereRands[2]);
                                        sphereRands[0] /= sphereN; sphereRands[1] /= sphereN; sphereRands[2] /= sphereN;
                                        randPos[0] = constants.shapes[impShape][0] + sphereRands[0]*constants.shapes[impShape][8];
                                        randPos[1] = constants.shapes[impShape][1] + sphereRands[1]*constants.shapes[impShape][8];
                                        randPos[2] = constants.shapes[impShape][2] + sphereRands[2]*constants.shapes[impShape][8];
                                    }

                                    float directDir[3];
                                    directDir[0] = randPos[0] - posHit[0];
                                    directDir[1] = randPos[1] - posHit[1];
                                    directDir[2] = randPos[2] - posHit[2];
                                    float dirLen = sqrt(directDir[0]*directDir[0] + directDir[1]*directDir[1] + directDir[2]*directDir[2]);
                                    directDir[0] /= dirLen; directDir[1] /= dirLen; directDir[2] /= dirLen;  

                                    //
                                    // Shadow Ray
                                    // Need to send shadow ray to see if point is in path of direct light
                                    bool shadowRayHit = false;

                                    for (int ind = 0; ind < constants.numShapes; ind++) {
                                        if (ind == impShape)
                                            continue;
                                        int shapeType = (int)constants.shapes[ind][7];
                                        float tempT = INFINITY;
                                        // ----- intersect shapes -----
                                        // aabb
                                        if ( shapeType == 0) {
                                            int sign[3] = {dir[0] < 0, dir[1] < 0, dir[2] < 0};
                                            float bounds[2][3];
                                            bounds[0][0] = constants.shapes[ind][8];
                                            bounds[0][1] = constants.shapes[ind][9];
                                            bounds[0][2] = constants.shapes[ind][10];
                                            bounds[1][0] = constants.shapes[ind][11];
                                            bounds[1][1] = constants.shapes[ind][12];
                                            bounds[1][2] = constants.shapes[ind][13];
                                            float tmin = (bounds[sign[0]][0] - posHit[0]) / dir[0];
                                            float tmax = (bounds[1 - sign[0]][0] - posHit[0]) / dir[0];
                                            float tymin = (bounds[sign[1]][1] - posHit[1]) / dir[1];
                                            float tymax = (bounds[1 - sign[1]][1] - posHit[1]) / dir[1];
                                            if ((tmin > tymax) || (tymin > tmax))
                                                continue;
                                            if (tymin > tmin)
                                                tmin = tymin;
                                            if (tymax < tmax)
                                                tmax = tymax;
                                            float tzmin = (bounds[sign[2]][2] - posHit[2]) / dir[2];
                                            float tzmax = (bounds[1 - sign[2]][2] - posHit[2]) / dir[2];
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
                                            float L[3];
                                            L[0] = constants.shapes[ind][0] - posHit[0];
                                            L[1] = constants.shapes[ind][1] - posHit[1];
                                            L[2] = constants.shapes[ind][2] - posHit[2];
                                            float tca = L[0]*dir[0] + L[1]*dir[1] + L[2]*dir[2];
                                            if (tca < E)
                                                continue;
                                            float dsq = L[0]*L[0] + L[1]*L[1] + L[2]*L[2] - tca * tca;
                                            float radiusSq = constants.shapes[ind][8] * constants.shapes[ind][8];
                                            if (radiusSq - dsq < E)
                                                continue;
                                            float thc = sqrt(radiusSq - dsq);
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
                                        float cosine = fabs(directDir[0]*randDir[0] +directDir[1]*randDir[1] +directDir[2]*randDir[2]);
                                        if (cosine < 0.01) {
                                            p0 = 1 / M_PI;
                                        }
                                        else {
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
                                if (constants.shapes[impShape][7] == 0) {
                                    // AABB pdf val
                                    float areaSum = 0;
                                    float xDiff = constants.shapes[impShape][11] - constants.shapes[impShape][8];
                                    float yDiff = constants.shapes[impShape][12] - constants.shapes[impShape][9];
                                    float zDiff = constants.shapes[impShape][13] - constants.shapes[impShape][10];
                                    areaSum += xDiff * yDiff * 2.0f;
                                    areaSum += zDiff * yDiff * 2.0f;
                                    areaSum += xDiff * zDiff * 2.0f;
                                    float cosine = dir[0]*normals[pos][0] + 
                                                    dir[1]*normals[pos][1] + 
                                                    dir[2]*normals[pos][2];
                                    cosine = cosine < 0.0001f ? 0.0001f : cosine;

                                    float diff[3];
                                    diff[0] = constants.shapes[impShape][0] - posHit[0];
                                    diff[1] = constants.shapes[impShape][1] - posHit[1];
                                    diff[2] = constants.shapes[impShape][2] - posHit[2];
                                    float dirLen = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);


                                    // AABB needs magic number for pdf calc, TODO: LOOK INTO, was too bright before
                                    //p1 = 1 / (cosine * areaSum);
                                    p1 = dirLen / (cosine * areaSum);

                                } else if (constants.shapes[impShape][7] == 1) {
                                    // Sphere pdf val
                                    float diff[3] = {constants.shapes[impShape][0]-posHit[0], 
                                                    constants.shapes[impShape][1]-posHit[1], 
                                                    constants.shapes[impShape][2]-posHit[2]};
                                    auto distance_squared = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
                                    auto cos_theta_max = sqrt(1 - constants.shapes[impShape][8] * constants.shapes[impShape][8] / distance_squared);
                                    // NaN check
                                    cos_theta_max = (cos_theta_max != cos_theta_max) ? 0.9999f : cos_theta_max;
                                    auto solid_angle = M_PI * (1.0f - cos_theta_max) *2.0f;

                                    // Sphere needs magic number for pdf calc, TODO: LOOK INTO, was too dark before
                                    //p1 = 1 / (solid_angle );
                                    p1 = constants.shapes[impShape][8] / (solid_angle * sqrt(distance_squared)*4.0f);
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

            float albedo[3] = {constants.shapes[shapeHit][3], 
                                constants.shapes[shapeHit][4], 
                                constants.shapes[shapeHit][5]};
            int matType = (int)constants.shapes[shapeHit][6];	
            int shapeType = (int)constants.shapes[shapeHit][7];
        
            float normal[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};

            float newDir[3];
            newDir[0] = pos == numShapeHit-1 ? dir[0] : rayPositions[pos + 1][0] - rayPositions[pos][0];
            newDir[1] = pos == numShapeHit-1 ? dir[1] : rayPositions[pos + 1][1] - rayPositions[pos][1];
            newDir[2] = pos == numShapeHit-1 ? dir[2] : rayPositions[pos + 1][2] - rayPositions[pos][2];
            if (pos < numShapeHit-1) {

                float l2 = sqrt((newDir[0])*(newDir[0]) + 
                                (newDir[1])*(newDir[1]) + 
                                (newDir[2])*(newDir[2]));
                newDir[0] /= l2;
                newDir[1] /= l2;
                newDir[2] /= l2;
            }

            float emittance[3];
            emittance[0] = matType == 1 ? albedo[0] : 0;
            emittance[1] = matType == 1 ? albedo[1] : 0;
            emittance[2] = matType == 1 ? albedo[2] : 0;

            float pdf_val = pdfVals[pos]; 

            float cosine2 = normal[0] * newDir[0] + 
                            normal[1] * newDir[1] + 
                            normal[2] * newDir[2];

            float scattering_pdf = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;// just cosine/pi for lambertian
            float multVecs[3] = {albedo[0]*finalCol[0],   // albedo*incoming 
                                  albedo[1]*finalCol[1],  
                                  albedo[2]*finalCol[2]}; 

            float directLightMult = shadowRays[pos]==1 && constants.numImportantShapes>1 ? constants.numImportantShapes : 1;

            float pdfs = scattering_pdf / pdf_val;
            finalCol[0] = emittance[0] + multVecs[0] * pdfs * directLightMult;
            finalCol[1] = emittance[1] + multVecs[1] * pdfs * directLightMult;
            finalCol[2] = emittance[2] + multVecs[2] * pdfs * directLightMult;
        }
        ReturnStruct ret;
        ret.xyz[0] = finalCol[0];
        ret.xyz[1] = finalCol[1];
        ret.xyz[2] = finalCol[2];
        if (constants.getDenoiserInf == 1) {
            ret.normal[0] = normals[0][0];
            ret.normal[1] = normals[0][1];
            ret.normal[2] = normals[0][2];
            ret.albedo1[0] = constants.shapes[(int)rayPositions[0][3]][3];
            ret.albedo1[1] = constants.shapes[(int)rayPositions[0][3]][4];
            ret.albedo1[2] = constants.shapes[(int)rayPositions[0][3]][5];
            ret.albedo2[0] = constants.shapes[(int)rayPositions[1][3]][3];
            ret.albedo2[1] = constants.shapes[(int)rayPositions[1][3]][4];
            ret.albedo2[2] = constants.shapes[(int)rayPositions[1][3]][5];
            ret.directLight = 0.0f;
            for (int c = 0; c<constants.maxDepth; c++)
                ret.directLight += (float)shadowRays[c] / (float)constants.maxDepth;
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
				uint64_t s0 = constants.GloRandS[0];
				uint64_t s1 = constants.GloRandS[1];
				s1 ^= s0;
				constants.GloRandS[0] = (s0 << 49) | ((s0 >> (64 - 49)) ^ s1 ^ (s1 << 21));
				constants.GloRandS[1] = (s1 << 28) | (s1 >> (64 - 28));
				s.s1 = constants.GloRandS[0];
				s.s2 = constants.GloRandS[1];

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
				uint64_t s0 = constants.GloRandS[0];
				uint64_t s1 = constants.GloRandS[1];
				s1 ^= s0;
				constants.GloRandS[0] = (s0 << 49) | ((s0 >> (64 - 49)) ^ s1 ^ (s1 << 21));
				constants.GloRandS[1] = (s1 << 28) | (s1 >> (64 - 28));
				s.s1 = constants.GloRandS[0];
				s.s2 = constants.GloRandS[1];

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
    void Renderers::SkePURender() {

        // Configure SkePU
        auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(skepuBackend)};
        spec.activateBackend();
        auto renderFunc = skepu::Map<1>(RenderFunc);
        renderFunc.setBackend(spec);
        auto outputContainer = skepu::Matrix<ReturnStruct>(yRes, xRes);
        auto seeds = skepu::Matrix<RandomSeeds>(yRes, xRes);

		// Generate random seeds for each pixel
		for (int j = 0; j < yRes; j++) {
			for (int i = 0; i < xRes; i++) {
				uint64_t s0 = constants.GloRandS[0];
				uint64_t s1 = constants.GloRandS[1];
				s1 ^= s0;
				constants.GloRandS[0] = (s0 << 49) | ((s0 >> (64 - 49)) ^ s1 ^ (s1 << 21));
				constants.GloRandS[1] = (s1 << 28) | (s1 >> (64 - 28));
				RandomSeeds s;
				s.s1 = constants.GloRandS[0];
				s.s2 = constants.GloRandS[1];
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

        UpdateCam();

        constants.maxAngleH = tan(M_PI * cam.hfov/360.0f);
        constants.maxAngleV = tan(M_PI * cam.vfov/360.0f);
        constants.focalLength = cam.focalLen;

        constants.GloRandS[0] = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        constants.GloRandS[1] = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        constants.backgroundColour[0] = 0.0f;
        constants.backgroundColour[1] = 0.0f;
        constants.backgroundColour[2] = 0.0f;

        constants.maxDepth = maxDepth;
        constants.randSamp = randSamp;

        uint numImportantShapes = 0;
        for (int i = 0; i < std::min(5, (int)scene.importantList.size()); i++) {
            numImportantShapes++;
            constants.importantShapes[i] = scene.importantList[i];
        }
        constants.numImportantShapes = numImportantShapes;

        constants.getDenoiserInf = denoising ? 1 : 0;

        auto shapes = scene.objList;
        int numShapesAdded = 0;
        for (int shapeInd = 0; shapeInd < std::min(20, (int)shapes.size()); shapeInd++) {
            auto s = shapes[shapeInd];
            if (s->type == 0 || s->type == 1) {
                auto ret = s->GetData();
                for (int i = 0; i < ret.size(); i++) {
                    constants.shapes[numShapesAdded][i] = ret[i];
                }
                numShapesAdded++;
            }
        }
        constants.numShapes = numShapesAdded;
    }
    void Renderers::UpdateCam() {
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
    }



