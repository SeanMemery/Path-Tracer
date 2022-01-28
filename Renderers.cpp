#include "Renderers.h"

    static ReturnStruct RenderFunc(skepu::Index2D ind, RandomSeeds seeds, Constants constants) {
        // Ray
        double camPos[3] = {constants.camPos[0], constants.camPos[1], constants.camPos[2]};
        double rayPos[3] = {camPos[0], camPos[1], camPos[2]};

        // Pixel Coord
        double camForward[3] = {constants.camForward[0], constants.camForward[1], constants.camForward[2]};
        double camRight[3] = {constants.camRight[0], constants.camRight[1], constants.camRight[2]};

        double pY = -constants.maxAngleV + 2*constants.maxAngleV*((double)ind.row/(double)constants.RESV);
        double pX = -constants.maxAngleH + 2*constants.maxAngleH*((double)ind.col/(double)constants.RESH);

        double pix[3] = {0,0,0};
        pix[0] = camPos[0] + constants.camForward[0]*constants.focalLength + constants.camRight[0]*pX + constants.camUp[0]*pY;
        pix[1] = camPos[1] + constants.camForward[1]*constants.focalLength + constants.camRight[1]*pX + constants.camUp[1]*pY;
        pix[2] = camPos[2] + constants.camForward[2]*constants.focalLength + constants.camRight[2]*pX + constants.camUp[2]*pY;

        double rayDir[3] = {pix[0]-camPos[0], pix[1]-camPos[1], pix[2]-camPos[2]};
        double n1 = sqrt(rayDir[0]*rayDir[0] + rayDir[1]*rayDir[1] + rayDir[2]*rayDir[2]);
        rayDir[0]/=n1;
        rayDir[1]/=n1;
        rayDir[2]/=n1;   

        // Random seeds
        uint64_t randSeeds[2];
        randSeeds[0] = seeds.s1; 
        randSeeds[1] = seeds.s2; 


        // Store ray collisions and reverse through them (last num is shape index)
        double rayPositions[5][4] = {{0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}};
        double normals[5][3] = {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}};
        double defaultPdf = 1.0 / M_PI;
        double pdfVals[5] = {defaultPdf, defaultPdf, defaultPdf, defaultPdf, defaultPdf};
        int shadowRays[5] = {0,0,0,0,0};

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
        double dir[3] = {rayDir[0], rayDir[1], rayDir[2]};
        int BOUNCES = 5;
        for (int pos = 0; pos < BOUNCES; pos++) {
            int shapeHit;
            double prevPos[3];
            if (pos > 0) {
                prevPos[0] = rayPositions[pos-1][0];
                prevPos[1] = rayPositions[pos-1][1];
                prevPos[2] = rayPositions[pos-1][2];
            } else {
                prevPos[0] = camPos[0];
                prevPos[1] = camPos[1];
                prevPos[2] = camPos[2];
            }
            double posHit[3];
            bool hitAnything = false;
            int shapeTypeHit;
            // Collide with shapes, generating new dirs as needed (i.e. random or specular)
            {

                double E = 0.00001f;

                // Find shape
                {
                    double t = INFINITY;
                    for (int ind = 0; ind < constants.numShapes; ind++) {
                        int shapeType = (int)constants.shapes[ind][7];
                        double tempT = INFINITY;
                        // ----- intersect shapes -----
                        // aabb
                        if ( shapeType == 0) {
                            int sign[3] = {dir[0] < 0, dir[1] < 0, dir[2] < 0};
                            double bounds[2][3] = {{0,0,0}, {0,0,0}};
                            bounds[0][0] = constants.shapes[ind][8];
                            bounds[0][1] = constants.shapes[ind][9];
                            bounds[0][2] = constants.shapes[ind][10];
                            bounds[1][0] = constants.shapes[ind][11];
                            bounds[1][1] = constants.shapes[ind][12];
                            bounds[1][2] = constants.shapes[ind][13];
                            double tmin = (bounds[sign[0]][0] - prevPos[0]) / dir[0];
                            double tmax = (bounds[1 - sign[0]][0] - prevPos[0]) / dir[0];
                            double tymin = (bounds[sign[1]][1] - prevPos[1]) / dir[1];
                            double tymax = (bounds[1 - sign[1]][1] - prevPos[1]) / dir[1];
                            if ((tmin > tymax) || (tymin > tmax))
                                continue;
                            if (tymin > tmin)
                                tmin = tymin;
                            if (tymax < tmax)
                                tmax = tymax;
                            double tzmin = (bounds[sign[2]][2] - prevPos[2]) / dir[2];
                            double tzmax = (bounds[1 - sign[2]][2] - prevPos[2]) / dir[2];
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
                            double L[3] = {0,0,0};
                            L[0] = constants.shapes[ind][0] - prevPos[0];
                            L[1] = constants.shapes[ind][1] - prevPos[1];
                            L[2] = constants.shapes[ind][2] - prevPos[2];
                            double tca = L[0]*dir[0] + L[1]*dir[1] + L[2]*dir[2];
                            if (tca < E)
                                continue;
                            double dsq = L[0]*L[0] + L[1]*L[1] + L[2]*L[2] - tca * tca;
                            double radiusSq = constants.shapes[ind][8] * constants.shapes[ind][8];
                            if (radiusSq - dsq < E)
                                continue;
                            double thc = sqrt(radiusSq - dsq);
                            double t0 = tca - thc;
                            double t1 = tca + thc;
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
                            double bounds[2][3] = {{0,0,0}, {0,0,0}};
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
                            double n = sqrt(normals[pos][0]*normals[pos][0] +
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
                            double randDir[3];
                            double rands[3];
                            {
                                // Rand vals
                                for (int n = 0; n < 3; n++) {
                                    uint64_t s0 = randSeeds[0];
                                    uint64_t s1 = randSeeds[1];
                                    uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                                    double one_two = ((uint64_t)1 << 63) * (double)2.0;
                                    rands[n] =  xorshiro / one_two;
                                    s1 ^= s0;
                                    randSeeds[0] = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                                    randSeeds[1] = (s1 << 28) | (s1 >> 36);
                                }

                                double axis[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
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
                                double n = sqrt(axis[1][0]*axis[1][0] + axis[1][1]*axis[1][1] + axis[1][2]*axis[1][2]);
                                axis[1][0] /= n; axis[1][1] /= n; axis[1][2] /= n;
                                // 0
                                // axis[0] = cross(axis[2], axis[1])
                                axis[0][0] = axis[2][1]*axis[1][2] - axis[2][2]*axis[1][1];
                                axis[0][1] = axis[2][2]*axis[1][0] - axis[2][0]*axis[1][2];
                                axis[0][2] = axis[2][0]*axis[1][1] - axis[2][1]*axis[1][0];
                                // rand dir
                                double phi = 2.0 * M_PI * rands[0];
                                double x = cos(phi) * sqrt(rands[1]);
                                double y = sin(phi) * sqrt(rands[1]);
                                double z = sqrt(1 - rands[1]);

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
                            double blur = constants.shapes[shapeHit][14];
                            double RI = 1.0 / constants.shapes[shapeHit][15];
                            double dirIn[3] = {dir[0], dir[1], dir[2]};
                            double refNorm[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};
                            double cosi = dirIn[0]*refNorm[0] + dirIn[1]*refNorm[1] + dirIn[2]*refNorm[2];

                            // If normal is same direction as ray, then flip
                            if (cosi > 0) {
                                refNorm[0]*=-1.0;refNorm[1]*=-1.0;refNorm[2]*=-1.0;
                                RI = 1.0 / RI;
                            }
                            else {
                                cosi*=-1.0;
                            }

                            // Can refract check
                            double sinSq = RI*RI*(1-cosi*cosi);
                            bool canRefract = 1 - sinSq > 0;
                            
                            // Schlick approx
                            double r0 = (1.0 - RI) / (1.0 + RI);
                            r0 = r0 * r0;
                            double schlick = r0 + (1.0 - r0) * pow((1.0 - cosi), 5.0);

                            if (!canRefract){//} || schlick > rands[2]) {
                                dir[0] = dirIn[0] - 2*cosi*refNorm[0] + blur*randDir[0];
                                dir[1] = dirIn[1] - 2*cosi*refNorm[1] + blur*randDir[1];
                                dir[2] = dirIn[2] - 2*cosi*refNorm[2] + blur*randDir[2];
                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                double cosine2 = normals[pos][0] * dir[0] + 
                                                normals[pos][1] * dir[1] + 
                                                normals[pos][2] * dir[2];
                                pdfVals[pos] = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;
                            }
                            else {

                                double refCalc = RI*cosi - sqrt(1-sinSq);
                                dir[0] = RI*dirIn[0] + refCalc*refNorm[0] + blur*randDir[0];
                                dir[1] = RI*dirIn[1] + refCalc*refNorm[1] + blur*randDir[1];
                                dir[2] = RI*dirIn[2] + refCalc*refNorm[2] + blur*randDir[2];

                                double length = sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
                                dir[0]/=length;dir[1]/=length;dir[2]/=length;

                                // pdf val as scattering pdf, to make total pdf 1 as ray is specular
                                // Danger here, scattering pdf was going to 0 for refracting and making colour explode
                                double cosine2 = normals[pos][0] * dir[0] + 
                                                normals[pos][1] * dir[1] + 
                                                normals[pos][2] * dir[2];
                                pdfVals[pos] = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;					
                            }
                        }
                        else if (constants.shapes[shapeHit][6] == 2) {
                            // Metal material

                            double dirIn[3] = {dir[0], dir[1], dir[2]};
                            double blur = constants.shapes[shapeHit][14];

                            double prevDirNormalDot = dirIn[0]*normals[pos][0] + 
                                                    dirIn[1]*normals[pos][1] + 
                                                    dirIn[2]*normals[pos][2];

                            dir[0] = dirIn[0] - 2*prevDirNormalDot*normals[pos][0] + blur*randDir[0];
                            dir[1] = dirIn[1] - 2*prevDirNormalDot*normals[pos][1] + blur*randDir[1];
                            dir[2] = dirIn[2] - 2*prevDirNormalDot*normals[pos][2] + blur*randDir[2];

                            double cosine2 = normals[pos][0] * dir[0] + 
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

                            // Get importance shape
                            bool mixPdf = constants.numImportantShapes > 0;
                            int impInd;
                            int impShape;
                            if (mixPdf) { 
                                impInd = rands[2] * constants.numImportantShapes * 0.99999f;
                                impShape = constants.importantShapes[impInd];
                                if (impShape==shapeHit) {
                                    mixPdf = constants.numImportantShapes > 1;
                                    impInd = (impInd+1) % constants.numImportantShapes;
                                    impShape = constants.importantShapes[impInd];
                                } 
                            }
                            // Calculate PDF val
                            if (mixPdf) {
                                // 0
                                double p0 = 1 / M_PI;
                                int choosePdf = rands[2] * 2.0f;
                                // dot(randomly generated dir, ray dir) / PI
                                if (choosePdf == 1) {
                                    // Generate dir towards importance shape
                                    double randPos[3] = {0,0,0};
                                    if (constants.shapes[impShape][7] == 0) {
                                        // Gen three new random variables : [0, 1]
                                        double aabbRands[3];
                                        for (int n = 0; n < 3; n++) {
                                            uint64_t s0 = randSeeds[0];
                                            uint64_t s1 = randSeeds[1];
                                            uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                                            double one_two = ((uint64_t)1 << 63) * (double)2.0;
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
                                        double sphereRands[3];
                                        for (int n = 0; n < 3; n++) {
                                            uint64_t s0 = randSeeds[0];
                                            uint64_t s1 = randSeeds[1];
                                            uint64_t xorshiro = (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + s0;
                                            double one_two = ((uint64_t)1 << 63) * (double)2.0;
                                            sphereRands[n] =  (xorshiro / one_two)*2.0f - 1.0f;
                                            s1 ^= s0;
                                            randSeeds[0] = (((s0 << 49) | ((s0 >> 15))) ^ s1 ^ (s1 << 21));
                                            randSeeds[1] = (s1 << 28) | (s1 >> 36);
                                        }
                                        double sphereN = sqrt(sphereRands[0]*sphereRands[0] + 
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
                                    double dirLen = sqrt(directDir[0]*directDir[0] + directDir[1]*directDir[1] + directDir[2]*directDir[2]);
                                    directDir[0] /= dirLen; directDir[1] /= dirLen; directDir[2] /= dirLen;  

                                    //
                                    // Shadow Ray
                                    // Need to send shadow ray to see if point is in path of direct light
                                    double t = INFINITY;
                                    bool shadowRayHit = false;
                                    for (int ind = 0; ind < constants.numShapes; ind++) {
                                        if (ind == impShape)
                                            continue;
                                        int shapeType = (int)constants.shapes[ind][7];
                                        double tempT = INFINITY;
                                        // ----- intersect shapes -----
                                        // aabb
                                        if ( shapeType == 0) {
                                            int sign[3] = {directDir[0] < 0, directDir[1] < 0, directDir[2] < 0};
                                            double bounds[2][3] = {{0,0,0}, {0,0,0}};
                                            bounds[0][0] = constants.shapes[ind][8];
                                            bounds[0][1] = constants.shapes[ind][9];
                                            bounds[0][2] = constants.shapes[ind][10];
                                            bounds[1][0] = constants.shapes[ind][11];
                                            bounds[1][1] = constants.shapes[ind][12];
                                            bounds[1][2] = constants.shapes[ind][13];
                                            double tmin = (bounds[sign[0]][0] - posHit[0]) / directDir[0];
                                            double tmax = (bounds[1 - sign[0]][0] - posHit[0]) / directDir[0];
                                            double tymin = (bounds[sign[1]][1] - posHit[1]) / directDir[1];
                                            double tymax = (bounds[1 - sign[1]][1] - posHit[1]) / directDir[1];
                                            if ((tmin > tymax) || (tymin > tmax))
                                                continue;
                                            if (tymin > tmin)
                                                tmin = tymin;
                                            if (tymax < tmax)
                                                tmax = tymax;
                                            double tzmin = (bounds[sign[2]][2] - posHit[2]) / directDir[2];
                                            double tzmax = (bounds[1 - sign[2]][2] - posHit[2]) / directDir[2];
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
                                            double L[3] = {0,0,0};
                                            L[0] = constants.shapes[ind][0] - posHit[0];
                                            L[1] = constants.shapes[ind][1] - posHit[1];
                                            L[2] = constants.shapes[ind][2] - posHit[2];
                                            double tca = L[0]*directDir[0] + L[1]*directDir[1] + L[2]*directDir[2];
                                            if (tca < E)
                                                continue;
                                            double dsq = L[0]*L[0] + L[1]*L[1] + L[2]*L[2] - tca * tca;
                                            double radiusSq = constants.shapes[ind][8] * constants.shapes[ind][8];
                                            if (radiusSq - dsq < E)
                                                continue;
                                            double thc = sqrt(radiusSq - dsq);
                                            double t0 = tca - thc;
                                            double t1 = tca + thc;
                                            // Check times are positive, but use E for floating point accuracy
                                            tempT = t0 > E ? t0 : (t1 > E ? t1 : INFINITY); 
                                        }
                                        if (tempT < dirLen) {
                                            shadowRayHit = true;
                                            break;
                                        }
                                    }
                                    // Shadow Ray
                                    //
                                    //

                                    if (!shadowRayHit) {

                                        double cosine = fabs(directDir[0]*randDir[0] + directDir[1]*randDir[1] + directDir[2]*randDir[2]);
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
                                }
                                // 1
                                double p1 = 0;
                                if (constants.shapes[impShape][7] == 0) {
                                    // AABB pdf val
                                    double areaSum = 0;
                                    double xDiff = constants.shapes[impShape][11] - constants.shapes[impShape][8];
                                    double yDiff = constants.shapes[impShape][12] - constants.shapes[impShape][9];
                                    double zDiff = constants.shapes[impShape][13] - constants.shapes[impShape][10];
                                    areaSum += xDiff * yDiff * 2.0f;
                                    areaSum += zDiff * yDiff * 2.0f;
                                    areaSum += xDiff * zDiff * 2.0f;
                                    double cosine = dir[0]*normals[pos][0] + 
                                                    dir[1]*normals[pos][1] + 
                                                    dir[2]*normals[pos][2];
                                    cosine = cosine < 0.0001f ? 0.0001f : cosine;
                                    p1 = 1 / (cosine * areaSum);
                                } else if (constants.shapes[impShape][7] == 1) {
                                    // Sphere pdf val
                                    double diff[3] = {constants.shapes[impShape][0]-posHit[0], 
                                                    constants.shapes[impShape][1]-posHit[1], 
                                                    constants.shapes[impShape][2]-posHit[2]};
                                    auto distance_squared = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
                                    auto cos_theta_max = sqrt(1 - constants.shapes[impShape][8] * constants.shapes[impShape][8] / distance_squared);
                                    // NaN check
                                    cos_theta_max = (cos_theta_max != cos_theta_max) ? 0.9999f : cos_theta_max;
                                    auto solid_angle = M_PI * (1.0f - cos_theta_max) *2.0f;
                                    p1 = 1 / solid_angle;
                                }

                                pdfVals[pos] = p0 + p1;
                            }
                        }
                    }

                    numShapeHit++;
                    rayPositions[pos][0] = posHit[0];
                    rayPositions[pos][1] = posHit[1];
                    rayPositions[pos][2] = posHit[2];
                    rayPositions[pos][3] = shapeHit;

                } else {
                    break;
                }
            }
        }

        double finalCol[3] = {constants.backgroundColour[0], 
                            constants.backgroundColour[1], 
                            constants.backgroundColour[2]};

        // Reverse through hit points and add up colour
        for (int pos = numShapeHit-1; pos >=0; pos--) {

            int shapeHit = (int)rayPositions[pos][3];

            double albedo[3] = {constants.shapes[shapeHit][3], 
                            constants.shapes[shapeHit][4], 
                            constants.shapes[shapeHit][5]};
            int matType = (int)constants.shapes[shapeHit][6];	
            int shapeType = (int)constants.shapes[shapeHit][7];
        
            double normal[3] = {normals[pos][0], normals[pos][1], normals[pos][2]};

            double newDir[3];
            newDir[0] = pos == numShapeHit-1 ? dir[0] : rayPositions[pos + 1][0] - rayPositions[pos][0];
            newDir[1] = pos == numShapeHit-1 ? dir[1] : rayPositions[pos + 1][1] - rayPositions[pos][1];
            newDir[2] = pos == numShapeHit-1 ? dir[2] : rayPositions[pos + 1][2] - rayPositions[pos][2];
            if (pos < numShapeHit-1) {

                double l2 = sqrt((newDir[0])*(newDir[0]) + 
                                (newDir[1])*(newDir[1]) + 
                                (newDir[2])*(newDir[2]));
                newDir[0] /= l2;
                newDir[1] /= l2;
                newDir[2] /= l2;
            }

            double emittance[3];
            emittance[0] = matType == 1 ? albedo[0] : 0;
            emittance[1] = matType == 1 ? albedo[1] : 0;
            emittance[2] = matType == 1 ? albedo[2] : 0;

            double pdf_val = pdfVals[pos]; 

            double cosine2 = normal[0] * newDir[0] + 
                            normal[1] * newDir[1] + 
                            normal[2] * newDir[2];

            double scattering_pdf = cosine2 < 0.00001f ? 0.00001f : cosine2 / M_PI;// just cosine/pi for lambertian
            double multVecs[3] = {albedo[0]*finalCol[0],   // albedo*incoming 
                                albedo[1]*finalCol[1],  
                                albedo[2]*finalCol[2]}; 

            double pdfs = scattering_pdf / pdf_val;
            finalCol[0] = emittance[0] + multVecs[0] * pdfs;
            finalCol[1] = emittance[1] + multVecs[1] * pdfs;
            finalCol[2] = emittance[2] + multVecs[2] * pdfs;
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
            ret.directLight = (shadowRays[0] + shadowRays[1] + shadowRays[2] + shadowRays[3] + shadowRays[4]) / 5.0f;
            ret.worldPos[0] = rayPositions[0][0];
            ret.worldPos[1] = rayPositions[0][1];
            ret.worldPos[2] = rayPositions[0][2];
        }
        ret.raysSent = numShapeHit;
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

                s.s1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                s.s2 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                auto skepuInd = skepu::Index2D();
                skepuInd.row = j;
                skepuInd.col = i;
                
				ret = RenderFunc(skepuInd, s, constants);
				index = j*xRes + i;

                preScreen[index] += vec3(ret.xyz[0], ret.xyz[1], ret.xyz[2]);
				rayCount += ret.raysSent;

				// Denoiser info
				if (denoising) {
					normal[index]      += vec3(ret.normal[0], ret.normal[1], ret.normal[2]);
					albedo1[index]     += vec3(ret.albedo1[0], ret.albedo1[1], ret.albedo1[2]);
					albedo2[index]     += vec3(ret.albedo2[0], ret.albedo2[1], ret.albedo2[2]);
					directLight[index] += vec3(ret.directLight, ret.directLight, ret.directLight);
					worldPos[index]    += vec3(ret.worldPos[0], ret.worldPos[1], ret.worldPos[2]);

					// Standard Deviations
					denoisingInf[index].stdDevVecs[0] += vec3(
                         pow(normal[index].x/sampleCount - ret.normal[0],2),
                         pow(normal[index].y/sampleCount - ret.normal[1],2),     
                         pow(normal[index].z/sampleCount - ret.normal[2],2));
					denoisingInf[index].stdDevVecs[1] += vec3(
                        pow(albedo1[index].x/sampleCount - ret.albedo1[0],2),
					    pow(albedo1[index].y/sampleCount - ret.albedo1[1],2),    
					    pow(albedo1[index].z/sampleCount - ret.albedo1[2],2));
					denoisingInf[index].stdDevVecs[2] += vec3(
                        pow(albedo2[index].x/sampleCount - ret.albedo2[0],2),
					    pow(albedo2[index].y/sampleCount - ret.albedo2[1],2),    
					    pow(albedo2[index].z/sampleCount - ret.albedo2[2],2));
					denoisingInf[index].stdDevVecs[3] += vec3(
                        pow(worldPos[index].x/sampleCount - ret.worldPos[0],2),
					    pow(worldPos[index].y/sampleCount - ret.worldPos[1],2),   
					    pow(worldPos[index].z/sampleCount - ret.worldPos[2],2));
					denoisingInf[index].stdDevVecs[4] += vec3(pow(directLight[index].x/sampleCount - ret.directLight,2),0,0);      
				}
			}
		}  
    }
    void Renderers::OMPRender() {
        ReturnStruct ret;
        int index;
        sampleCount++;
        rayCount = 0;
        RandomSeeds s;
        #pragma omp parallel for
		for (int j = 0; j < yRes; j++) {
            #pragma omp parallel for
			for (int i = 0; i < xRes; i++) {

                s.s1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                s.s2 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                auto skepuInd = skepu::Index2D();
                skepuInd.row = j;
                skepuInd.col = i;
                
				ret = RenderFunc(skepuInd, s, constants);
				index = j*xRes + i;

                preScreen[index] += vec3(ret.xyz[0], ret.xyz[1], ret.xyz[2]);
				rayCount += ret.raysSent;

				// Denoiser info
				if (denoising) {
					normal[index]      += vec3(ret.normal[0], ret.normal[1], ret.normal[2]);
					albedo1[index]     += vec3(ret.albedo1[0], ret.albedo1[1], ret.albedo1[2]);
					albedo2[index]     += vec3(ret.albedo2[0], ret.albedo2[1], ret.albedo2[2]);
					directLight[index] += vec3(ret.directLight, ret.directLight, ret.directLight);
					worldPos[index]    += vec3(ret.worldPos[0], ret.worldPos[1], ret.worldPos[2]);

					// Standard Deviations
					denoisingInf[index].stdDevVecs[0] += vec3(
                         pow(normal[index].x/sampleCount - ret.normal[0],2),
                         pow(normal[index].y/sampleCount - ret.normal[1],2),     
                         pow(normal[index].z/sampleCount - ret.normal[2],2));
					denoisingInf[index].stdDevVecs[1] += vec3(
                        pow(albedo1[index].x/sampleCount - ret.albedo1[0] ,2),
					    pow(albedo1[index].y/sampleCount - ret.albedo1[1],2),    
					    pow(albedo1[index].z/sampleCount - ret.albedo1[2],2));
					denoisingInf[index].stdDevVecs[2] += vec3(
                        pow(albedo2[index].x/sampleCount - ret.albedo2[0] ,2),
					    pow(albedo2[index].y/sampleCount - ret.albedo2[1],2),    
					    pow(albedo2[index].z/sampleCount - ret.albedo2[2],2));
					denoisingInf[index].stdDevVecs[3] += vec3(
                        pow(worldPos[index].x/sampleCount - ret.worldPos[0],2),
					    pow(worldPos[index].y/sampleCount - ret.worldPos[1],2),   
					    pow(worldPos[index].z/sampleCount - ret.worldPos[2],2));
					denoisingInf[index].stdDevVecs[4] += vec3(pow(directLight[index].x/sampleCount - ret.directLight,2),0,0);      
				}
			}
		}  
    }
    void Renderers::CUDARender() {

    }
    void Renderers::OpenGLRender() {

    }

    void Renderers::SkePURender() {

        // Configure SkePU
        auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(skepuBackend)};
        spec.activateBackend();
        auto renderFunc = skepu::Map<1>(RenderFunc);
        renderFunc.setBackend(spec);
        auto outputContainer = skepu::Matrix<ReturnStruct>(constants.RESV, constants.RESH);
        auto seeds = skepu::Matrix<RandomSeeds>(constants.RESV, constants.RESH);

		// Generate random seeds for each pixel
		for (int j = 0; j < constants.RESV; j++) {
			for (int i = 0; i < constants.RESH; i++) {
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
		for (int j = 0; j < yRes; j++) {
			for (int i = 0; i < xRes; i++) {
				ret = outputContainer(j, i);
				index = j*xRes + i;

                preScreen[index] += vec3(ret.xyz[0], ret.xyz[1], ret.xyz[2]);
				rayCount += ret.raysSent;

				// Denoiser info
				if (denoising) {
					normal[index]      += vec3(ret.normal[0], ret.normal[1], ret.normal[2]);
					albedo1[index]     += vec3(ret.albedo1[0], ret.albedo1[1], ret.albedo1[2]);
					albedo2[index]     += vec3(ret.albedo2[0], ret.albedo2[1], ret.albedo2[2]);
					directLight[index] += vec3(ret.directLight, ret.directLight, ret.directLight);
					worldPos[index]    += vec3(ret.worldPos[0], ret.worldPos[1], ret.worldPos[2]);

					// Standard Deviations
					denoisingInf[index].stdDevVecs[0] += vec3(
                         pow(normal[index].x/sampleCount - ret.normal[0],2),
                         pow(normal[index].y/sampleCount - ret.normal[1],2),     
                         pow(normal[index].z/sampleCount - ret.normal[2],2));
					denoisingInf[index].stdDevVecs[1] += vec3(
                        pow(albedo1[index].x/sampleCount - ret.albedo1[0] ,2),
					    pow(albedo1[index].y/sampleCount - ret.albedo1[1],2),    
					    pow(albedo1[index].z/sampleCount - ret.albedo1[2],2));
					denoisingInf[index].stdDevVecs[2] += vec3(
                        pow(albedo2[index].x/sampleCount - ret.albedo2[0] ,2),
					    pow(albedo2[index].y/sampleCount - ret.albedo2[1],2),    
					    pow(albedo2[index].z/sampleCount - ret.albedo2[2],2));
					denoisingInf[index].stdDevVecs[3] += vec3(
                        pow(worldPos[index].x/sampleCount - ret.worldPos[0],2),
					    pow(worldPos[index].y/sampleCount - ret.worldPos[1],2),   
					    pow(worldPos[index].z/sampleCount - ret.worldPos[2],2));
					denoisingInf[index].stdDevVecs[4] += vec3(pow(directLight[index].x/sampleCount - ret.directLight,2),0,0);      
				}
			}
		}        
    }

    void Renderers::UpdateConstants() {
        // Update Constants

        constants = Constants();

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
        constants.maxAngleH = tan(M_PI * cam.hfov/360.0f);
        constants.maxAngleV = tan(M_PI * cam.vfov/360.0f);
        constants.focalLength = cam.focalLen;

        constants.GloRandS[0] = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        constants.GloRandS[1] = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        constants.backgroundColour[0] = 0.0f;
        constants.backgroundColour[1] = 0.0f;
        constants.backgroundColour[2] = 0.0f;

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




