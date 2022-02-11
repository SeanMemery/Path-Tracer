#include "ManageConstants.h"

    void ManageConstants::UpdateConstants() {
        // Update Constants

        constants = Constants();

        UpdateCam();

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
        for (int shapeInd = 0; shapeInd < std::min(50, (int)shapes.size()); shapeInd++) {
            auto s = shapes[shapeInd];
            constants.shapes[numShapesAdded][0] = s->type;
            constants.shapes[numShapesAdded][1] = s->mat_ind;
            constants.shapes[numShapesAdded][2] = objAttributes.size();

            // Add Object Attributes
            if (s->type==0) {
                // Sphere
                auto sphere = std::dynamic_pointer_cast<Sphere>(s);
                objAttributes.push_back(sphere->pos.x);
                objAttributes.push_back(sphere->pos.y);
                objAttributes.push_back(sphere->pos.z);
                objAttributes.push_back(sphere->r);            
            }
            else if (s->type==1) {
                // AABB
                auto aabb = std::dynamic_pointer_cast<AABB>(s);
                objAttributes.push_back(aabb->pos.x);
                objAttributes.push_back(aabb->pos.y);
                objAttributes.push_back(aabb->pos.z);
                objAttributes.push_back(aabb->pos.x + aabb->min.x);
                objAttributes.push_back(aabb->pos.y + aabb->min.y);
                objAttributes.push_back(aabb->pos.z + aabb->min.z);
                objAttributes.push_back(aabb->pos.x + aabb->max.x);
                objAttributes.push_back(aabb->pos.y + aabb->max.y);
                objAttributes.push_back(aabb->pos.z + aabb->max.z);
            }
            else if (s->type==1) {
                // Model
                auto model = std::dynamic_pointer_cast<Model>(s);
                objAttributes.push_back(model->pos.x);
                objAttributes.push_back(model->pos.y);
                objAttributes.push_back(model->pos.z);
                objAttributes.push_back(model->scale);
                objAttributes.push_back(model->vert_ind);
                objAttributes.push_back(model->num_vertices);
            }

            numShapesAdded++;
        }
        constants.numShapes = numShapesAdded;

        CUDARender::UpdateConstants();
    }
    void ManageConstants::UpdateCam() {
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

        CUDARender::UpdateCam();
    }



