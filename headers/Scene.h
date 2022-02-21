#pragma once

#include "GLOBALS.h"
#include "Obj.h"
#include <vector>
#include <memory>

class Scene {
public:
    void InitScene() {

        ResetScene();

       	matList.push_back(std::make_shared<Mat>(0, vec3(0.65, .05, .05)  , 0, 1)); // Red Wall
		matList.push_back(std::make_shared<Mat>(0, vec3(0.73, 0.73, 0.73), 0, 1));
		matList.push_back(std::make_shared<Mat>(0, vec3(0.73, 0.73, 0.73), 0, 1));
		matList.push_back(std::make_shared<Mat>(0, vec3(0.73, 0.73, 0.73), 0, 1));
		matList.push_back(std::make_shared<Mat>(0, vec3(0.73, 0.73, 0.73), 0, 1));
		matList.push_back(std::make_shared<Mat>(0, vec3(.12, 0.45, .15)  , 0, 1)); // Green Wall

		matList.push_back(std::make_shared<Mat>(1, vec3(1,1,1), 0, 1)); // Light

		double _wallDist = 10;
		double _wallRadius = 10;

		//objList.push_back(std::make_shared<Sphere>(vec3(0, _wallDist, 0), lightMat1, 1));
        objList.push_back(std::make_shared<AABB>(vec3(0, _wallDist-1.0f, 0), 6, vec3(-2.5f, 0, -2.5f), vec3(2.5f, 1.0f, 2.5f)));
        AddToImpList(0);

		objList.push_back(std::make_shared<AABB>(vec3(0,0,_wallDist), 1, vec3(-_wallRadius, -_wallRadius, 0), vec3(_wallRadius, _wallRadius, 0)));      // Front
		objList.push_back(std::make_shared<AABB>(vec3(_wallDist,0,0), 5, vec3(0, -_wallRadius, -_wallRadius), vec3(0, _wallRadius, _wallRadius)));	    // Right
		objList.push_back(std::make_shared<AABB>(vec3(-_wallDist,0,0), 0, vec3(0, -_wallRadius, -_wallRadius),vec3(0, _wallRadius, _wallRadius)));	    // Left
		objList.push_back(std::make_shared<AABB>(vec3(0,_wallDist,0), 2, vec3(-_wallRadius, 0, -_wallRadius), vec3(_wallRadius, 0, _wallRadius)));      // Top
		objList.push_back(std::make_shared<AABB>(vec3(0,-_wallDist,0), 3, vec3(-_wallRadius, 0, -_wallRadius),vec3(_wallRadius, 0, _wallRadius)));      // Bottom
		objList.push_back(std::make_shared<AABB>(vec3(0,0,-_wallDist), 4, vec3(-_wallRadius, -_wallRadius, 0),vec3(_wallRadius, _wallRadius, 0)));      // Back
    }
    void ResetScene() {
        objList = std::vector<std::shared_ptr<Obj>>();
        matList = std::vector<std::shared_ptr<Mat>>();
        importantList = std::vector<int>();
    }
    void AddShape(int s) {
        AddMat(0);
        switch(s) {
            case 0:
                objList.push_back(std::make_shared<Sphere>(vec3(), matList.size()-1, 1));
                break;
            case 1:
                objList.push_back(std::make_shared<AABB>(vec3(), matList.size()-1, vec3(-1,-1,-1), vec3(1,1,1)));
                break;
        }
    }   
    void RemoveShape(int s) {
        if (s >= 0 && s < objList.size()) {

            std::swap(objList.at(s), objList.back());
            objList.pop_back();

            // Need to remake important list
            importantList = std::vector<int>();
            for (int ind = 0; ind < objList.size(); ind++) 
                if (objList[ind]->inImportantList)
                    AddToImpList(ind);
        }  
    }

    void AddToImpList(int index) {
        importantList.push_back(index);
        objList[index]->inImportantList = true;
    }
    void RemoveFromImpList(int index) {
        for (int objInd = 0; objInd < importantList.size(); objInd++)
            if (importantList[objInd] == index) {
                objList[index]->inImportantList = false;
                std::swap(importantList.at(objInd), importantList.back());
                importantList.pop_back();
                return;
            }
    }

    void AddMat(int type) {
        matList.push_back(std::make_shared<Mat>(type, vec3(1,1,1), 0, 1));
    }


    // o type
    // d posx posy posz (radius mat)/(minx miny minz maxx maxy max mat)
    // m albx alby albz blur RI type
    // i impInd

    bool LoadScene(std::string name) {
        std::string path = std::string("../Scenes/").append(name).append(".scene");
        std::ifstream in(path, std::ios::in);
        if (!in) {
            std::cout << "Cannot open scene file!" << std::endl;
            return false;
        }

        // Reset Scene
        ResetScene();

        std::string line;
        std::shared_ptr<Obj> currentObj;
        int currentType;
        while (std::getline(in, line)) {
            // Check for new shapes
            if (line.substr(0,2)=="o ") {
                const char* chh=line.c_str();
                sscanf (chh, "o %i", &currentType);                
            }
            // Check for imp shapes
            else if (line.substr(0,2)=="i ") {
                const char* chh=line.c_str();
                int ind;
                sscanf (chh, "i %i", &ind );
                AddToImpList(ind);
            }
            // Check for materials
            else if (line.substr(0,2)=="m "){
                const char* chh=line.c_str();
                float albx, alby, albz, blur, RI;
                int type;
                sscanf (chh, "m %f/%f/%f %f %f %i", &albx, &alby, &albz, &blur, &RI, &type );
                AddMat(type);
                matList.at(matList.size()-1)->alb  = vec3(albx, alby, albz);
                matList.at(matList.size()-1)->blur = blur;
                matList.at(matList.size()-1)->RI   = RI;
            }
            //check for details
            else if(line.substr(0,2)=="d "){
                const char* chh=line.c_str();
                if (currentType == 0) {
                    float px, py, pz, r;
                    int matInd;

                    sscanf (chh, "d %f/%f/%f %f %i", &px, &py, &pz, &r, &matInd);

                    auto sphere = std::make_shared<Sphere>();
                    sphere->pos = vec3(px, py, pz);
                    sphere->r = r;
                    sphere->mat_ind = matInd;
                    objList.push_back(sphere);
                }
                else if (currentType == 1) {
                    float px, py, pz, minx, miny, minz, maxx, maxy, maxz, rotx, roty, rotz;
                    int matInd;

                    sscanf (chh, "d %f/%f/%f %f/%f/%f %f/%f/%f %f/%f/%f %i", &px, &py, &pz, &minx, &miny, &minz, &maxx, &maxy, &maxz, &rotx, &roty, &rotz, &matInd);

                    auto aabb = std::make_shared<AABB>();
                    aabb->pos = vec3(px, py, pz);
                    aabb->min = vec3(minx, miny, minz);
                    aabb->max = vec3(maxx, maxy, maxz);
                    aabb->rot = vec3(rotx, roty, rotz);
                    aabb->UpdateRot();
                    aabb->mat_ind = matInd;
                    objList.push_back(aabb);
                }
                else {
                    printf("Error parsing .scene file, unkown shape type !");
                    return false;
                }
            }
        }  
        in.close();
        return true;
    }

    bool SaveScene(std::string name) {
        std::stringstream outString;

        // Scene details
        for (auto mat : matList) {
            outString << "m " << mat->alb.x << "/" << mat->alb.y << "/" << mat->alb.z << " " << 
                                 mat->blur<< " " << 
                                 mat->RI << " " << 
                                 mat->matType << "\n"; 
        }
        for (auto obj : objList) {
            int type = obj->type;
            outString << "o " << type << "\n";
            if (type == 0) {
                auto sphere = std::dynamic_pointer_cast<Sphere>(obj);
                outString << "d " << sphere->pos.x << "/" << sphere->pos.y << "/" << sphere->pos.z << " " << 
                                     sphere->r << " " <<
                                     sphere->mat_ind << "\n"; 
            }
            else if (type == 1) {
               auto aabb = std::dynamic_pointer_cast<AABB>(obj);
               outString << "d " << aabb->pos.x << "/" << aabb->pos.y << "/" << aabb->pos.z << " " << 
                                    aabb->min.x << "/" << aabb->min.y << "/" << aabb->min.z << " " << 
                                    aabb->max.x << "/" << aabb->max.y << "/" << aabb->max.z << " " << 
                                    aabb->rot.x << "/" << aabb->rot.y << "/" << aabb->rot.z << " " << 
                                    aabb->mat_ind << "\n";  
            }
        }
        for (auto i : importantList) {
            outString << "i " << i << "\n";
        }

        std::ofstream myfile;
        std::stringstream fileName;
        fileName << "../Scenes/" << name << ".scene";
        myfile.open(fileName.str());
        myfile << outString.str();
        myfile.close();

        return true;
    }

    std::vector<std::shared_ptr<Obj>> objList; 
    std::vector<std::shared_ptr<Mat>> matList; 
    std::vector<int> importantList;

};