#pragma once

#include "GLOBALS.h"
#include "Obj.h"
#include <vector>
#include <memory>
#include <ios>
#include <fstream>
#include <sstream>

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

		objList.push_back(std::make_shared<AABB>(vec3(0,0,_wallDist), 1, vec3(-_wallRadius, -_wallRadius, 0), vec3(_wallRadius, _wallRadius, 0)) );      // Front
		objList.push_back(std::make_shared<AABB>(vec3(_wallDist,0,0), 5, vec3(0, -_wallRadius, -_wallRadius), vec3(0, _wallRadius, _wallRadius)) );	   // Right
		objList.push_back(std::make_shared<AABB>(vec3(-_wallDist,0,0), 0, vec3(0, -_wallRadius, -_wallRadius),vec3(0, _wallRadius, _wallRadius)));	   // Left
		objList.push_back(std::make_shared<AABB>(vec3(0,_wallDist,0), 2, vec3(-_wallRadius, 0, -_wallRadius), vec3(_wallRadius, 0, _wallRadius)) );      // Top
		objList.push_back(std::make_shared<AABB>(vec3(0,-_wallDist,0), 3, vec3(-_wallRadius, 0, -_wallRadius),vec3(_wallRadius, 0, _wallRadius)));      // Bottom
		objList.push_back(std::make_shared<AABB>(vec3(0,0,-_wallDist), 4, vec3(-_wallRadius, -_wallRadius, 0),vec3(_wallRadius, _wallRadius, 0)));      // Back

    }
    void ResetScene() {
        objList = std::vector<std::shared_ptr<Obj>>();
        importantList = std::vector<int>();
    }
    void AddShape(int s) {
        matList.push_back(std::make_shared<Mat>(0, vec3(1,1,1), 0, 1));
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

    bool LoadScene(std::string sceneName) {
        sceneName = std::string("../Scenes/") + sceneName + std::string(".obj");
        std::ifstream in(sceneName, std::ios::in);
        if (!in) {
            printf("Cannot open scene file! \n");
            return false;
        }

        // Reset Scene
        

        std::string line;
        std::shared_ptr<Model> currentModel;
        while (std::getline(in, line))
        {
            // Check for new models
            if (line.substr(0,2)=="o " || line.substr(0,2)=="g ") {
                if (currentModel)
                    currentModel->num_vertices = vertexIndices.size() - currentModel->vert_ind;
                auto name = line.substr(2,line.size());
                currentModel = std::make_shared<Model>(vec3(), 0, name);
                currentModel->vert_ind = vertexIndices.size();
                objList.push_back(currentModel);
            }
            //check for vertices
            else if (line.substr(0,2)=="v "){
                std::istringstream v(line.substr(2));
                float x,y,z;
                v>>x;v>>y;v>>z;
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
            }
            //check for faces
            else if(line.substr(0,2)=="f "){
                int aV, bV, cV; // store mesh index
                int aT, bT, cT; // store texture index
                int aN, bN, cN; // store normal index
                const char* chh=line.c_str();
                sscanf (chh, "f %i/%i/%i %i/%i/%i %i/%i/%i",&aV,&aT,&aN,&bV,&bT,&bN,&cV,&cT,&cN);
                //Decrement indices
                aV--; bV--; cV--;
                aT--; bT--; cT--;
                aN--; bN--; cN--;
                //std::cout<<a<<b<<c<<A<<B<<C;
                vertexIndices.push_back(aV);
                vertexIndices.push_back(bV);
                vertexIndices.push_back(cV);
            }
        }  
    }

    std::vector<std::shared_ptr<Obj>> objList; 
    std::vector<std::shared_ptr<Mat>> matList; 
    std::vector<int> importantList;

};