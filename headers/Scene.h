#pragma once

#include "Obj.h"
#include <vector>
#include <memory>

class Scene {
public:
    void InitScene() {

        ResetScene();

       	auto wallMat1 = Mat(0, vec3(0.65, .05, .05)  , 0, 0); // Red Wall
		auto wallMat2 = Mat(0, vec3(0.73, 0.73, 0.73), 0, 0);
		auto wallMat3 = Mat(0, vec3(0.73, 0.73, 0.73), 0, 0);
		auto wallMat4 = Mat(0, vec3(0.73, 0.73, 0.73), 0, 0);
		auto wallMat5 = Mat(0, vec3(0.73, 0.73, 0.73), 0, 0);
		auto wallMat6 = Mat(0, vec3(.12, 0.45, .15)  , 0, 0); // Green Wall

		auto lightMat1 = Mat(1, vec3(1,1,1), 0, 0);

		double _wallDist = 10;
		double _wallRadius = 10;

		//objList.push_back(std::make_shared<Sphere>(vec3(0, _wallDist, 0), lightMat1, 1));
        objList.push_back(std::make_shared<AABB>(vec3(0, _wallDist-1.0f, 0), lightMat1, vec3(-2.5f, 0, -2.5f), vec3(2.5f, 1.0f, 2.5f)));
        AddToImpList(0);

		objList.push_back(std::make_shared<AABB>(vec3(0,0,_wallDist), wallMat2, vec3(-_wallRadius, -_wallRadius, 0), vec3(_wallRadius, _wallRadius, 0)) );      // Front
		objList.push_back(std::make_shared<AABB>(vec3(_wallDist,0,0), wallMat6, vec3(0, -_wallRadius, -_wallRadius), vec3(0, _wallRadius, _wallRadius)) );	   // Right
		objList.push_back(std::make_shared<AABB>(vec3(-_wallDist,0,0), wallMat1, vec3(0, -_wallRadius, -_wallRadius),vec3(0, _wallRadius, _wallRadius)));	   // Left
		objList.push_back(std::make_shared<AABB>(vec3(0,_wallDist,0), wallMat3, vec3(-_wallRadius, 0, -_wallRadius), vec3(_wallRadius, 0, _wallRadius)) );      // Top
		objList.push_back(std::make_shared<AABB>(vec3(0,-_wallDist,0), wallMat4, vec3(-_wallRadius, 0, -_wallRadius),vec3(_wallRadius, 0, _wallRadius)));      // Bottom
		objList.push_back(std::make_shared<AABB>(vec3(0,0,-_wallDist), wallMat5, vec3(-_wallRadius, -_wallRadius, 0),vec3(_wallRadius, _wallRadius, 0)));      // Back


    }
    void ResetScene() {
        objList = std::vector<std::shared_ptr<Obj>>();
        importantList = std::vector<int>();
    }
    void AddShape(int s) {
        switch(s) {
            case 0:
                objList.push_back(std::make_shared<Sphere>(vec3(), Mat(), 1));
                break;
            case 1:
                objList.push_back(std::make_shared<AABB>(vec3(), Mat(), vec3(-1,-1,-1), vec3(1,1,1)));
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
    std::vector<std::shared_ptr<Obj>> objList; 
    std::vector<int> importantList;

};