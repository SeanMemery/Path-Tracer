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

		auto l_min = vec3(-2.5, -0.5f, -2.5);
		auto l_max = vec3(2.5, -0.5f, 2.5);
		objList.push_back(std::make_shared<AABB>(lightMat1, l_min, l_max));

        importantList.push_back(0);

		objList.push_back(std::make_shared<AABB>(wallMat2, vec3(-_wallRadius, -_wallRadius, _wallDist), vec3(_wallRadius, _wallRadius, _wallDist)) );      // Front
		objList.push_back(std::make_shared<AABB>(wallMat1, vec3(_wallDist, -_wallRadius, -_wallRadius), vec3(_wallDist, _wallRadius, _wallRadius)) );	   // Right
		objList.push_back(std::make_shared<AABB>(wallMat6, vec3(-_wallDist, -_wallRadius, -_wallRadius),vec3(-_wallDist, _wallRadius, _wallRadius)));	   // Left
		objList.push_back(std::make_shared<AABB>(wallMat3, vec3(-_wallRadius, _wallDist, -_wallRadius), vec3(_wallRadius, _wallDist, _wallRadius)) );      // Top
		objList.push_back(std::make_shared<AABB>(wallMat4, vec3(-_wallRadius, -_wallDist, -_wallRadius),vec3(_wallRadius, -_wallDist, _wallRadius)));      // Bottom
		objList.push_back(std::make_shared<AABB>(wallMat5, vec3(-_wallRadius, -_wallRadius, -_wallDist),vec3(_wallRadius, _wallRadius, -_wallDist)));      // Back


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
                objList.push_back(std::make_shared<AABB>(Mat(), vec3(-1,-1,-1), vec3(1,1,1)));
                break;
        }
    }   
    
    std::vector<std::shared_ptr<Obj>> objList; 
    std::vector<int> importantList;

};