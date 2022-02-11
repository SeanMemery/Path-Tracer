#pragma once

#include "vec3.h"
#include "Mat.h"
#include "ext/imgui.h"
#include <vector>

// Sphere: 0, AABB: 1, Model: 2
//
// Shape GPU Structure: obj_type, mat_ind, attr_ind
//
// Every shape has an index to its attributes:
// 	- Sphere (4 attrs)  : pos.x, pos.y, pos.z, radius
// 	- AABB   (9 attrs)  : pos.x, pos.y, pos.z, min.x, min.y, min.z, max.x, max.y, max.z
//	- Model  (6 attrs)  : pos.x, pos.y, pos.z, scale, vert_ind, num_vertices
//

class Obj {
public:

    Obj() : inImportantList(false) {};

    // ImGui Edit Screen
    virtual bool ImGuiEdit() = 0;
	
    vec3 pos;
	int type, mat_ind;
	bool inImportantList;
};

class Sphere : public Obj {
public:

    Sphere(vec3 _pos, int _material, float _r)
    : r(_r) {
        pos = _pos;
        mat_ind = _material;
		type = 0;
    }

    bool ImGuiEdit() {
		ImGui::Text("--------Sphere-------");
		bool ref = false;
		float sPos[3]{ pos.x,pos.y,pos.z };
		if (ImGui::InputFloat3("Mid Position", sPos)) {
		    pos = vec3(sPos[0], sPos[1], sPos[2]);
			ref = true;
		}
		ref |= ImGui::InputFloat("Radius", &r);
		return ref;
	}

    float r;

};

class AABB : public Obj {
public:

    AABB(vec3 _pos, int _material, vec3 _min, vec3 _max) 
    : min(_min), max(_max) {
        pos = _pos;
        mat_ind = _material; 
		type = 1;
    }

    bool ImGuiEdit() {
		ImGui::Text("--------AABB-------");
		bool ref = false;
		float sPos[3]{ pos.x,pos.y,pos.z };
		if (ImGui::InputFloat3("Mid Position", sPos)) {
		    pos = vec3(sPos[0], sPos[1], sPos[2]);
			ref = true;
		}
		float minA[3]{min.x,min.y,min.z };
		if (ImGui::InputFloat3("Relative Min", minA)) {
		    min = vec3(minA[0], minA[1], minA[2]);
			ref = true;
		}
		float maxA[3]{max.x,max.y,max.z };
		if (ImGui::InputFloat3("Relative Max", maxA)) {
		    max = vec3(maxA[0], maxA[1], maxA[2]);
			ref = true;
		}
		return ref;
	}

    vec3 min, max;

};

class Model : public Obj {
public:

    Model(vec3 _pos, int _material, std::string _name) 
    : name(_name) {
        pos = _pos;
        mat_ind = _material; 
		type = 2;
		scale = 1;

		vert_ind = 0;
		num_vertices = 0;
    }

	bool ImGuiEdit() {
		ImGui::Text("--------Model-------");
		bool ref = false;
		float sPos[3]{ pos.x,pos.y,pos.z };
		if (ImGui::InputFloat3("Mid Position", sPos)) {
		    pos = vec3(sPos[0], sPos[1], sPos[2]);
			ref = true;
		}
		ref |= ImGui::InputFloat("Scale", &scale, 0, 100.0f);
		return ref;
	}

	float scale;
	std::string name;
	int vert_ind, num_vertices;

};