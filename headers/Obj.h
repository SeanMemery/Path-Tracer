#pragma once

#include "vec3.h"
#include "Mat.h"
#include "ext/imgui.h"
#include <vector>

// Sphere: 0, AABB: 1

class Obj {
public:

    Obj() {};

    // ImGui Edit Screen
    virtual bool ImGuiEdit() = 0;

	// Rendering Information
	virtual std::vector<float> GetData()=0;

    vec3 pos;
    Mat material;
	int type;
	bool inImportantList;
};

class Sphere : public Obj {
public:

    Sphere(vec3 _pos, Mat _material, float _r)
    : r(_r) {
        pos = _pos;
        material = _material;
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

	std::vector<float> GetData() {
		auto vec = std::vector<float>();

		vec.push_back(pos.x);
		vec.push_back(pos.y);
		vec.push_back(pos.z);

		vec.push_back(material.alb.x);
		vec.push_back(material.alb.y);
		vec.push_back(material.alb.z);

		vec.push_back(material.matType);
		vec.push_back(1);

		vec.push_back(r);

		// BUFFERS
		vec.push_back(0.0f);
		vec.push_back(0.0f);
		vec.push_back(0.0f);
		vec.push_back(0.0f);
		vec.push_back(0.0f);
		// BUFFERS

		vec.push_back(material.blur);
		vec.push_back(material.RI);


		return vec;
	}
	

    float r;

};

class AABB : public Obj {
public:

    AABB(vec3 _pos, Mat _material, vec3 _min, vec3 _max) 
    : min(_min), max(_max) {
        pos = _pos;
        material = _material; 
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

	std::vector<float> GetData() {
		auto vec = std::vector<float>();

		vec.push_back(pos.x);
		vec.push_back(pos.y);
		vec.push_back(pos.z);

		vec.push_back(material.alb.x);
		vec.push_back(material.alb.y);
		vec.push_back(material.alb.z);

		vec.push_back(material.matType);
		vec.push_back(0);

		vec.push_back(pos.x + min.x);
		vec.push_back(pos.y + min.y);
		vec.push_back(pos.z + min.z);

		vec.push_back(pos.x + max.x);
		vec.push_back(pos.y + max.y);
		vec.push_back(pos.z + max.z);

		vec.push_back(material.blur);
		vec.push_back(material.RI);

		return vec;

	}

    vec3 min, max;

};