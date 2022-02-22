#pragma once

#include "vec3.h"
#include "Mat.h"
#include "ext/imgui.h"
#include <vector>

// Sphere: 0, AABB: 1
//
// Shape GPU Structure: obj_type, mat_ind, attr_ind
//
// Every shape has an index to its attributes:
// 	- Sphere (4 attrs)   : pos.x, pos.y, pos.z, radius
// 	- AABB   (18 attrs)  : pos.x, pos.y, pos.z, min.x, min.y, min.z, max.x, max.y, max.z
// 						   q.r, q.x, q.y, q.z
//

struct Quaternion {

	Quaternion() 
	: w(1), x(0), y(0), z(0) {}

	Quaternion(float _w, float _x, float _y, float _z)
	: w(_w), x(_x), y(_y), z(_z) {}

	static Quaternion Mult(Quaternion q1, Quaternion q2) {
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

        return Quaternion(Q1, Q2, Q3, Q4);
    };

	float operator[](int ind) {
		switch(ind) {
			case 0:
				return w;
			case 1:
				return x;
			case 2:
				return y;
			case 3:
				return z;
		}
		return 0;
	}
	
	float w, x, y, z;

};

class Obj {
public:

    Obj() : inImportantList(false) {};

	// Rotation Matrix
/* 	
	void Rotate(vec3 t[3]) {
		vec3 mat[3];
		mat[0] = vec3(t[0].dot(vec3(rot[0].x, rot[1].x, rot[2].x)), t[0].dot(vec3(rot[0].y, rot[1].y, rot[2].y)), t[0].dot(vec3(rot[0].z, rot[1].z, rot[2].z)));
		mat[1] = vec3(t[1].dot(vec3(rot[0].x, rot[1].x, rot[2].x)), t[1].dot(vec3(rot[0].y, rot[1].y, rot[2].y)), t[1].dot(vec3(rot[0].z, rot[1].z, rot[2].z)));
		mat[2] = vec3(t[2].dot(vec3(rot[0].x, rot[1].x, rot[2].x)), t[2].dot(vec3(rot[0].y, rot[1].y, rot[2].y)), t[2].dot(vec3(rot[0].z, rot[1].z, rot[2].z)));

		rot[0] = mat[0];
		rot[1] = mat[1];
		rot[2] = mat[2];
	}
	void rotateX(float angle) {
		vec3 rotation[3];
		rotation[0] = vec3(1,0,0);
		rotation[1] = vec3(0,cos(angle),-sin(angle));
		rotation[2] = vec3(0,sin(angle), cos(angle));
		Rotate(rotation);
	}
	void rotateY(float angle) {
		vec3 rotation[3];
		rotation[0] = vec3(cos(angle),0, sin(angle));
		rotation[1] = vec3(0,1,0);
		rotation[2] = vec3(-sin(angle),0,cos(angle));
		Rotate(rotation);
	}
	void rotateZ(float angle) {
		vec3 rotation[3];
		rotation[0] = vec3(cos(angle),-sin(angle),0);
		rotation[1] = vec3(sin(angle), cos(angle),0);
		rotation[2] = vec3(0,0,1);
		Rotate(rotation);
	}
 */
    
	// ImGui Edit Screen
    virtual bool ImGuiEdit() = 0;
	
    vec3 pos, rot;
	// vec3 rot[3] = {vec3(1,0,0), vec3(0,1,0), vec3(0,0,1)};
	int type, mat_ind;
	bool inImportantList;
};

class Sphere : public Obj {
public:

	Sphere() {
			r = 1;
			pos = vec3();
			mat_ind = 0;
			type = 0;
	}
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
	AABB() {
        pos = vec3();
		min = vec3();
		max = vec3();
        mat_ind = 0; 
		type = 1;
    }
    AABB(vec3 _pos, int _material, vec3 _min, vec3 _max) 
    : min(_min), max(_max) {
        pos = _pos;
        mat_ind = _material; 
		type = 1;
    }

	void UpdateRot() {
		Quaternion qx = Quaternion(cosf(rot.x/2.0f), sinf(rot.x/2.0f), 0, 0);
		Quaternion qz = Quaternion(cosf(rot.z/2.0f), 0, 0, sinf(rot.z/2.0f));
		Quaternion qy = Quaternion(cosf(rot.y/2.0f), 0, sinf(rot.y/2.0f), 0);
		q = Quaternion::Mult(qx, qy);
		q = Quaternion::Mult(q, qz);
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
		float tempRot[3]{rot.x,rot.y,rot.z};
		if (ImGui::InputFloat3("Apply Rotation", tempRot)) {
		    rot.x = tempRot[0];
			rot.y = tempRot[1];
			rot.z = tempRot[2];

			UpdateRot();

			ref = true;
		}
		return ref;
	}

    vec3 min, max;
	Quaternion q;

};
