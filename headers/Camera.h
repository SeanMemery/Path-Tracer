#pragma once

#include "GLOBALS.h"
#include "vec3.h"
#include "ext/imgui.h"

enum CamAxis {
    FORWARD, UP, RIGHT,
	REVFORWARD, REVUP, REVRIGHT
};

class Camera {
public:
    void moveDir(vec3 dir) {
        pos += dir*speed*ImGui::GetIO().DeltaTime;
		moving = true;
		refresh = true;
    }
    void rotateAroundAxis(CamAxis axis) {
		moving = true;
		refresh = true;
        switch (axis) {
			case CamAxis::FORWARD:
				//PTMath::QuaternionRotateAroundVector(forward, up, angle);
				//right = up.cross(forward).normalize();
				return;
			case CamAxis::UP:
				forward += right * 0.5f*speed*ImGui::GetIO().DeltaTime;
				forward.normalize();
				//PTMath::QuaternionRotateAroundVector(up, forward, angle);
				right = up.cross(forward).normalize();
				return;
			case CamAxis::RIGHT:
				//PTMath::QuaternionRotateAroundVector(up, right, angle);
				//forward = right.cross(up).normalize();
				return;
			case CamAxis::REVFORWARD:
				//PTMath::QuaternionRotateAroundVector(forward, up, -angle);
				//right = up.cross(forward).normalize();
				return;
			case CamAxis::REVUP:
				forward -= right * 0.5f*speed*ImGui::GetIO().DeltaTime;
				forward.normalize();
				//PTMath::QuaternionRotateAroundVector(up, forward, -angle);
				right = up.cross(forward).normalize();
				return;
			case CamAxis::REVRIGHT:
				//PTMath::QuaternionRotateAroundVector(up, right, -angle);
				//forward = right.cross(up).normalize();
				return;
		}
    }

    float speed = 5.0f;
    vec3 pos, forward, up, right;
	float focalLen, vfov, hfov;

};