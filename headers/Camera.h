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
				up -= right * 0.25f*speed*ImGui::GetIO().DeltaTime;
				up.normalize();
				right = up.cross(forward).normalize();
				return;
			case CamAxis::UP:
				forward += right * 0.5f*speed*ImGui::GetIO().DeltaTime;
				forward.normalize();
				right = up.cross(forward).normalize();
				return;
			case CamAxis::RIGHT:
				forward += up * 0.5f*speed*ImGui::GetIO().DeltaTime;
				forward.normalize();
				up = forward.cross(right).normalize();
				return;
			case CamAxis::REVFORWARD:
				up += right * 0.25f*speed*ImGui::GetIO().DeltaTime;
				up.normalize();
				right = up.cross(forward).normalize();
				return;
			case CamAxis::REVUP:
				forward -= right * 0.5f*speed*ImGui::GetIO().DeltaTime;
				forward.normalize();
				right = up.cross(forward).normalize();
				return;
			case CamAxis::REVRIGHT:
				forward -= up * 0.5f*speed*ImGui::GetIO().DeltaTime;
				forward.normalize();
				up = forward.cross(right).normalize();
				return;
		}
    }

    float speed = 5.0f;
    vec3 pos, forward, up, right;
	float focalLen, vfov, hfov;

};