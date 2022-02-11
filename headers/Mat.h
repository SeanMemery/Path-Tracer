#pragma once

#include "vec3.h"
#include "ext/imgui.h"

// Lambertian: 0, Light: 1, Metal: 2, Dielectric: 3

class Mat {
public:

    Mat() : matType(0), alb(vec3(1,1,1)), blur(0), RI(1) {}

    Mat(int _matType, vec3 _alb, float _blur, float _RI)
     : matType(_matType), alb(_alb), blur(_blur), RI(_RI) {}

    // ImGui Edit Screen
    bool ImGuiEdit() {
        bool ref = false;
        ref |= ImGui::SliderInt("Mat Type", &matType, 0, 3);
        switch(matType) {
            case 0:
                ImGui::Text("Lambertian");
                break;
            case 1:
                ImGui::Text("Light");
                break;
            case 2:
                ImGui::Text("Metal");
                break;
            case 3:
                ImGui::Text("Dielectric");
                break;
        }
		float a[3]{ alb.x,alb.y,alb.z };
		if (ImGui::InputFloat3("Mat Diffuse", a)) {
		    alb = vec3(a[0], a[1], a[2]);
            ref = true;
        }
        ref |= ImGui::InputFloat("Mat Blur", &blur);
        ref |= ImGui::InputFloat("Mat Refractive Index", &RI);
        return ref;
    }

    vec3 alb;
    float blur, RI;
    int matType;

};