#pragma once

#include "vec3.h"

// Lambertian: 0, Light: 1, Metal: 2, Dielectric: 3

class Mat {
public:

    Mat() : matType(0), alb(vec3(1,1,1)), blur(0), RI(0) {}

    Mat(int _matType, vec3 _alb, float _blur, float _RI)
     : matType(_matType), alb(_alb), blur(_blur), RI(_RI) {}

    // ImGui Edit Screen
    void ImGuiEdit() {
        ImGui::Text("");
		ImGui::Text("--------Mat--------");
        ImGui::SliderInt("Mat Type", &matType, 0, 3);
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
		if (ImGui::InputFloat3("Mat Diffuse", a))
		    alb = vec3(a[0], a[1], a[2]);
        ImGui::InputFloat("Mat Blur", &blur);
        ImGui::InputFloat("Mat Refractive Index", &RI);
		ImGui::Text("-------------------"); 
    }

    vec3 alb;
    float blur, RI;
    int matType;

};