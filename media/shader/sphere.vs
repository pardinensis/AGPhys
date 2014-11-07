#version 400

layout(location = 0)in vec3 v_position;
layout(location = 1)in vec3 v_normal;
layout(location = 2)in vec4 v_color;
layout(location = 3)in vec2 v_st;

uniform mat4 view_matrix;
uniform mat4 proj_matrix;
uniform mat4 model_matrix;
uniform mat3 normal_matrix;

out vec4 C; // red green blue radius

void main()
{
    C = v_color;
    gl_Position = model_matrix * vec4(v_position,1);
}
