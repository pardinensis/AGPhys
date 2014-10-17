#version 400

layout(location = 0)in vec3 v_position;
layout(location = 1)in vec3 v_velocity;
layout(location = 2)in vec4 v_color;
layout(location = 3)in vec2 v_st;

uniform mat4 view_matrix;
uniform mat4 proj_matrix;
uniform mat4 model_matrix;

out vec3 V;
out vec4 C;
out vec2 ST;

void main()
{
    // output vertex data
	V = v_velocity;
	C = vec4(v_color.xyz, 0.05);
	C *= smoothstep(0.0,0.5,v_position.y),
	ST = v_st;
	
    // and projected
    gl_Position = proj_matrix * view_matrix * model_matrix * vec4(v_position,1);
}
