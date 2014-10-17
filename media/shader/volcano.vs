#version 400

layout(location = 0)in vec3 v_position;
layout(location = 1)in vec3 v_normal;
layout(location = 2)in vec4 v_color;
layout(location = 3)in vec2 v_st;

uniform mat4 view_matrix;
uniform mat4 proj_matrix;
uniform mat4 model_matrix;
uniform mat3 normal_matrix;
uniform int bGlow;

out vec3 N;
out vec4 C;
out vec2 ST;
out vec3 V_e;
out vec3 L_e;
out float glow;

void main()
{
	vec4 posE = view_matrix * model_matrix * vec4(v_position,1);

    // output vertex data
    N = normalize(normal_matrix * v_normal);
    C = vec4(1,1,1,1);
    ST = vec2(v_st.x, 1-v_st.y);
	V_e = vec3(posE);
	
	// lightsource in volcano
	L_e = vec3(view_matrix * model_matrix * vec4(0,0.35,0,1));
	
	if(v_position[1] >= 0.2 && bGlow > 0)
		glow = 1.0;
	else
		glow = 0.0;

    // and projected
    gl_Position = proj_matrix * posE;
}
