#version 400

in vec3 N;
in vec4 C;
in vec2 ST;
in vec3 V_e;
in vec3 L_e;
in float glow;

layout(location=0) out vec4 frag_color;

void main( void )
{
    // directional light
    vec4 I_d = vec4(0);
    vec4 I_s = vec4(0);
    vec3 L = normalize(vec3(0,1,1));
    float NdotL = dot(N,L);
    float lambert = max(NdotL, 0.0);

    if(lambert > 0)
    {
        I_d = C * vec4(0.5,0.5,0.5,1.0) * lambert;
                
        vec3 E = normalize(-V_e);
        vec3 R = normalize(-reflect(L, N));
        I_s = C * vec4(0.2,0.2,0.2,1.0) * pow(max(dot(R, E), 0.1), 32.0);
        I_s = clamp(I_s, 0.0, 1.0);
    }

    frag_color = I_d + I_s;

    // glow
    vec3 L2 = normalize(L_e-V_e);
    NdotL = dot(N,L2);
    lambert = max(NdotL, 0.0);
    if(lambert > 0.0)
    {
        frag_color += vec4(1,0.8,0.8,1)*lambert*glow;
    }


    frag_color = max(frag_color, vec4(0.1,0.1,0.1,1.0) * C);
}
