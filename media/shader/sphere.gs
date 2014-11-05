#version 400

layout (points) in;
layout (triangle_strip, max_vertices = 3) out;

in vec4 C[];
in float R[];

out VertexData
{
    vec4 color;
    vec3 center;
    vec3 fPos;
} VertexOut;

out float radius;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 proj_matrix;
uniform mat3 normal_matrix;

void main()
{
    mat4 Rt = transpose(view_matrix);
    vec3 eye_position = -(Rt*view_matrix)[3].xyz;

    vec3 pos = gl_in[0].gl_Position.xyz;
    vec3 view = normalize(pos - eye_position);
    vec3 up = vec3(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]);
    vec3 right = cross(view, up);

    radius = R[0];
    vec4 color = C[0];

    float a_05 = 1.73205 * radius;

    vec3 wp = pos-radius*up - a_05*right;
    gl_Position = proj_matrix * view_matrix * vec4(wp, 1);
        VertexOut.color = color;
    VertexOut.center = (view_matrix * vec4(pos,1)).xyz;
    VertexOut.fPos = (view_matrix * vec4(wp, 1)).xyz;
    EmitVertex();

    wp = pos - radius*up + a_05*right;
    gl_Position = proj_matrix * view_matrix * vec4(wp, 1);
        VertexOut.color = color;
        VertexOut.center = (view_matrix * vec4(pos,1)).xyz;
    VertexOut.fPos = (view_matrix * vec4(wp, 1)).xyz;
    EmitVertex();

    wp = pos + 2.0*radius*up;
    gl_Position = proj_matrix * view_matrix * vec4(wp, 1);
        VertexOut.color = color;
    VertexOut.center = (view_matrix * vec4(pos,1)).xyz;
    VertexOut.fPos = (view_matrix * vec4(wp, 1)).xyz;
    EmitVertex();

    EndPrimitive();
}
