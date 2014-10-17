#version 400

in vec3 V;
in vec4 C;
in vec2 ST;

layout(location=0) out vec4 frag_color;

void main( void )
{
    frag_color = C;
}
