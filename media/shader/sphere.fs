#version 400

in VertexData
{
    vec4 color;
    vec3 center;
    vec3 fPos;
} VertexIn;

in float radius;

out vec4 FragColor;

void main( void )
{
    vec3 surface = VertexIn.fPos - VertexIn.center;
    float sl = length(surface);

    if(sl > radius)
        discard;

    surface.z = sqrt(radius*radius - sl*sl);

    vec3 l = vec3(0,0,1);
    vec4 di = VertexIn.color * max(dot(l,normalize(surface)),0);
    FragColor = di;

    gl_FragDepth = gl_FragCoord.z - 0.0005;
}
