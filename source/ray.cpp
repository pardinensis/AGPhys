#include "ray.h"
#include "intersection.h"
#include "matrixutils.h"

Ray::Ray()
    :   minT(std::numeric_limits<REAL>::epsilon()),
        maxT(std::numeric_limits<REAL>::max())
{
    origin.setZero();
    direction.setZero();
}

Ray::Ray(const vec3& o, const vec3& d)
    :   origin(o),
        direction(d),
        minT(std::numeric_limits<REAL>::epsilon()),
        maxT(std::numeric_limits<REAL>::max())
{

}

Ray::Ray(const vec3& camera_position, const vec3& camera_direction, const vec3& camera_up,
    const REAL& camera_fov_x, const REAL& camera_nearplane,
    const ivec2& screen_pos, const ivec2& screen_dimensions)
    :   minT(std::numeric_limits<REAL>::epsilon()),
        maxT(std::numeric_limits<REAL>::max())

{
    // TODO: consider the ortho case

    REAL x = (REAL) screen_pos[0];
    REAL y = (REAL) screen_pos[1];
    vec3 h = camera_direction.cross(camera_up).normalized();
    vec3 v = h.cross(camera_direction).normalized();
    REAL radians = camera_fov_x * REAL(M_PI) * 0.005555f;
    REAL vLen = tanf(radians*0.5f)*camera_nearplane;
    REAL hLen = vLen * ((REAL)screen_dimensions[0]/(REAL)screen_dimensions[1]);
    v *= vLen;
    h *= hLen;
    x -= screen_dimensions[0]*.5f;
    y -= screen_dimensions[1]*.5f;
    x /= (screen_dimensions[0]*.5f);
    y /= (screen_dimensions[1]*.5f);
    origin = camera_position;
    direction = camera_direction*camera_nearplane + h*x - v*y;
    direction.normalize();
}

Ray::Ray(const mat4& viewM, const mat4& projM, const ivec2& screenPos, const ivec2& screenDim)
	:	minT(std::numeric_limits<REAL>::epsilon()),
		maxT(std::numeric_limits<REAL>::max())
{
	// calculate clip space
	
	vec2 clipCoord;
	clipCoord.x() = REAL(2*screenPos.x())/screenDim.x() - REAL(1);
	clipCoord.y() = REAL(1) - REAL(2*screenPos.y())/screenDim.y();
	extractEyePosFromViewMatrix(viewM, origin);
	mat4 unviewM = (projM*viewM).inverse();
	vec4 nearPoint = unviewM * vec4(clipCoord.x(), clipCoord.y(), 0, 1);
	vec3 nearPoint3 = vec3(nearPoint.x()/nearPoint.w(), nearPoint.y()/nearPoint.w(), nearPoint.z()/nearPoint.w());
	direction = (nearPoint3-origin).normalized();
}

void Ray::operator=(const Ray& other)
{
    origin = other.origin;
    direction = other.direction;
    minT = other.minT;
    maxT = other.maxT;
}

bool Ray::intersectsSphere(const vec3& center, REAL radius, Intersection& outIsct)
{
	//FROM: http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-7-intersecting-simple-shapes/ray-sphere-intersection/

	REAL radius2 = radius*radius;

	vec3 L = center - origin;
	REAL tca = L.dot(direction);
	if(tca < 0)
		return false;

	REAL d2 = L.dot(L) - tca*tca;
	if(d2 > radius2)
		return false;
	REAL thc = std::sqrt(radius2 - d2);
	REAL t0 = tca - thc;
	REAL t1 = tca + thc;

	if(t0 > maxT)
		return false;
	
	// fill intersection structure
	outIsct.t = t0;
	outIsct.position = origin + t0*direction;
	outIsct.nearestVertexId = 0;
	outIsct.barycenter = vec3(0,0,0);
	outIsct.triangleId = 0;
	outIsct.normal = vec3(outIsct.position - center).normalized();

	return true;
}