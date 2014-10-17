#ifndef __RAY_H__
#define __RAY_H__

#include "platform.h"

struct Intersection;

/*! \brief Representation of a ray
 *
 * origin, direction and min, max thresholds.
 * used for intersection tests.
 */
struct Ray
{
    //! ray origin in 3d
    vec3 origin;

    //! ray direction in 3d
    vec3 direction;

    //! minimum offset of the ray
    REAL minT;

    //! maximum range of the ray
    REAL maxT;

    //! default constructor
    Ray();

    //! constructor with initialization for origin (o) and direction (d)
    Ray(const vec3& o, const vec3& d);

    //! creates a pick ray for given camera parameters
    Ray(const vec3& camera_position, const vec3& camera_direction, const vec3& camera_up,
        const REAL& camera_fov_x, const REAL& camera_nearplane,
        const ivec2& screen_pos, const ivec2& screen_dimensions);

	//! creates a pick ray for camera matrices
	Ray(const mat4& viewM, const mat4& projM, const ivec2& screenPos, const ivec2& screenDim);

    //! assignment operator
    void operator= (const Ray& other);

	//! intersection test with a sphere
	bool intersectsSphere(const vec3& center, REAL radius, Intersection& outIsct);
};

static std::ostream& operator<< (std::ostream& out, const Ray& v)
{
    out << "(" << v.origin.x() << ", " << v.origin.y() << ", " << v.origin.z() << ") -> ";
    out << "(" << v.direction.x() << ", " << v.direction.y() << ", " << v.direction.z() << ") [";
    out << v.minT << ", " << v.maxT << "]";
    return out;
}

#endif
