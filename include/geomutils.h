#ifndef GEOMUTILS_H
#define GEOMUTILS_H

#include "platform.h"
#include "matrixutils.h"
#include "ray.h"
#include "intersection.h"

/*! computeTriangleMeshNormals()
 *
 *  \brief  calculates a smooth per vertex normal set (normalized)
 */
static void computeTriangleMeshNormals(const std::vector<vec3>& v, const std::vector<ivec3>& t, std::vector<vec3>& out)
{
    out.resize(v.size());
	for(unsigned int i = 0; i < out.size(); ++i)
	{
		out[i] = vec3(0,0,0);
	}

    for(unsigned int i = 0; i < t.size(); ++i)
    {
        vec3 A = v[t[i][0]];
        vec3 B = v[t[i][1]];
        vec3 C = v[t[i][2]];
        vec3 fn = (B-A).cross(C-A).normalized();
        out[t[i][0]] += fn;
        out[t[i][1]] += fn;
        out[t[i][2]] += fn;
    }

    for(unsigned int j = 0; j < out.size(); ++j)
        out[j].normalize();
}

#endif // GEOMUTILS_H
