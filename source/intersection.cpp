#include "intersection.h"

Intersection::Intersection()
    : t(std::numeric_limits<float>::max()),
      triangleId(std::numeric_limits<unsigned int>::max()),
      nearestVertexId(std::numeric_limits<unsigned int>::max())
{
    position = barycenter = normal = vec3(0,0,0);
}

Intersection::Intersection(const Intersection& other)
    : position(other.position),
      barycenter(other.barycenter),
      normal(other.normal),
      t(other.t),
      triangleId(other.triangleId),
      nearestVertexId(other.nearestVertexId)
{

}


Intersection::~Intersection()
{

}

void Intersection::operator=(const Intersection& other)
{
    position = other.position;
    barycenter = other.barycenter;
    normal = other.normal;
    t = other.t;
    nearestVertexId = other.nearestVertexId;
    triangleId = other.triangleId;
}
