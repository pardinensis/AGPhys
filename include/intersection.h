#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "platform.h"

struct Intersection
{
    vec3 position;
    vec3 barycenter;
    vec3 normal;
    unsigned int triangleId;
    unsigned int nearestVertexId;
    REAL t;

    Intersection();

    Intersection(const Intersection& other);

    ~Intersection();

    void operator=(const Intersection& other);
};

#endif // INTERSECTION_H
