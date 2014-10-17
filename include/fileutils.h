#ifndef FILEUTILS_H
#define FILEUTILS_H

#include "platform.h"
#include "geomutils.h"

/*
 *  file handle helpers
 *
 */

/*
 *  import for meshes
 *
 */
static bool importTriangleMeshFromOFF(const std::string& filename, std::vector<vec3>& vertices, std::vector<ivec3>& triangles)
{
	vertices.clear();
    triangles.clear();

    std::ifstream inF(filename);

    if(!inF.good())
    {
        PRINTERROR("importTriangleMeshFromOFF error: file " << filename.c_str() << " not found");
        return false;
    }

    char head[5];
    inF >> head;
    int numV = 0, numT = 0, numE = 0;

    inF >> numV >> numT >> numE;
    for(int i = 0; i < numV; ++i)
    {
        vec3 mI;
        inF >> mI[0] >> mI[1] >> mI[2];
        vertices.push_back(mI);
    }

    for(int i2 = 0; i2 < numT; ++i2)
    {
        int numF;
        ivec3 tI;
        inF >> numF >> tI[0] >> tI[1] >> tI[2];
        triangles.push_back(tI);
    }

    return true;
}

// without triangles version
static bool importTriangleMeshFromOFF(const std::string& filename, std::vector<vec3>& vertices)
{
    vertices.clear();

    std::ifstream inF(filename);

    if(!inF.good())
    {
        PRINTERROR("importTriangleMeshFromOFF error: file " << filename.c_str() << " not found");
        return false;
    }

    char head[5];
    inF >> head;
    int numV = 0, numT = 0, numE = 0;

    inF >> numV >> numT >> numE;
    for(int i = 0; i < numV; ++i)
    {
        vec3 mI;
        inF >> mI[0] >> mI[1] >> mI[2];
        vertices.push_back(mI);
    }

    return true;
}

// with normal version
static bool importTriangleMeshFromOFF(const std::string& filename, std::vector<vec3>& vertices, std::vector<vec3>& normals, std::vector<ivec3>& triangles)
{
    bool res = importTriangleMeshFromOFF(filename, vertices, triangles);
    if(res) computeTriangleMeshNormals(vertices, triangles, normals);
    return res;
}

#endif // FILEUTILS_H
