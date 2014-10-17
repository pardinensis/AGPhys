#ifndef __PLATFORM_H__
#define __PLATFORM_H__

//! stl related stuff
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <limits>
#include <list>
#include <set>
#include <stack>
#include <string>
#include <sstream>

#ifdef WIN32
 #include <windows.h>
#endif

// sophisticated vector/matrix/numeric
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/StdVector>

#undef max
#undef min

//
#define EPSILON 0.0000001
#define MERGE_EPSILON 0.0001

//
#define PRINTERROR(x) { std::cerr << "ERROR: " << x << " (line: " << __LINE__ << " of file: " << __FILE__ << ")" << std::endl; }
#define LOG(x) { std::cerr << "LOG: " << x << " (line: " << __LINE__ << " of file: " << __FILE__ << ")" << std::endl; }

//
#define SAFE_DELETE(x) { if(x) delete x; x = 0; }

// typedefs
typedef float REAL;
typedef unsigned int INDEX;

typedef Eigen::Matrix<REAL, 2, 1> vec2;
typedef Eigen::Matrix<REAL, 3, 1> vec3;
typedef Eigen::Matrix<REAL, 4, 1> vec4;
typedef Eigen::Matrix<REAL, Eigen::Dynamic, 1> vecX;
typedef Eigen::Matrix<INDEX, 2, 1> ivec2;
typedef Eigen::Matrix<INDEX, 3, 1> ivec3;
typedef Eigen::Matrix<INDEX, 4, 1> ivec4;
typedef Eigen::Matrix<INDEX, Eigen::Dynamic, 1> ivecX;
typedef Eigen::Matrix<REAL, 3, 3> mat3;
typedef Eigen::Matrix<REAL, 4, 4> mat4;
typedef Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic> matX;
typedef Eigen::Quaternion<REAL> quat;
typedef Eigen::AngleAxis<REAL> anax;

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(vec4)

#endif //__PLATFORM_H__
