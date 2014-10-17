#ifndef MATRIXUTILS_H
#define MATRIXUTILS_H

#include "platform.h"

static void computeCovarianceMatrix(const std::vector<vec3>& P0, const std::vector<vec3>& P1, mat3& COV)
{
    if(P0.size() != P1.size())
    {
        PRINTERROR("computeCovarianceMatrix error: size mismatch");
        return;
    }

    COV.setZero();
    for(unsigned int i = 0; i < P0.size(); ++i)
    {
        COV += (P1[i] * P0[i].transpose());
    }
}

static void computeCovarianceMatrix(const std::vector<vec3>& P0, const vec3& cog0,
                                    const std::vector<vec3>& P1, const vec3& cog1,
                                    mat3& COV)
{
    if(P0.size() != P1.size())
    {
        PRINTERROR("computeCovarianceMatrix error: size mismatch");
        return;
    }

    COV.setZero();
    for(unsigned int i = 0; i < P0.size(); ++i)
    {
        COV += ((P1[i] - cog1) * (P0[i] - cog0).transpose());
    }
}

static void computeSVD(const matX& matrix, matX& U, matX& V, vecX& sigma)
{
    Eigen::JacobiSVD<matX> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    sigma = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();
}

static void computeSVD(const mat3& matrix, mat3& U, mat3& V, vec3& sigma)
{
    Eigen::JacobiSVD<mat3> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    sigma = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();
}

static void cleanMatrix(matX& M, float eps = 0.000001)
{
    for(int i = 0; i < M.cols(); ++i)
    {
        for(int j = 0; j < M.rows(); ++j)
        {
            if(std::abs(M(i,j)) < eps)
                M(i,j) = 0;
        }
    }
}

static void computeBestRotation(const mat3& M, mat3& R)
{
	mat3 U,V;
	vec3 sigma;
	
	computeSVD(M,U,V,sigma);

	mat3 V_t = V.transpose();
	R = U * V_t;

	if(R.determinant() < REAL(0))
	{
		int sm_id = 0;
		REAL smallest = sigma[sm_id];
		for(short l = 0; l < 3; ++l)
		{
			if(sigma[l] < smallest)
			{
				smallest = sigma[l];
				sm_id = l;
			}
		}

		U.col(sm_id) *= -1;
		R = U*V_t;
	}
}

static void separateRotationAndShear(const matX& M, matX& R, matX& S)
{
    matX U, V;
    vecX sigma;

    computeSVD(M,U,V,sigma);

    R = U*V.transpose();

    if(R.determinant() < 0)
    {
        R = R.transpose();
    }

    S = R.inverse()*M;
}

static REAL computeCOV_and_EigenVector_Mises(const std::vector< std::vector<REAL> >& M, std::vector<REAL>& seed, std::vector<REAL>& eVec, int numIter = 10)
{
    REAL eVal;
    REAL old_eVal = std::numeric_limits<REAL>::max();
    eVec.resize(seed.size(),0);

    for(int i = 0; i < numIter; ++i)
    {
        // estimate the new eVec
        for(unsigned int j = 0; j < M.size(); ++j)
        {
            REAL dot = 0;
            for(unsigned int k = 0; k < seed.size(); ++k)
            {
                dot += M[j][k]*seed[k];
            }
            for(unsigned int k = 0; k < seed.size(); ++k)
            {
                eVec[k] += dot*M[j][k];
            }
        }

        // estimate the new eVal
        eVal = 0;
        for(unsigned int j = 0; j < seed.size(); ++j)
        {
            eVal += eVec[j]*eVec[j];
        }
        eVal = std::sqrt(eVal);

        // early termination test
        if(std::abs(eVal-old_eVal)/fabs(eVal) < std::numeric_limits<REAL>::epsilon())
        {
            i = numIter;
        }

        old_eVal = eVal;

        // normalize eVec and set result to seed for next iter
        for(unsigned int j = 0; j < seed.size(); ++j)
        {
            eVec[j] /= eVal;
            seed[j] = eVec[j];
        }
    }

    return eVal;
}

static void computePerspeciveProjectionMatrix(const REAL& fovy,
                                                const REAL& aspect,
                                                const REAL& near_plane,
                                                const REAL& far_plane,
                                                mat4& outM)
{
    REAL fovy_rad = fovy * float(M_PI) / 360.0f;
    REAL y_scale = cosf(fovy_rad) / sinf(fovy_rad);
    REAL x_scale = y_scale / aspect;
    REAL c_23 = -((far_plane+near_plane) / (far_plane-near_plane));
    REAL c_33 = -((2.0f*near_plane*far_plane) / (far_plane-near_plane));

    outM << x_scale, 0, 0, 0,
            0, y_scale, 0, 0,
            0, 0, c_23, c_33,
            0, 0, -1, 0;
}

static void computeOrthoProjectionMatrix(const REAL& left,
                                            const REAL& right,
                                            const REAL& bottom,
                                            const REAL& top,
                                            const REAL& near_plane,
                                            const REAL& far_plane,
                                            mat4& out)
{
    REAL x_scale = REAL(2) / (right - left);
    REAL y_scale = REAL(2) / (top - bottom);
    REAL z_scale = -REAL(2) / (far_plane - near_plane);
    REAL x_offset = -(right+left)/(right-left);
    REAL y_offset = -(top+bottom)/(top-bottom);
    REAL z_offset = -(far_plane+near_plane)/(far_plane-near_plane);

    out << x_scale, 0, 0, x_offset,
            0, y_scale, 0, y_offset,
            0, 0, z_scale, z_offset,
            0, 0, 0, 1;
}

static void computeLookAtMatrix(const vec3& eye, const vec3& center, const vec3& up, mat4& outM)
{
    vec3 z = (eye - center).normalized();
    vec3 x = (up.cross(z)).normalized();
    vec3 y = z.cross(x);

    mat4 rotM;
    rotM << x.x(), x.y(), x.z(), 0,
            y.x(), y.y(), y.z(), 0,
            z.x(), z.y(), z.z(), 0,
            0,0,0,1;

    mat4 trfM;
    trfM << 1, 0, 0, -eye.x(),
            0, 1, 0, -eye.y(),
            0, 0, 1, -eye.z(),
            0, 0, 0, 1;

    outM = rotM * trfM;
}

static void extractEyePosFromViewMatrix(const mat4& viewM, vec3& out)
{
    mat4 Rt = viewM.transpose();
    Rt(3,0) = Rt(3,1) = Rt(3,2) = REAL(0);
    mat4 D = Rt * viewM;
    out = -D.block<3,1>(0,3);
}

#endif // MATRIXUTILS_H
