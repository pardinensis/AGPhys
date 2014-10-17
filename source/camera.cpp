#include "camera.h"
#include "shaderprogram.h"
#include "matrixutils.h"

#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/GL.h>

Camera::Camera()
{
    mViewMatrix.setIdentity();
    mProjectionMatrix.setIdentity();
}

Camera::Camera(const Camera &other)
{
    mViewMatrix = other.mViewMatrix;
    mProjectionMatrix = other.mProjectionMatrix;
}

Camera::~Camera()
{
    mViewMatrix.setIdentity();
    mProjectionMatrix.setIdentity();
}

void Camera::setLookAt(const vec3 &eye, const vec3 &center, const vec3 &up)
{
    computeLookAtMatrix(eye, center, up, mViewMatrix);
}

void Camera::setPerspectiveProjection(const REAL& fovy,
                                        const REAL& aspect,
                                        const REAL& near_plane,
                                        const REAL& far_plane)
{
    computePerspeciveProjectionMatrix(fovy, aspect, near_plane, far_plane, mProjectionMatrix);
}

void Camera::setOrthoProjection(const REAL& left,
                                    const REAL& right,
                                    const REAL& bottom,
                                    const REAL& top,
                                    const REAL& near_plane,
                                    const REAL& far_plane)
{
    computeOrthoProjectionMatrix(left, right, bottom, top, near_plane, far_plane, mProjectionMatrix);
}

void Camera::set(const int& viewLocation, const int& projLocation)
{
    glUniformMatrix4fv(viewLocation, 1, false, mViewMatrix.data());
    glUniformMatrix4fv(projLocation, 1, false, mProjectionMatrix.data());
}

void Camera::getViewMatrix(mat4 &M) const
{
    M = mViewMatrix;
}

void Camera::getProjMatrix(mat4& M) const
{
    M = mProjectionMatrix;
}

//! spherical camera - inherited
SphericalCamera::SphericalCamera() :
	Camera(),
	mRadius(1),
	mAngleX(0),
	mAngleY(0)
{
	mCenter = vec3(0,0,0);
}

SphericalCamera::~SphericalCamera()
{
	//
}

void SphericalCamera::set(REAL radius, REAL angleX, REAL angleY, const vec3& center)
{
	mRadius = radius;
	mAngleX = angleX;
	mAngleY = angleY;
	mCenter = center;
	checkForConstraints();
	updateViewMatrix();
}

void SphericalCamera::setRadius(REAL radius)
{
	set(radius, mAngleX, mAngleY, mCenter);
}

void SphericalCamera::setAngles(REAL angleX, REAL angleY)
{
	set(mRadius, angleX, angleY, mCenter);
}

void SphericalCamera::setCenter(const vec3& center)
{
	set(mRadius, mAngleX, mAngleY, center);
}

void SphericalCamera::addAngles(REAL dAngleX, REAL dAngleY)
{
	setAngles(mAngleX+dAngleX, mAngleY+dAngleY);
}

void SphericalCamera::addCenter(const vec3& dCenter)
{
	setCenter(mCenter+dCenter);
}

void SphericalCamera::addRadius(REAL dRadius)
{
	setRadius(mRadius+dRadius);
}

void SphericalCamera::checkForConstraints()
{
	mAngleX = std::max(mAngleX, REAL(-M_PI*0.49));
	mAngleX = std::min(mAngleX, REAL(M_PI*0.49));
	//mRadius = std::max(mRadius, REAL(0.01));
}

void SphericalCamera::updateViewMatrix()
{
	vec3 eye = mCenter + vec3(mRadius*cos(mAngleX)*cos(mAngleY), mRadius*sin(mAngleX), mRadius*cos(mAngleX)*sin(mAngleY));
	setLookAt(eye, mCenter, vec3(0,1,0));
}

//! arcball
ArcballCamera::ArcballCamera()
	: Camera(),
	mMoving(false),
	mScreenWidth(0),
	mScreenHeight(0),
	mRadius(3)
{
	//
	mCenter = vec3(0,0,0);
}

ArcballCamera::ArcballCamera(int screenWidth, int screenHeight)
	: Camera(),
	mMoving(false),
	mScreenWidth(screenWidth),
	mScreenHeight(screenHeight),
	mRadius(3)
{
	//
	mCenter = vec3(0,0,0);
}

ArcballCamera::~ArcballCamera()
{
	//
}

void ArcballCamera::resize(int w, int h)
{
	mScreenWidth = w;
	mScreenHeight = h;
	mMoving = false;
}

void ArcballCamera::startMovement(int x, int y)
{
	convertXY(x, y, mStartRotation);	
	mCurrentRotation = mStartRotation;
	mMoving = true;
}

void ArcballCamera::move(int x, int y)
{
	if(mMoving)
	{
		convertXY(x, y, mCurrentRotation);
		updateViewMatrix();
		mStartRotation = mCurrentRotation;
	}
}

void ArcballCamera::stopMovement()
{
	mMoving = false;
}

void ArcballCamera::setRadius(REAL r)
{
	mRadius = r;
	r = std::max(REAL(0.1), mRadius);
	updateViewMatrix();
}

void ArcballCamera::addRadius(REAL r)
{
	REAL scroll = REAL(0.2)*(std::exp(mRadius)-1);
	scroll = std::max(REAL(0), scroll);
	scroll = std::min(REAL(10), scroll);

	mRadius += scroll*r;
	r = std::max(REAL(0.1), mRadius);
	updateViewMatrix();
}

void ArcballCamera::setCenter(const vec3& c)
{
	mCenter = c;
	updateViewMatrix();
}

void ArcballCamera::addCenter(const vec3& c)
{
	mCenter += c;
	updateViewMatrix();
}

void ArcballCamera::convertXY(int x, int y, vec3& outV)
{
	REAL _x = REAL(2)*REAL(x)/(mScreenWidth) - REAL(1);
	REAL _y = REAL(2)*REAL(y)/(mScreenHeight) - REAL(1);
	outV[0] = REAL(_x);
	outV[1] = -REAL(_y);
	
	REAL d = (_x*_x+_y*_y);
	if(d <= REAL(0.5))
	{
		outV[2] = REAL(std::sqrt(REAL(1)-d));

	} else {
		outV[2] = REAL(0.5/std::sqrt(d));
	}

	outV.normalize();
}

void ArcballCamera::updateViewMatrix()
{
	mat4 Mr, Mt, Mtc;
	mat3 Rupdate;
	mat3 Rold = mViewMatrix.topLeftCorner<3,3>();
	
	if(mStartRotation != mCurrentRotation)
	{
		// calculate angle-axis between start and current rotation
		REAL angle = std::acos(std::min(REAL(1), REAL(mStartRotation.dot(mCurrentRotation))));
		vec3 axisEye = mStartRotation.cross(mCurrentRotation);
		axisEye.normalize();
		mat3 eye2WorldMat = mViewMatrix.topLeftCorner<3,3>().inverse();
		vec3 axisWorld = eye2WorldMat * axisEye;

		// transform the current view matrix
		anax T(REAL(2)*angle, axisWorld);
		mat3 R = T.toRotationMatrix();

		Rupdate = Rold*R;
	}
	else
	{
		mat3 R;
		R.setIdentity();
		Rupdate = Rold * R;
	}
	
	Mr << Rupdate, vec3(0,0,0), 0, 0, 0, 1; 	
	Mt = Eigen::Affine3f(Eigen::Translation3f(0,0,-mRadius)).matrix();
	Mtc = Eigen::Affine3f(Eigen::Translation3f(mCenter)).matrix();
	mViewMatrix = Mt * Mr * Mtc;
}

