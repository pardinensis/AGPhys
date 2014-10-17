#ifndef CAMERA_H
#define CAMERA_H

#include "platform.h"

class ShaderProgram;

class Camera
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! constructor
    Camera();

    //! copy constructor
    Camera(const Camera& other);

    //! destructor
    ~Camera();

    //! generates a view matrix
    void setLookAt(const vec3& eye, const vec3& center, const vec3& up);

    //! generates a perspective projection matrix
    void setPerspectiveProjection(const REAL& fovy,
                                    const REAL& aspect,
                                    const REAL& near_plane,
                                    const REAL& far_plane);

    //! generates a orthographic projection matrix
    void setOrthoProjection(const REAL& left,
                                const REAL& right,
                                const REAL& bottom,
                                const REAL& top,
                                const REAL& near_plane,
                                const REAL& far_plane);

    //! sets the matrices to the bound shader locations
    void set(const int& viewLocation, const int& projLocation);

    void getViewMatrix(mat4& M) const;

    void getProjMatrix(mat4& M) const;

protected:

    mat4 mViewMatrix;

    mat4 mProjectionMatrix;
};

class SphericalCamera : public Camera
{

public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	//! constructor
	SphericalCamera();

	//! destructor
	~SphericalCamera();

	//! set the camera radius, position and look-at-point and updates matrices
	void set(REAL radius, REAL angleX, REAL angleY, const vec3& center);

	//! set radius
	void setRadius(REAL radius);

	//! set the camera position
	void setAngles(REAL angleX, REAL angleY);

	//! set the look-at-point
	void setCenter(const vec3& center);

	//! add angles
	void addAngles(REAL dAngleX, REAL dAngleY);

	//! add to center point
	void addCenter(const vec3& dCenter);

	//! zoom functionality
	void addRadius(REAL dRadius);

protected:

	void checkForConstraints();

	void updateViewMatrix();

	REAL mRadius;

	REAL mAngleX;

	REAL mAngleY;

	vec3 mCenter;
};

class ArcballCamera : public Camera 
{

public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	ArcballCamera();

	ArcballCamera(int screenWidth, int screenHeight);

	~ArcballCamera();

	void resize(int w, int h);

	void startMovement(int x, int y);

	void move(int x, int y);
	
	void stopMovement();

	void setRadius(REAL r);

	void addRadius(REAL r);

	void setCenter(const vec3& c);
	
	void addCenter(const vec3& c);

protected:

	void convertXY(int x, int y, vec3& outV);

	void updateViewMatrix();

	bool mMoving;

	int mScreenWidth;

	int mScreenHeight;
	
	REAL mRadius;

	vec3 mCenter;

	vec3 mStartRotation;

	vec3 mCurrentRotation;
	
};

#endif // CAMERA_H
