#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <algorithm>
#include <ctime>
#include "misc.h"
#include "camera.h"
#include "shaderprogram.h"
#include "platform.h"
#include "fileutils.h"
#include "matrixutils.h"

#include <Windows.h>

//#define DEBUG_TO_COUT
//#define VOLCANO
//#define PARTICLE_TEST

#define NUM_PARTICLES 1024 * 64
#define WIDTH 1024
#define HEIGHT 768
#define PARTICLE_GRIDSIZE_X 16
#define PARTICLE_GRIDSIZE_Z 16
#define PARTICLE_GRIDSCALE 0.2
#define PARTICLE_JITTER_FACTOR 0.2
#define PARTICLE_SPAWN_HEIGHT 3

// globals
ArcballCamera* camera;
ShaderProgram* volcanoShaderProgram;
ShaderProgram* fireShaderProgram;
GLuint volcanoVBO;
GLuint volcanoIBO;
GLuint volcanoNumTriangles = 0;
GLuint particlesVBO;
GLuint particlesNumVertices = 0;
struct cudaGraphicsResource* cuVboResource;
bool runSimulation = false;
REAL t = REAL(0);

float* particle_data;
float* particle_f_tmp;
float* particle_sas_key1, *particle_sas_value1;

// forward declaration
extern "C" 
{
    void launchParticleKernel(float *ptVbo, float* f_tmp,
                              float* sas_key1, float* sas_value1,
                              int numParticles, float t, float dT);
    void launchKernel(float* a, float* b, float* c);
}

bool initGL()
{
	glewInit();

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(TRUE);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0f);
	glClearStencil(0);
	glShadeModel(GL_SMOOTH);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glReadBuffer(GL_BACK);
	glDrawBuffer(GL_BACK);
	glDisable(GL_STENCIL_TEST);
	glStencilMask(0xFFFFFFFF);
	glStencilFunc(GL_EQUAL, 0x00000000, 0x00000001);
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDisable(GL_ALPHA_TEST);
	glDisable(GL_DITHER);

	if(!glewIsSupported("GL_VERSION_2_0"))
	{
		LOG("your system does not support open gl shader model 2");
		return false;
	}
	
	// init shader
	volcanoShaderProgram = new ShaderProgram();
	volcanoShaderProgram->addShaderFromSource(ShaderProgram::SHADER_VERTEX, "./media/shader/volcano.vs");
	volcanoShaderProgram->addShaderFromSource(ShaderProgram::SHADER_FRAGMENT, "./media/shader/volcano.fs");
	if(!volcanoShaderProgram->link())
	{
		LOG("volcano shader linkage error");
		return false;
	}
	
	fireShaderProgram = new ShaderProgram();
    fireShaderProgram->addShaderFromSource(ShaderProgram::SHADER_VERTEX, "./media/shader/sphere.vs");
    fireShaderProgram->addShaderFromSource(ShaderProgram::SHADER_GEOMETRY, "./media/shader/sphere.gs");
    fireShaderProgram->addShaderFromSource(ShaderProgram::SHADER_FRAGMENT, "./media/shader/sphere.fs");
	if(!fireShaderProgram->link())
	{
		LOG("particle shader linkage error");
		return false;
	}
	
	return true;
}

void initBuffers()
{
    float* v;
    unsigned int vertexSize;
#ifdef VOLCANO
	// load volcano
	std::vector<vec2> _ST(0);
	std::vector<vec3> _V(0), _N(0);
	std::vector<vec4> _C(0);
    std::vector<ivec3> _T(0);
	importTriangleMeshFromOFF("./media/meshes/volcano.off", _V, _N, _T);

	glGenBuffers(1, &volcanoVBO);
	glBindBuffer(GL_ARRAY_BUFFER, volcanoVBO);
    vertexSize = 12; // vertex interleaved layout: 3 pos, 3 norm, 4 col, 2 uv
    v = new float[_V.size() * vertexSize];
	for(unsigned int i = 0; i < _V.size(); ++i)
    {
		v[vertexSize*i+0] = _V[i].x();
		v[vertexSize*i+1] = _V[i].y();
		v[vertexSize*i+2] = _V[i].z();
		v[vertexSize*i+3] = _N[i].x();
		v[vertexSize*i+4] = _N[i].y();
		v[vertexSize*i+5] = _N[i].z();
		v[vertexSize*i+6] = (_C.size() == _V.size()) ? _C[i].x() : 1;
		v[vertexSize*i+7] = (_C.size() == _V.size()) ? _C[i].y() : 1;
		v[vertexSize*i+8] = (_C.size() == _V.size()) ? _C[i].z() : 1;
		v[vertexSize*i+9] = (_C.size() == _V.size()) ? _C[i].w() : 1;
		v[vertexSize*i+10] = (_ST.size() == _V.size()) ? _ST[i].x() : 0;
		v[vertexSize*i+11] = (_ST.size() == _V.size()) ? _ST[i].y() : 0;
	}
	glBufferData(GL_ARRAY_BUFFER, _V.size()*vertexSize*sizeof(float), &v[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	delete [] v;

	glGenBuffers(1, &volcanoIBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, volcanoIBO);
	unsigned int* t = new unsigned int[_T.size()*3];
	for(unsigned int i = 0; i < _T.size(); ++i)
	{
		t[3*i+0] = _T[i].x();
		t[3*i+1] = _T[i].y();
		t[3*i+2] = _T[i].z();
	}
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, _T.size()*3*sizeof(unsigned int), &t[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	volcanoNumTriangles = _T.size();
    delete [] t;
#endif

    cudaMalloc((void**) &particle_f_tmp, NUM_PARTICLES * 3 * sizeof(float));
    cudaMalloc((void**) &particle_sas_key1, NUM_PARTICLES * 2 * sizeof(float));
    cudaMalloc((void**) &particle_sas_value1, NUM_PARTICLES * 2 * sizeof(float));

    // create particles
    glGenBuffers(1, &particlesVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
    vertexSize = 12; //layout: 3 position, 3 impulse, 4 color, 1 age, 1 mass
    v = new float[NUM_PARTICLES * vertexSize];
    for (unsigned int i = 0; i < NUM_PARTICLES; ++i) {
        float radius = random::normal(0.05, 0.003);
#ifdef PARTICLE_TEST
        v[vertexSize*i+0] = random::normal(0.0, 0.03);
        v[vertexSize*i+1] = 1;
        v[vertexSize*i+2] = (i%2) ? 0.4 : -0.4;
        v[vertexSize*i+3] = 0;
        v[vertexSize*i+4] = 0;
        v[vertexSize*i+5] = ((i%2) ? -1 : 1);
#else
        // pos
        int idx_x = i % PARTICLE_GRIDSIZE_X;
        int idx_y = i / (PARTICLE_GRIDSIZE_X * PARTICLE_GRIDSIZE_Z);
        int idx_z = (i / PARTICLE_GRIDSIZE_X) % PARTICLE_GRIDSIZE_Z;
        float posx = idx_x * PARTICLE_GRIDSCALE - PARTICLE_GRIDSCALE*(PARTICLE_GRIDSIZE_X-1)/2.f;
        float posy = idx_y * PARTICLE_GRIDSCALE + PARTICLE_SPAWN_HEIGHT;
        float posz = idx_z * PARTICLE_GRIDSCALE - PARTICLE_GRIDSCALE*(PARTICLE_GRIDSIZE_Z-1)/2.f;
        float jitter = PARTICLE_GRIDSCALE * PARTICLE_JITTER_FACTOR;
        v[vertexSize*i+0] = posx + random::uniform(-jitter, jitter);
        v[vertexSize*i+1] = posy + random::uniform(-jitter, jitter);
        v[vertexSize*i+2] = posz + random::uniform(-jitter, jitter);
        // impulse
        v[vertexSize*i+3] = 0;
        v[vertexSize*i+4] = 0;
        v[vertexSize*i+5] = 0;
#endif
        // color
        v[vertexSize*i+6] = random::uniform(0.3, 1);
        v[vertexSize*i+7] = random::uniform(0.3, 1);
        v[vertexSize*i+8] = random::uniform(0.3, 1);
        v[vertexSize*i+9] = radius;
        // age
        v[vertexSize*i+10] = 0;
        // mass
        v[vertexSize*i+11] = 1;
    }
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES*vertexSize*sizeof(float), &v[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    particlesNumVertices = NUM_PARTICLES;
    delete[] v;

	// connect particle buffer to cuda
	// TODO: 1b)
    cudaGLRegisterBufferObject(particlesVBO);
}

void cuda_test() {
    float a = 33;
    float b = 9;
    float c = 0;
    float *pa, *pb, *pc;
    cudaMalloc((void**) &pa, sizeof(float));
    cudaMalloc((void**) &pb, sizeof(float));
    cudaMalloc((void**) &pc, sizeof(float));
    cudaMemcpy(pa, &a, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pb, &b, sizeof(float), cudaMemcpyHostToDevice);
    launchKernel(pa, pb, pc);
    cudaMemcpy(&c, pc, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(pa);
    cudaFree(pb);
    cudaFree(pc);
    std::cout << " === " << c << " === " << std::endl;
}

void init()
{
    random::init_generator();
    camera = new ArcballCamera();
    vec3 pos = vec3(2, 2, 3);
    camera->setRadius(pos.norm());
    camera->setLookAt(pos, vec3(0,0,0), vec3(0,1,0));
    initGL();
    initBuffers();
    glutPostRedisplay();
}

void deleteBuffers() {
    cudaGLUnregisterBufferObject(particlesVBO);
    cudaFree(particle_f_tmp);
    cudaFree(particle_sas_key1);
    cudaFree(particle_sas_value1);
    cudaFree(particle_data);

    if(glIsBuffer(particlesVBO))
        glDeleteBuffers(1, &particlesVBO);

    if(glIsBuffer(volcanoVBO))
        glDeleteBuffers(1, &volcanoVBO);

    if(glIsBuffer(volcanoIBO))
        glDeleteBuffers(1, &volcanoIBO);
}

void shutdown()
{
    deleteBuffers();
	SAFE_DELETE(camera);
	SAFE_DELETE(volcanoShaderProgram);
	SAFE_DELETE(fireShaderProgram);
}

std::vector<double> msec_times;
ULONGLONG ull_before = 0;
void calc_time_stats() {
    if (msec_times.empty()) return;

//    const double outlier_factor = 10;

    // calculate median
    std::sort(msec_times.begin(), msec_times.end());
    double median = msec_times[(msec_times.size()-1)/2];

    // calculate mean
    double msec_sum = 0;
    unsigned int msec_ctr = 0;
    for (unsigned int i = 0; i < msec_times.size(); ++i) {
        double t = msec_times[i];
//        if (t / median < outlier_factor && median / t < outlier_factor) {
            msec_sum += t;
            ++msec_ctr;
//        }
    }
    double mean = msec_sum / msec_ctr;

    // calculate variance
    double msec_varsum = 0;
    for (unsigned int i = 0; i < msec_times.size(); ++i) {
        double t = msec_times[i];
//        if (t / median < outlier_factor && median / t < outlier_factor) {
            msec_varsum += (t - mean) * (t - mean);
//        }
    }
    double variance = 0;
    if (msec_ctr > 1)
        variance = msec_varsum / (msec_ctr - 1);

    std::cout << std::endl;
    std::cout << "[[ TIMING RESULTS ]]" << std::endl;
    std::cout << "median = " << median << std::endl;
    std::cout << "avg = " << mean << std::endl;
    std::cout << "stddev = " << sqrt(variance) << std::endl;
    std::cout << std::endl;

    msec_times.clear();
}

void display()
{
	// simulation
	if(runSimulation)
    {
        cudaGLMapBufferObject((void**) &particle_data, particlesVBO);

        // Query Performance Counter (does not work correctly)
        // LARGE_INTEGER freq;
        // LARGE_INTEGER before;
        // LARGE_INTEGER after;
        // QueryPerformanceFrequency(&freq);
        // QueryPerformanceCounter(&before);

        // System Time
//        FILETIME ft_before;
//        GetSystemTimeAsFileTime(&ft_before);
//        ULARGE_INTEGER uli_before;
//        uli_before.HighPart = ft_before.dwHighDateTime;
//        uli_before.LowPart = ft_before.dwLowDateTime;

        const REAL timestep = REAL(0.005);
        launchParticleKernel(particle_data, particle_f_tmp,
                             particle_sas_key1, particle_sas_value1,
                             NUM_PARTICLES, t, timestep);
        t += timestep;

        // QueryPerformanceCounter(&after);
        // double msecs = (after.QuadPart - before.QuadPart) * 1000.0 / freq.QuadPart;

        FILETIME ft_after;
        GetSystemTimeAsFileTime(&ft_after);
        ULARGE_INTEGER uli_after;
        uli_after.HighPart = ft_after.dwHighDateTime;
        uli_after.LowPart = ft_after.dwLowDateTime;
        ULONGLONG ull_after = uli_after.QuadPart;

        if (ull_before) {
            unsigned int diff = ull_after - ull_before;
            double msecs = diff / 10000.0;
            msec_times.push_back(msecs);
            std::cout << msecs << std::endl;
        }
        ull_before = ull_after;

#ifdef DEBUG_TO_COUT
        float tmp[2*NUM_PARTICLES];
        cudaMemcpy(tmp, particle_sas_value1, 2*NUM_PARTICLES*sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < 2*NUM_PARTICLES; ++i)
            std::cout << tmp[i] << " ";
        std::cout << std::endl;
#endif

        cudaGLUnmapBufferObject(particlesVBO);
    }

	//
	mat4 viewM, projM, modelM;
	mat3 normalM;
	camera->getViewMatrix(viewM);
	camera->getProjMatrix(projM);
	modelM.setIdentity();
	normalM = viewM.topLeftCorner<3,3>();

	// TODO calc distance dependent point size
	vec3 eyeP;
	extractEyePosFromViewMatrix(viewM, eyeP);
	glPointSize(2.0f);

	// rendering
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// draw volcano
#ifdef VOLCANO
    volcanoShaderProgram->bind();
    glUniformMatrix4fv(volcanoShaderProgram->uniform("view_matrix"), 1, false, viewM.data());
    glUniformMatrix4fv(volcanoShaderProgram->uniform("proj_matrix"), 1, false, projM.data());
    glUniformMatrix4fv(volcanoShaderProgram->uniform("model_matrix"), 1, false, modelM.data());
    glUniformMatrix3fv(volcanoShaderProgram->uniform("normal_matrix"), 1, false, normalM.data());
    glUniform1i(volcanoShaderProgram->uniform("bGlow"), 1);

    glBindBuffer(GL_ARRAY_BUFFER, volcanoVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(12));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(24));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(40));
    glEnableVertexAttribArray(3);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, volcanoIBO);
    glDrawElements(GL_TRIANGLES, 3*volcanoNumTriangles, GL_UNSIGNED_INT, (void*)(0));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    volcanoShaderProgram->release();
#endif

	// draw particles
	fireShaderProgram->bind();
	glUniformMatrix4fv(fireShaderProgram->uniform("view_matrix"), 1, false, viewM.data());
	glUniformMatrix4fv(fireShaderProgram->uniform("proj_matrix"), 1, false, projM.data());
	glUniformMatrix4fv(fireShaderProgram->uniform("model_matrix"), 1, false, modelM.data());
			
	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(0));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(12));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(24));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(40));
	glEnableVertexAttribArray(3);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
#ifdef BLENDING
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);
#endif
	glDrawArrays(GL_POINTS, 0, particlesNumVertices);
#ifdef BLENDING
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
#endif
	
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
    fireShaderProgram->release();
    glutSwapBuffers();
}

void resize(int w, int h)
{
	REAL aspect = REAL(w)/REAL((h == 0) ? 1 : h);
	camera->resize(w, h);
	camera->setPerspectiveProjection(REAL(45), aspect, REAL(0.3), REAL(100.0));
	glViewport(0,0,w,h);
    glutPostRedisplay();
}

void idle()
{
	// step forward
	if(runSimulation)
        t += REAL(0.005);
	
    glutPostRedisplay();
}

void mouseButton(int button, int state, int x, int y)
{
	switch(state)
	{
	case GLUT_DOWN:
		if(button == GLUT_RIGHT_BUTTON)
		{
			// activate camera movement
			camera->startMovement(x,y);
		}
		break;

	case GLUT_UP:
		if(button == GLUT_RIGHT_BUTTON)
		{
			// deactivate camera movement
			camera->stopMovement();
		}
		break;

	default:
		break;
	}
}

void mouseMove(int x, int y)
{
	camera->move(x,y);
    glutPostRedisplay();
}

void key(unsigned char key, int x, int y)
{
    switch(key) {

        case 27: // ESCAPE
			shutdown();
			exit(0);
			break;

        case 32: // SPACE
            runSimulation = !runSimulation;
            if (!runSimulation) {
                calc_time_stats();
                ull_before = 0;
            }
			break;

        case 114: // R
            deleteBuffers();
            initBuffers();
            break;
	}
//    std::cout << (int)key << std::endl;
    glutPostRedisplay();
}


int main(int argc, char* argv[])
{
    cuda_test();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Ball Pit");
    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutIdleFunc(idle);
    glutKeyboardFunc(key);
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMove);
    glutCloseFunc(shutdown);

    init();
    glutMainLoop();
	return 0;
}
