#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <algorithm>
#include "misc.h"
#include "camera.h"
#include "shaderprogram.h"
#include "platform.h"
#include "fileutils.h"
#include "matrixutils.h"

//#define DEBUG_TO_COUT

//#define VOLCANO
//#define PARTICLE_TEST

#define WIDTH 1024
#define HEIGHT 768
#define NUM_PARTICLES 4096
#define PARTICLE_GRIDSIZE 8
#define PARTICLE_GRIDSCALE 1
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

// forward declaration
extern "C" 
{
    void launchParticleKernel(float *ptVbo, float* f_tmp, int numParticles, float t, float dT);
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

    // create particles
    cudaMalloc((void**) &particle_f_tmp, NUM_PARTICLES * 3 * sizeof(float));
    glGenBuffers(1, &particlesVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
    vertexSize = 12; //layout: 3 position, 3 impulse, 4 color, 1 age, 1 mass
    v = new float[NUM_PARTICLES * vertexSize];
    float distance = PARTICLE_GRIDSCALE / (float)std::max(PARTICLE_GRIDSIZE - 1, 1);
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
        int idx_x = i % PARTICLE_GRIDSIZE;
        int idx_y = i / (PARTICLE_GRIDSIZE * PARTICLE_GRIDSIZE);
        int idx_z = (i / PARTICLE_GRIDSIZE) % PARTICLE_GRIDSIZE;
        float posx = idx_x * distance - PARTICLE_GRIDSCALE/2.f;
        float posy = idx_y * distance + PARTICLE_SPAWN_HEIGHT;
        float posz = idx_z * distance - PARTICLE_GRIDSCALE/2.f;
        float jitter = distance * PARTICLE_JITTER_FACTOR;
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

void display()
{
	// simulation
	if(runSimulation)
    {
        // TODO: 1c)
        cudaGLMapBufferObject((void**) &particle_data, particlesVBO);

        const REAL timestep = REAL(0.005);
        launchParticleKernel(particle_data, particle_f_tmp, NUM_PARTICLES, t, timestep);
        t += timestep;

#ifdef DEBUG_TO_COUT
        float tmp[12];
        cudaMemcpy(tmp, particle_data, 12*sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < 12; ++i)
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
