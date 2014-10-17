#include "shaderprogram.h"

#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/GL.h>

ShaderProgram::ShaderProgram()
    : mIsCompute(false)
{
    mProgram = glCreateProgram();
    mPaths.clear();
}

ShaderProgram::~ShaderProgram()
{
    if(glIsProgram(mProgram))
    {
        glDeleteProgram(mProgram);
    }
    mPaths.clear();
}

bool ShaderProgram::addShaderFromSource(const ShaderType &type, const std::string &filename)
{
    if(mIsCompute)
    {
        LOG("program is already a compute shader");
        return false;
    }

    GLuint shader = 0;

    switch(type)
    {
    case SHADER_VERTEX:
        shader = glCreateShader(GL_VERTEX_SHADER);
        break;

    case SHADER_GEOMETRY:
        shader = glCreateShader(GL_GEOMETRY_SHADER);
        break;

    case SHADER_FRAGMENT:
        shader = glCreateShader(GL_FRAGMENT_SHADER);
        break;

    case SHADER_COMPUTE:
        if(mPaths.size() == 0)
        {
            shader = glCreateShader(GL_COMPUTE_SHADER);
        }
        else
        {
            LOG("compute shader must be a single shader program");
            return false;
        }
        break;

    default:
        LOG("unknown shader type");
        return false;
    }

    int size = 0;
    std::ifstream inF(filename, std::ifstream::ate);
    if(inF.bad())
    {
        LOG("could not open file " << filename.c_str());
        return false;
    }

    // get data from file
    size = (int) inF.tellg();
    char* pl = new char[size];
    memset(pl,0,sizeof(char)*size);
    inF.seekg(0,std::ios::beg);
    inF.read(pl,size);
    inF.close();
    const char *c_pl = pl;

    GLenum err = 0;

    glShaderSource(shader, 1, &c_pl, &size);
    err = glGetError();
    if(err)
    {
        LOG("glShaderSource error: " << std::hex << err);
        return false;
    }

    glCompileShader(shader);
    err = glGetError();
    if(err)
    {
        LOG("glCompileShader error: " << std::hex << err);
        return false;
    }

    printShaderInfoLog(shader);

    glAttachShader(mProgram, shader);
    err = glGetError();
    if(err)
    {
        LOG("glAttachShader error: " << std::hex << err);
        return false;
    }

    mPaths.push_back(filename);

    if(type == SHADER_COMPUTE)
        mIsCompute = true;

    return true;
}

bool ShaderProgram::link()
{
    GLenum err = 0;
    glLinkProgram(mProgram);
    err = glGetError();
    if(err)
    {
        LOG("glLinkProgram error: " << std::hex << err);
        return false;
    }

    printProgramInfoLog();

    return true;
}

void ShaderProgram::bind()
{
    glUseProgram(mProgram);
}

void ShaderProgram::release()
{
    glUseProgram(0);
}

void ShaderProgram::execute(const unsigned int& numX,
                            const unsigned int& numY,
                            const unsigned int& numZ)
{
    if(!mIsCompute)
    {
        LOG("not a compute shader - execute failed");
        return;
    }

    glDispatchCompute(numX, numY, numZ);
}

int ShaderProgram::attribute(const std::string& name)
{
    return glGetAttribLocation(mProgram, name.c_str());
}

int ShaderProgram::uniform(const std::string& name)
{
    return glGetUniformLocation(mProgram, name.c_str());
}

unsigned int ShaderProgram::getId()
{
    return mProgram;
}

void ShaderProgram::printShaderInfoLog(unsigned int shader)
{
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if(status == GL_FALSE)
    {
        int logLen = 0;
        int charsWritten = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, & logLen);
        if(logLen > 0)
        {
            char* infoLog = new char[logLen];
            glGetShaderInfoLog(shader, logLen, &charsWritten, infoLog);
            LOG(infoLog);
            delete [] infoLog;
        }
    }
}

void ShaderProgram::printProgramInfoLog()
{
    int logLen = 0;
    int charsWritten  = 0;
    glGetProgramiv(mProgram, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0)
    {
        char* infoLog = new char[logLen];
        glGetProgramInfoLog(mProgram, logLen, &charsWritten, infoLog);

        LOG("program infor log: " << infoLog);
        delete [] infoLog;
    }
}
