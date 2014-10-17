#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include "platform.h"

class ShaderProgram
{

public:

    enum ShaderType
    {
        SHADER_VERTEX,
        SHADER_GEOMETRY,
        SHADER_FRAGMENT,
        SHADER_COMPUTE
        // TO BE CONTINUED
    };

    //! constructor
    ShaderProgram();

    //! destructor
    ~ShaderProgram();

    //! add a single file as shader source and compiles it
    bool addShaderFromSource(const ShaderType& type, const std::string& filename);

    //! links the program
    bool link();

    //! bind this program for usage
    void bind();

    //! release the shader (use program 0)
    void release();

    //! in case of compute shader execute
    void execute(const unsigned int &numX, const unsigned int &numY, const unsigned int &numZ);

    //! get the attribute location and returns -1 on failure
    int attribute(const std::string& name);

    //! get the uniform location and returns -1 on failure
    int uniform(const std::string& name);

    //! get the id
    unsigned int getId();

protected:

    //! can be invoked after compiling to check shader comp. status
    void printShaderInfoLog(unsigned int shader);

    //! can be invoked after linking to get a shader debug output
    void printProgramInfoLog();

    //! flag if this is a compute shader
    bool mIsCompute;

    //! list of all paths (for future recompile checking)
    std::vector<std::string> mPaths;

    //! the program handle
    unsigned int mProgram;

};

#endif // SHADER_H
