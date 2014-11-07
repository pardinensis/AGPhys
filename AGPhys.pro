#-------------------------------------------------
#
# Project created by Christian Lobmeier
#
#-------------------------------------------------

CONFIG += console
CONFIG -= qt
CONFIG -= app_bundle

## target configuration
TARGET = AGPhys
TEMPLATE = app

## linking issue stuff
QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrt.lib
QMAKE_LFLAGS_DEBUG = /NODEFAULTLIB:msvcrtd.lib
CPPFLAGS = /MT -D_ITERATOR_DEBUG_LEVEL=0
QMAKE_CFLAGS_RELEASE += $$CPPFLAGS
QMAKE_CXXFLAGS_RELEASE += $$CPPFLAGS
QMAKE_CFLAGS_DEBUG += $$CPPFLAGS
QMAKE_CXXFLAGS_DEBUG += $$CPPFLAGS

## cuda directory
CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5"

## includes
INCLUDEPATH += $$PWD/contrib
INCLUDEPATH += $$PWD/contrib/Eigen
INCLUDEPATH += $$PWD/contrib/freeglut/include
INCLUDEPATH += $$PWD/contrib/glew/include
INCLUDEPATH += $$CUDA_DIR/include

## libs
LIBS += -L$$PWD/contrib/freeglut/lib -lfreeglut
LIBS += -L$$PWD/contrib/glew/lib -lglew32s
LIBS += -L$$CUDA_DIR/lib/Win32 -lcuda -lcudart

## headers
INCLUDEPATH += $$PWD/include
HEADERS += $$PWD/include/*.h \
    include/misc.h

## sources
SOURCES += main.cpp
SOURCES += \
    $$PWD/source/camera.cpp \
    $$PWD/source/intersection.cpp \
    $$PWD/source/ray.cpp \
    $$PWD/source/shaderprogram.cpp

## shaders
OTHER_FILES += $$PWD/media/shader/* \
    media/shader/sphere.gs \
    media/shader/sphere.vs \
    media/shader/sphere.fs

## cuda sources
CUDA_HEADERS += \
    $$PWD/cuda/random.cuh \
    $$PWD/cuda/integrator.cuh \
OTHER_FILES += $$CUDA_HEADERS
CUDA_SOURCES += \
    $$PWD/cuda/particle.cu \
    $$PWD/cuda/hello_world.cu
OTHER_FILES += $$CUDA_SOURCES

## cuda libs
CUDA_LIBS = cuda cudart
NVCC_LIBS = $$join(CUDA_LIBS,' -l','-l', '')
CUDA_INC = $$join(INCLUDEPATH,'" -I "','-I "','"')

## cuda compiler options
SYSTEM_TYPE = 32
CUDA_ARCH = sm_20

## cuda compiler
cuda.input = CUDA_SOURCES
cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$CUDA_INC $$NVCC_LIBS \
    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda
