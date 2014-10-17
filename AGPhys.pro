#-------------------------------------------------
#
# Project created by QtCreator 2014-10-17T16:45:03
#
#-------------------------------------------------

## packages
QT       += core
QT       -= gui

## target configuration
TARGET = AGPhys
TEMPLATE = app

## includes
INCLUDEPATH += $$PWD/contrib
INCLUDEPATH += $$PWD/contrib/Eigen
INCLUDEPATH += $$PWD/contrib/freeglut/include
INCLUDEPATH += $$PWD/contrib/glew/include

## libs
LIBS += -L$$PWD/contrib/freeglut/lib -lfreeglut
LIBS += -L$$PWD/contrib/glew/lib -lglew32s

## headers
INCLUDEPATH += $$PWD/include
HEADERS += $$PWD/include/*.h

## sources
SOURCES += main.cpp
SOURCES += $$PWD/source/*.cpp

