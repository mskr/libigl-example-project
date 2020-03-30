cmake_minimum_required(VERSION 3.1)

set(CMAKE_CONFIGURATION_TYPES RelWithDebInfo CACHE TYPE  INTERNAL FORCE ) # set release mode to make libigl much faster

project(example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       ON)

find_package(LIBIGL REQUIRED QUIET)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../executable ) # place exe where it can be found
set_property(GLOBAL PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/executable/RelWithDebInfo) # set cwd to exe location, so that relative file paths can be used

# Add your project files
add_subdirectory("../assimp" "../assimp/build") # folder where a dependent CMakeLists.txt is
include_directories("../assimp/include" "../assimp/build/include")
file(GLOB SRCFILES main.cpp)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::triangle assimp) # magically finds assimp/build/code/Debug/assimp-vc141-mtd.lib
