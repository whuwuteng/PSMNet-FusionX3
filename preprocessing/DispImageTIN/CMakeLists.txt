cmake_minimum_required(VERSION 3.10)

project(DispTool)

SET(CMAKE_BUILD_TYPE "Release")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wall -g -ggdb ")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# set output dir
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

find_package(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# error for cmake-3.20.2
find_package(OpenImageIO REQUIRED)
include_directories(${OPENIMAGEIO_INCLUDES})

# install boost by apt-get method
find_package(Boost 1.65 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/IO io_src)
include_directories(${CMAKE_CURRENT_LIST_DIR}/IO)
include_directories(${CMAKE_CURRENT_LIST_DIR}/triangle)

add_library(Triangulation SHARED ${CMAKE_CURRENT_LIST_DIR}/triangle/triangle.cpp)
add_executable(CreateDispDenseTIN ${CMAKE_CURRENT_LIST_DIR}/DispImageTIN.cpp ${io_src})

target_link_libraries(CreateDispDenseTIN OpenImageIO)
target_link_libraries(CreateDispDenseTIN Triangulation)


