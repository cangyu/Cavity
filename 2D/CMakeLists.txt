cmake_minimum_required(VERSION 3.10)
project(Cavity)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
add_executable(${PROJECT_NAME} src/main.cc)

