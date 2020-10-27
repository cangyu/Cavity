cmake_minimum_required(VERSION 3.10)

project(IncompressibleFlow VERSION 1.3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(SXAMG STATIC
	3rd_party/sxamg/src/mat.c
	3rd_party/sxamg/src/vec.c
	3rd_party/sxamg/src/utils.c
	3rd_party/sxamg/src/amg-coarse-solver.c
	3rd_party/sxamg/src/amg-cycle.c
	3rd_party/sxamg/src/amg-setup.c
	3rd_party/sxamg/src/amg-utils.c
	3rd_party/sxamg/src/op-blas.c
	3rd_party/sxamg/src/op-coarsen.c
	3rd_party/sxamg/src/op-inter.c
	3rd_party/sxamg/src/op-smoother.c
	3rd_party/sxamg/src/sxamg.c)

target_include_directories(SXAMG PRIVATE 3rd_party/sxamg/include)

find_package(Eigen3 3.3.7 REQUIRED)

add_executable(CAVITY
	src/IO.cc
	src/IC.cc
	src/BC.cc
	src/CHEM.cc
	src/Discretization.cc
	src/Gradient.cc
	src/LeastSquare.cc
	src/PoissonEqn.cc
	src/Diagnose.cc
	src/Miscellaneous.cc
	src/main.cc)

target_link_libraries(CAVITY PRIVATE SXAMG Eigen3::Eigen)

install(TARGETS CAVITY RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
