cmake_minimum_required(VERSION 3.10)

project(CAVITY VERSION 1.2.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(TYDF STATIC
	3rd_party/TYDF/src/common.cc
	3rd_party/TYDF/src/xf.cc)

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

add_executable(${PROJECT_NAME}
	src/IO.cc
	src/IC.cc
	src/BC.cc
	src/CHEM.cc
	src/Discretization.cc
	src/Flux.cc
	src/Gradient.cc
	src/LeastSquare.cc
	src/PoissonEqn.cc
	src/Diagnose.cc
	src/Miscellaneous.cc
	src/main.cc)

target_link_libraries(${PROJECT_NAME} PRIVATE TYDF SXAMG)

install(TARGETS ${PROJECT_NAME} 
	RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)

