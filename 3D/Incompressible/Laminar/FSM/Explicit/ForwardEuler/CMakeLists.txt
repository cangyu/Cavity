cmake_minimum_required(VERSION 3.10)

project(Cavity VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(
	TYDF STATIC
	3rd_party/TYDF/src/common.cc
	3rd_party/TYDF/src/xf.cc
)

target_compile_definitions(TYDF PUBLIC XF_EXTRACT_NODE_CONNECTIVITY)

add_executable(
	${PROJECT_NAME}
	src/IO.cc
	src/IC.cc
	src/BC.cc
	src/CHEM.cc
	src/Discretization.cc
	src/Flux.cc
	src/Gradient.cc
	src/LeastSquare.cc
	src/PoissonEqn.cc
	src/Miscellaneous.cc
	src/main.cc
)

target_link_libraries(
	${PROJECT_NAME} PRIVATE 
	TYDF 
	stdc++fs
)

install(TARGETS ${PROJECT_NAME} 
	RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/test
)
