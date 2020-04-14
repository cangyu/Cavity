cmake_minimum_required(VERSION 3.10)

project(Cavity VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(
	TYDF STATIC
	3rd_party/TYDF/src/common.cc
	3rd_party/TYDF/src/xf.cc
)

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
)

target_compile_definitions(
	${PROJECT_NAME} PRIVATE 
	XF_EXTRACT_NODE_CONNECTIVITY
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_definitions(
		${PROJECT_NAME} PRIVATE 
		_CRT_SECURE_NO_WARNINGS
	)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_link_libraries(
		${PROJECT_NAME} PRIVATE
		stdc++fs
	)
endif()

install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/test
)
