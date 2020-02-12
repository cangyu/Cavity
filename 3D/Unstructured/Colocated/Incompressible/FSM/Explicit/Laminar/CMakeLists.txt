cmake_minimum_required(VERSION 3.10)

project(Cavity)

set(CMAKE_CXX_STANDARD 17)

include_directories("C:\\Program Files\\eigen\\3.3.7")
include_directories("C:\\Users\\Yu\\Desktop\\Grid-Tools\\inc")

add_executable(${PROJECT_NAME} 
	main.cc
	C:/Users/Yu/Desktop/Grid-Tools/src/xf.cc
	C:/Users/Yu/Desktop/Grid-Tools/src/common.cc)
