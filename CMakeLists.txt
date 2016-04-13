cmake_minimum_required(VERSION 2.8)
project(ipiano)

# set(CMAKE_CXX_COMPILER "g++-4.8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic -Wall -O4 -g")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(lib_ipiano)
add_subdirectory(signal_denoising_cli)
add_subdirectory(image_denoising_cli)
add_subdirectory(phase_field_cli)
add_subdirectory(phase_field_color_cli)