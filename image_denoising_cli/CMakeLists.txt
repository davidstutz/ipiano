cmake_minimum_required(VERSION 2.8)
project(ipiano)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)

include_directories(../lib_ipiano ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} 
        ${EIGEN3_INCLUDE_DIR} ${GLOG_INCLUDE_DIRS})
add_executable(image_denoising_cli main.cpp)
target_link_libraries(image_denoising_cli ${Boost_LIBRARIES} ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES})