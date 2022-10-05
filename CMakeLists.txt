cmake_minimum_required(VERSION 3.18)
project(strong_sort_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN_DIR "/usr/include/eigen3/" CACHE PATH "Path to Eigen library")

find_package(OpenCV REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(strong_sort_cpp
    src/tracker.h  src/tracker.cpp
    src/types.h
    src/py.cpp
    src/lsap/rectangular_lsap.cpp
    )

target_include_directories(strong_sort_cpp PRIVATE
    ${EIGEN_DIR}
    src/lsap
    )

target_link_libraries(strong_sort_cpp PRIVATE
    opencv_core
    )
