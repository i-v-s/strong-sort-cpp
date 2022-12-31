cmake_minimum_required(VERSION 3.20)
project(strongsort
    VERSION 0.0.1
    LANGUAGES CXX)

# Common options

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN_DIR "/usr/include/eigen3/" CACHE PATH "Path to Eigen library")

find_package(OpenCV REQUIRED)

# Strongsort library

add_library(strongsort)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
set_property(TARGET strongsort PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(strongsort
    PRIVATE
        ${EIGEN_DIR}
        src/lsap
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/;${CMAKE_CURRENT_BINARY_DIR}/>"
        "$<INSTALL_INTERFACE:include/>"
    )

target_sources(strongsort
    PRIVATE src/tracker.cpp src/types.cpp src/lsap/rectangular_lsap.cpp src/tracker.h src/types.h
)

target_link_libraries(strongsort PRIVATE opencv_core)


# Install library

include(GNUInstallDirs)

install(
    TARGETS strongsort
    EXPORT strongsort-config
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

install(
    EXPORT ${PROJECT_NAME}-config
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

install(FILES src/tracker.h src/types.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})


# Python bindings

add_subdirectory(pybind11)
pybind11_add_module(strongsort_py src/py.cpp)
target_include_directories(strongsort_py PRIVATE ${EIGEN_DIR})
target_link_libraries(strongsort_py PRIVATE opencv_core strongsort)


# Build .deb package

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Igor Smirnov")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libopencv-core-dev")
include(CPack)
