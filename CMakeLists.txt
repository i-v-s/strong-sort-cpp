cmake_minimum_required(VERSION 3.20)
project(strongsort LANGUAGES CXX)


# Common options

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN_DIR "/usr/include/eigen3/" CACHE PATH "Path to Eigen library")

find_package(OpenCV REQUIRED)


# Strongsort library

add_library(strongsort STATIC)

set_property(TARGET strongsort PROPERTY PUBLIC_HEADER "src/types.h;src/tracker.h")
set_property(TARGET strongsort PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(strongsort
    PRIVATE
        ${EIGEN_DIR}
        src/lsap
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:include>"
    )

target_sources(strongsort
    PRIVATE src/tracker.cpp src/lsap/rectangular_lsap.cpp
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/tracker.h> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/types.h>
)

target_link_libraries(strongsort PRIVATE
    opencv_core
    )


# Install library

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

configure_package_config_file(strongsortConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/strongsortConfig.cmake
    INSTALL_DESTINATION lib/strongsort/cmake
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/strongsortConfigVersion.cmake
    VERSION 0.0.1
    COMPATIBILITY SameMajorVersion
    )

if(CMAKE_INSTALL_PREFIX)
  set(strongsort_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/strongsort)
endif()

install(TARGETS strongsort
    EXPORT strongsortTargets
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/strongsort
    )

install(EXPORT strongsortTargets
    FILE strongsortTargets.cmake
    NAMESPACE strongsort::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/strongsort
    )

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/strongsortConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/strongsortConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/strongsort
    )


# Python bindings

add_subdirectory(pybind11)
pybind11_add_module(strongsort_py
    src/py.cpp
    )

target_include_directories(strongsort_py PRIVATE
    ${EIGEN_DIR}
    )

target_link_libraries(strongsort_py PRIVATE
    opencv_core
    strongsort
    )
