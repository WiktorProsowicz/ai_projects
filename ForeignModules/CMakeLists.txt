cmake_minimum_required(VERSION "${CMAKE_MINIMUM_REQUIRED_VERSION}")

project(ForeignModules VERSION 1.0.0)

if(${BUILD_TESTS})
    add_subdirectory(googletest)
endif()

add_subdirectory(fmt)