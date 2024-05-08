set(CMAKE_MINIMUM_REQUIRED_VERSION "3.22.1")
cmake_minimum_required(VERSION "${CMAKE_MINIMUM_REQUIRED_VERSION}")

project(ai_projects VERSION 1.0.0)

option(BUILD_TESTS "Build test executables." OFF)

include(setuputils/Common.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/ConanFiles ${CMAKE_PREFIX_PATH})

find_package(fmt REQUIRED)
include_directories(SYSTEM ${fmt_INCLUDE_DIRS})

find_package(Tracy REQUIRED)
include_directories(SYSTEM ${Tracy_INCLUDE_DIRS})

if(${BUILD_TESTS})
    enable_testing()

    find_package(GTest REQUIRED)
    include_directories(SYSTEM ${GTest_INCLUDE_DIRS})

    include(GoogleTest)
endif()

add_subdirectory(cpplibs/LoggingLib)
add_subdirectory(cpplibs/MLCore)
add_subdirectory(cpplibs/Utilities)
add_subdirectory(cpplibs/Interfaces)
add_subdirectory(cpplibs/Layers)