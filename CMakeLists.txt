set(CMAKE_MINIMUM_REQUIRED_VERSION "3.22.1")
cmake_minimum_required(VERSION "${CMAKE_MINIMUM_REQUIRED_VERSION}")

project(ai_projects VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BUILD_SHARED_LIBS True)

enable_testing()

option(BUILD_TESTS "Build test executables." OFF)

include(GoogleTest)

set(COMMON_COMPILE_OPTIONS
    -Wall
    -std=c++23
    -Wextra 
    -Weffc++ 
    -Wunused 
    -Wnarrowing 
    -Wlogical-op 
    -Wcast-align 
    -Wconversion 
    -Winvalid-pch 
    -Wuseless-cast
    -Wold-style-cast 
    -Wredundant-decls 
    -Wsign-conversion 
    -Wduplicated-cond 
    -Wnull-dereference 
    -Wnon-virtual-dtor 
    -Woverloaded-virtual 
    -Wmissing-declarations 
    -Wmissing-include-dirs 
    -Wmisleading-indentation 
    -Wmissing-format-attribute 
    -Wno-deprecated
)

add_subdirectory(LoggingLib)
add_subdirectory(ForeignModules)

add_subdirectory(MLCore)
add_subdirectory(Utilities)
