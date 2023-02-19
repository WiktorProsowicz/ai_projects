set(CMAKE_MINIMUM_REQUIRED_VERSION "3.22.1")
cmake_minimum_required(VERSION "${CMAKE_MINIMUM_REQUIRED_VERSION}")

project(ai_projects VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CMAKE_C_COMPILER "verision of c compiler" gcc-8.1)
option(CMAKE_CXX_COMPILER "version of c++ compiler" gcc-11.3)

set(COMMON_COMPILE_OPTIONS
    -Wall
)

add_subdirectory(LoggingLib)
add_subdirectory(ForeignModules)

add_subdirectory(LocalSearch)
add_subdirectory(MLCore)
