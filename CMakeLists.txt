set(CMAKE_MINIMUM_REQUIRED_VERSION "3.22.1")
cmake_minimum_required(VERSION "${CMAKE_MINIMUM_REQUIRED_VERSION}")

project(ai_projects VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(LoggingLib)
add_subdirectory(ForeignModules)

add_subdirectory(LocalSearch)
add_subdirectory(MLCore)
