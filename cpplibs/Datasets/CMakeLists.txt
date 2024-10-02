cmake_minimum_required(VERSION "${CMAKE_MINIMUM_REQUIRED_VERSION}")

project(Datasets VERSION 1.0.0)

include("${CMAKE_SOURCE_DIR}/setuputils/Common.cmake")

install_library()

target_link_libraries(${PROJECT_NAME} MLCore Interfaces)

add_tests(LoggingLib fmt::fmt)