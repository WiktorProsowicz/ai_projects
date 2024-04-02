cmake_minimum_required(VERSION "${CMAKE_MINIMUM_REQUIRED_VERSION}")

project(LoggingLib VERSION 1.0.0)

include("${CMAKE_SOURCE_DIR}/setuputils/Common.cmake")

install_library()

target_link_libraries(${PROJECT_NAME} fmt::fmt)

add_tests()
