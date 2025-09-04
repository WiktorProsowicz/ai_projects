set(CMAKE_MINIMUM_REQUIRED_VERSION "3.22.1")
cmake_minimum_required(VERSION "${CMAKE_MINIMUM_REQUIRED_VERSION}")

project(ai_projects VERSION 1.0.0 LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Declare options.
# ----------------------------------------------------------------------------

option(BUILD_TESTS "Build test executables." OFF)
option(ENABLE_TRACY "Enable Tracy profiler." OFF)
option(ENABLE_IWYU "Enable include-what-you-use compilation checks" OFF)

# ----------------------------------------------------------------------------
# Set CMake policies.
# ----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------------------------
# Set custom variables.
# ----------------------------------------------------------------------------

set(MAIN_PROJECT_VERSION ${PROJECT_VERSION})

# ----------------------------------------------------------------------------
# Set default properties.
# ----------------------------------------------------------------------------

set(DEFAULT_BUILD_TYPE "Release")
set(DEFAULT_SHARED_LIBS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to default '${DEFAULT_BUILD_TYPE}'.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
endif()

if(NOT DEFINED BUILD_SHARED_LIBS)
    message(STATUS "Setting BUILD_SHARED_LIBS to default: ${DEFAULT_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS ${DEFAULT_SHARED_LIBS} CACHE BOOL "Build using shared libraries." FORCE)
endif()

# ----------------------------------------------------------------------------
# Print config.
# ----------------------------------------------------------------------------

include(CMakePrintHelpers)

cmake_print_variables(
    CMAKE_BUILD_TYPE
    BUILD_SHARED_LIBS
    BUILD_TESTS
    ENABLE_TRACY
)

# ----------------------------------------------------------------------------
# Enable ccache.
# ----------------------------------------------------------------------------

find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()



# ----------------------------------------------------------------------------
# Include all subdirectories.
# ----------------------------------------------------------------------------

include("${CMAKE_SOURCE_DIR}/setuputils/cmake/Common.cmake")

if(${BUILD_TESTS})
    enable_testing()
endif()

add_subdirectory(3rdParty)

add_subdirectory(cpplibs/LoggingLib)
add_subdirectory(cpplibs/MLCore)
add_subdirectory(cpplibs/Utilities)
add_subdirectory(cpplibs/Interfaces)
add_subdirectory(cpplibs/Layers)
add_subdirectory(cpplibs/Datasets)
