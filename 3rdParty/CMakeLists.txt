
find_package(Git QUIET)

# Ensure the submodules are synced.
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}!")
    endif()
endif()

set(FMT_SYSTEM_HEADERS ON CACHE BOOL "Expose fmt headers as system headers")
add_subdirectory(fmt)

if(${ENABLE_TRACY})
    add_subdirectory(tracy)
endif()

if(${BUILD_TESTS})
    add_subdirectory(googletest)
    include(GoogleTest)
endif()