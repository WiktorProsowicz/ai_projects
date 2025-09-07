# ***********************************************************************
#  Collectes files in current directory and creates a library from them.
# ***********************************************************************
function(aiprojects_add_library)

    cmake_parse_arguments(PARSED_ARGS "" "LIBRARY_NAME" "" ${ARGN})

    set(LIBRARY_NAME "${PARSED_ARGS_LIBRARY_NAME}")

    # Get all files for library
    file(GLOB_RECURSE ${LIBRARY_NAME}_PUBLIC_HEADERS CONFIGURE_DEPENDS include/**.h*)
    file(GLOB_RECURSE ${LIBRARY_NAME}_PRIVATE_HEADERS CONFIGURE_DEPENDS src/**/include/**.h*)
    file(GLOB_RECURSE ${LIBRARY_NAME}_SRC src/**.cpp)

    # Exclude cpp for executable.
    list(FILTER ${LIBRARY_NAME}_SRC EXCLUDE REGEX "main.cpp")

    add_library(${LIBRARY_NAME})

    target_sources(${LIBRARY_NAME} PUBLIC "${${LIBRARY_NAME}_PUBLIC_HEADERS}"
                                   PRIVATE "${${LIBRARY_NAME}_SRC}" "${${LIBRARY_NAME}_PRIVATE_HEADERS}")

    set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)

    set_target_properties(${LIBRARY_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
        target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    endif()

    target_include_directories(${LIBRARY_NAME}
                                PUBLIC
                                    "$<INSTALL_INTERFACE:include>"
                                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

    target_compile_options(${LIBRARY_NAME} PRIVATE ${COMMON_COMPILE_OPTIONS})

    if(ENABLE_IWYU)
        aiprojects_setup_iwyu_for_target(${LIBRARY_NAME})
    endif()

    if(ENABLE_CLANG_TIDY)
        aiprojects_setup_clang_tidy_for_target(${LIBRARY_NAME})
    endif()

    if(ENABLE_CPPCHECK)
        aiprojects_setup_cppcheck_for_target(${LIBRARY_NAME})
    endif()

    # adding executable
    add_executable_for_lib()

endfunction()



# ****************************************
#  Creates an executable for the library.
# ****************************************
macro(add_executable_for_lib)

    file(GLOB_RECURSE ${PROJECT_NAME}_MAIN src/main.cpp)

    if(${PROJECT_NAME}_MAIN)
        add_executable(${PROJECT_NAME}Executable ${${PROJECT_NAME}_MAIN})

        target_link_libraries(${PROJECT_NAME}Executable PUBLIC ${PROJECT_NAME})

        set_target_properties(${PROJECT_NAME}Executable PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

        target_compile_options(${PROJECT_NAME}Executable PRIVATE ${COMMON_COMPILE_OPTIONS})

        if(ENABLE_IWYU)
            aiprojects_setup_iwyu_for_target(${PROJECT_NAME}Executable)
        endif()

        if(ENABLE_CLANG_TIDY)
            aiprojects_setup_clang_tidy_for_target(${PROJECT_NAME}Executable)
        endif()

        if(ENABLE_CPPCHECK)
            aiprojects_setup_cppcheck_for_target(${PROJECT_NAME}Executable)
        endif()
    endif()
endmacro()


# ********************************************************************
#  Looks for tests files and creates an executable for each of them.
#  Links given libraries to the tests.
# ********************************************************************
function(add_tests)

    cmake_parse_arguments(PARSED_ARGS "" "" "LINKED_LIBRARIES" ${ARGN})

    if(${BUILD_TESTS})

        # get tests files
        file(GLOB TEST_FILES tests/*.cpp)

        # make test executables
        foreach(TEST_FILE IN LISTS TEST_FILES)
            get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WLE)

            add_executable("${TEST_NAME}" "${TEST_FILE}")

            if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
                target_include_directories("${TEST_NAME}" PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
            endif()

            target_link_libraries("${TEST_NAME}" PUBLIC ${PROJECT_NAME} GTest::gtest GTest::gmock GTest::gtest_main)

            if(${ARGC} GREATER 0)
                target_link_libraries("${TEST_NAME}" PUBLIC ${PARSED_ARGS_LINKED_LIBRARIES})
            endif()

            set_target_properties("${TEST_NAME}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test/${PROJECT_NAME}/)

            target_compile_options("${TEST_NAME}" PRIVATE ${COMMON_COMPILE_OPTIONS_PERMISSIVE})

            target_compile_definitions("${TEST_NAME}" PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/res")

            if(ENABLE_IWYU)
                aiprojects_setup_iwyu_for_target("${TEST_NAME}")
            endif()

            gtest_discover_tests("${TEST_NAME}")

        endforeach()

    endif()
endfunction()

# ********************************************************************
#  Wraps add_subdirectory for supressing messages from external subdirs.
# ********************************************************************
macro(add_subdirectory_supress_messages DIR)
    set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
    set(CMAKE_MESSAGE_LOG_LEVEL "WARNING")
    add_subdirectory("${DIR}")
    set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})
endmacro()


# ********************************************************************
#  Sets up include-what-you-use for the given target if enabled.
# ********************************************************************
function(aiprojects_setup_iwyu_for_target TARGET)

    set(iwyu_options
        "-Xiwyu" "--quoted_includes_first"
        "-Xiwyu" "--mapping_file=${IWYU_MAPPING_FILE}"
        "-w"
    )

    set_property(TARGET ${TARGET} PROPERTY CXX_INCLUDE_WHAT_YOU_USE "${iwyu_path};${iwyu_options}")

endfunction()

# ********************************************************************
#  Enables clang-tidy checks for the given target if enabled.
# ********************************************************************
function(aiprojects_setup_clang_tidy_for_target TARGET)

    set(clang_tidy_options
        "--use-color"
        "--config-file" "${CLANG_TIDY_CONFIG_FILE}"
        "-extra-arg" "-Wno-unknown-warning-option"
    )

    set_property(TARGET ${TARGET} PROPERTY CXX_CLANG_TIDY "${clang_tidy_path};${clang_tidy_options}")
endfunction()

# ********************************************************************
#  Enables cppcheck checks for the given target if enabled.
# ********************************************************************
function(aiprojects_setup_cppcheck_for_target TARGET)

    set(cppcheck_options
        "--quiet"
        "--enable=all"
        "--inline-suppr"
        "--std=c++23"
        "--check-level=exhaustive"
        "--suppress=unusedFunction"
        "--suppress=missingIncludeSystem"
        "--suppress=checkersReport"
        "--suppress=unmatchedSuppression"
    )

    set_property(TARGET ${TARGET} PROPERTY CXX_CPPCHECK "${cppcheck_path};${cppcheck_options}")
endfunction()


# ***********************************************
# Used for building libraries and executables
# ***********************************************
set(COMMON_COMPILE_OPTIONS
    -Wall
    -Werror
    -Wextra
    -Weffc++
    -Wunused
    -Wpedantic
    -Wnarrowing
    -Wcast-align
    -Wreturn-type
    -Wconversion
    -Wlogical-op
    -Wtype-limits
    -Winvalid-pch
    -Wsign-compare
    -Wuseless-cast
    -Wunused-result
    -Wold-style-cast
    -Wredundant-decls
    -Wduplicated-cond
    -Wnull-dereference
    -Wunused-parameter
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wstringop-truncation
    -Wduplicated-branches
    -Wimplicit-fallthrough
    -Wmissing-declarations
    -Wmissing-include-dirs
    -Wmisleading-indentation
    -Wmissing-format-attribute
)

# ***********************************************
# Used for building test executables
# ***********************************************
set(COMMON_COMPILE_OPTIONS_PERMISSIVE
    -Wno-all
    -Wno-error
    -Wno-extra
    -Wno-effc++
    -Wno-unused
    -Wno-pedantic
    -Wno-narrowing
    -Wno-cast-align
    -Wno-return-type
    -Wno-conversion
    -Wno-logical-op
    -Wno-type-limits
    -Wno-invalid-pch
    -Wno-sign-compare
    -Wno-useless-cast
    -Wno-unused-result
    -Wno-old-style-cast
    -Wno-redundant-decls
    -Wno-duplicated-cond
    -Wno-null-dereference
    -Wno-unused-parameter
    -Wno-non-virtual-dtor
    -Wno-overloaded-virtual
    -Wno-stringop-truncation
    -Wno-duplicated-branches
    -Wno-implicit-fallthrough
    -Wno-missing-declarations
    -Wno-missing-include-dirs
    -Wno-misleading-indentation
    -Wno-missing-format-attribute
)
