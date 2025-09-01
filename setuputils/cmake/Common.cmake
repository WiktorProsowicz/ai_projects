# ***********************************************************************
#  Collectes files in current directory and creates a library from them.
# ***********************************************************************
macro(install_library)

    # get all files for library
    file(GLOB_RECURSE ${PROJECT_NAME}_PUBLIC_HEADERS CONFIGURE_DEPENDS include/**.h*)
    file(GLOB_RECURSE ${PROJECT_NAME}_PRIVATE_HEADERS CONFIGURE_DEPENDS src/**/include/**.h*)
    file(GLOB_RECURSE ${PROJECT_NAME}_SRC src/**.cpp)

    list(FILTER ${PROJECT_NAME}_SRC EXCLUDE REGEX "main.cpp")

    # install library
    add_library(${PROJECT_NAME} SHARED)

    target_sources(${PROJECT_NAME} PUBLIC "${${PROJECT_NAME}_PUBLIC_HEADERS}"
                                   PRIVATE "${${PROJECT_NAME}_SRC}" "${${PROJECT_NAME}_PRIVATE_HEADERS}")

    set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
        target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    endif()

    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

    target_compile_options(${PROJECT_NAME} PRIVATE ${COMMON_COMPILE_OPTIONS})

    # adding executable
    add_executable_for_lib()

endmacro()



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
    endif()
endmacro()


# ********************************************************************
#  Looks for tests files and creates an executable for each of them.
#  Links given libraries to the tests.
# ********************************************************************
macro(add_tests)
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
                target_link_libraries("${TEST_NAME}" PUBLIC ${ARGV})
            endif()

            set_target_properties("${TEST_NAME}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test/${PROJECT_NAME}/)

            target_compile_options("${TEST_NAME}" PRIVATE ${COMMON_COMPILE_OPTIONS_PERMISSIVE})

            target_compile_definitions("${TEST_NAME}" PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/res")

            gtest_discover_tests("${TEST_NAME}")

        endforeach()

    endif()
endmacro()

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
