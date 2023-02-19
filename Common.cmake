# installs library
macro(install_library)

    # get all files for library
    file(GLOB_RECURSE ${PROJECT_NAME}_PUBLIC_HEADERS CONFIGURE_DEPENDS include/**.h*)
    file(GLOB_RECURSE ${PROJECT_NAME}_PRIVATE_HEADERS CONFIGURE_DEPENDS src/**/include/**.h*)
    file(GLOB_RECURSE ${PROJECT_NAME}_SRC src/**.cpp)

    list(FILTER ${PROJECT_NAME}_SRC EXCLUDE REGEX "main.cpp")

    # install library
    add_library(${PROJECT_NAME} SHARED)

    target_sources(${PROJECT_NAME} INTERFACE "${${PROJECT_NAME}_PUBLIC_HEADERS}" "${${PROJECT_NAME}_PRIVATE_HEADERS}"
                                   PRIVATE "${${PROJECT_NAME}_SRC}")

    set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

    target_include_directories(${PROJECT_NAME} PUBLIC
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>)

    target_include_directories(${PROJECT_NAME} PUBLIC include PRIVATE src)

    install(TARGETS ${PROJECT_NAME} DESTINATION lib)

    # get tests files
    file(GLOB TEST_FILES tests/*.cpp)

    # make test executables
    foreach(TEST_FILE IN LISTS TEST_FILES)
        get_filename_component(FILE_NAME "${TEST_FILE}" NAME_WLE)

        add_executable("${FILE_NAME}" "${TEST_FILE}")
        target_link_libraries("${FILE_NAME}" PUBLIC ${PROJECT_NAME} gtest gtest_main)

        set_target_properties(${FILE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test/${PROJECT_NAME}/)
    endforeach()

    target_compile_options(${PROJECT_NAME} PRIVATE ${COMMON_COMPILE_OPTIONS})

endmacro()